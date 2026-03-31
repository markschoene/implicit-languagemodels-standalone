from functools import partial

import torch
from torch import Tensor, nn

from .evaluation import EvaluationConfig
from .solver import reset_deq, jac_reg, _spectral_radius


class ImplicitMixin(nn.Module):
    """
    Wrapper class for implicit DEQ models.
    Requires the following attributes:
        - self.deq: DEQ object
        - self.layers: list of DEQ layers
        - self.injection: injection layer
        - self.injection_norm: injection normalization layer
        - self.norm_f: normalization function
        - self.training: training flag
        - self.pretrain: pretraining flag
        - self.pretrain_steps: number of pretraining steps
        - self.pretrain_counter: counter for pretraining steps
    """

    def _init_eval_config(self):
        """Initialize the evaluation config with defaults from DEQ params."""
        self.eval_config = EvaluationConfig()

    def sequential_evaluation(self):
        """Switch to sequential (token-by-token) evaluation mode."""
        self.eval_config.mode = "sequential"

    def simultaneous_evaluation(self):
        """Switch to simultaneous (parallel) evaluation mode."""
        self.eval_config.mode = "simultaneous"

    def convergence_warning(self, idx, diff, steps):
        if steps >= self._eval_max_iter - 1:
            print(
                    f"Token {idx:4d} Warning: DEQ did not converge with rel diff {diff:.3f} after {steps} steps", "red"
            )

    @property
    def _eval_max_iter(self) -> int:
        """Effective max iterations for eval, respecting EvaluationConfig overrides."""
        if self.eval_config.max_iter is not None:
            return self.eval_config.max_iter
        return self.f_thres

    @property
    def _eval_tol(self) -> float:
        """Effective convergence tolerance for eval, respecting EvaluationConfig overrides."""
        if self.eval_config.tol is not None:
            return self.eval_config.tol
        return self.f_tol

    @property
    def _eval_momentum(self) -> float:
        """Effective momentum for eval, respecting EvaluationConfig overrides."""
        if self.eval_config.momentum is not None:
            return self.eval_config.momentum
        return self.beta

    def is_pretraining(self):
        if self.pretrain and self.pretrain_counter < self.pretrain_steps:
            if self.training:
                self.pretrain_counter += 1
            return True
        elif self.pretrain and self.pretrain_counter == self.pretrain_steps:
            self.pretrain = False
            print(f"Pretraining finished after {self.pretrain_counter} steps")
            torch.cuda.empty_cache()
            return False
        else:
            return False

    def func(self, z: Tensor, u: Tensor, mixer_kwargs: dict = {}) -> Tensor:
        """defines the DEQ function whose roots we want to find"""
        # create a new list since we do not want to overwrite the old cache in place
        # since DEQs should get the last injected at every iteration
        for l, layer in enumerate(self.layers):
            z = layer(z, u, **mixer_kwargs)

        z = self.norm_f(z)
        # return new_cache. The CacheRetainer class will retain the cache from the last iteration
        return z

    def implicit_forward(self, hidden_states: Tensor, mixer_kwargs: dict = {}) -> tuple[Tensor, Tensor, dict]:
        """
        DEQ forward pass.

        Args:
            hidden_states: input of embedding vectors (B, L, D)
            mixer_kwargs: additional arguments for the DEQ function
        Returns:
            output: output of the DEQ model (B, L, D)
            jac_loss: Jacobian regularization loss
            log_dict: dictionary with logging information
        """
        # Reset dropout masks before each forward pass
        reset_deq(self)

        # compute injected inputs
        injected_states = self.injection_norm(hidden_states)
        injected_states = self.injection(injected_states)
        zs = torch.zeros_like(hidden_states)

        # compute the DEQ output
        if self.training:
            # training always uses simultaneous mode (or pretrain)
            if self.is_pretraining():
                output, log_dict = self._weighttied_parallel_forward(injected_states, zs, n_iter=self.pretrain_iter, mixer_kwargs=mixer_kwargs)
            else:
                output, log_dict = self._simultaneous_forward(injected_states, zs, mixer_kwargs)
        else:
            # eval mode: branch on eval_config
            if self.eval_config.mode == "sequential":
                output, log_dict = self._sequential_forward(injected_states, zs, mixer_kwargs)
            else:
                output, log_dict = self._simultaneous_forward(injected_states, zs, mixer_kwargs)

        # compute the Jacobian regularization loss
        # torch runs into issues when trying to compute gradients while in eval mode due to torch.no_grad()
        compute_jac_loss = self.jac_loss_weight > 0.0 and torch.rand(1).item() < self.jac_loss_freq
        if self.training and compute_jac_loss:
            jac_loss = self.jac_loss_weight * jac_reg(self.func(output, injected_states), output)
        else:
            jac_loss = torch.tensor(0.0).to(output.device)
        log_dict["jac_loss"] = jac_loss

        return output, jac_loss, log_dict

    def _weighttied_parallel_forward(
        self, x: Tensor, zs: Tensor, n_iter: int, indexing: bool = False, mixer_kwargs: dict = {}
    ) -> tuple[Tensor, dict]:
        """
        Formerly _pretrain_forward.
        """

        zs = self.func(zs, x, mixer_kwargs=mixer_kwargs)
        indexing_list = [] if not indexing else [zs]
        # run the DEQ for a fixed number of steps
        for _ in range(n_iter-1):
            z_f = self.func(zs, x, mixer_kwargs=mixer_kwargs)
            zs = (1.0-self.beta) * zs + self.beta * z_f
            if indexing:
                indexing_list.append(zs)

        log_dict = {
            "steps": n_iter,
            "indexing_list": indexing_list,
        }

        return zs, log_dict

    def _simultaneous_forward(self, x: Tensor, zs: Tensor, mixer_kwargs: dict) -> tuple[Tensor, dict]:
        sradius_mode = self.eval_config.spectral_radius if not self.training else self.sradius_mode

        func_to_use = partial(self.func, u=x, mixer_kwargs=mixer_kwargs)
        z_out, info = self.deq(func_to_use, zs, sradius_mode=sradius_mode)

        output = z_out[-1]

        log_dict = {
            "abs diff": info["abs_lowest"].mean(),
            "rel diff": info["rel_lowest"].mean(),
            "steps": info["nstep"].mean(),
        }
        if not self.training and sradius_mode:
            log_dict["spectral radius"] = info["sradius"].mean().to(output)

        return output, log_dict


    def _sequential_step(self, u: Tensor, z: Tensor, mixer_kwargs: dict, is_first_token=False) -> tuple[Tensor, dict]:
        """"
        Args:
        u: injected input (B, D_in_proj)
        z: the initial state (B, D)
        mixer_kwargs: dictionary carrying additional parameters (including kv update flag)
        is_first_token: if True, we use a more lenient tolerance (f_tol = 0.2) to help convergence.

        Returns:
        final_output: final output of the DEQ model (B, D)
        log_dict: a dictionary with convergence metrics (abs diff, rel diff, iterations count)
        """

        def relative_diff(a, b) -> Tensor:
            return torch.norm(b - a) / (1e-6 + torch.norm(a))

        def absolute_diff(a, b) -> Tensor:
            return torch.norm(b - a)

        max_iter = self._eval_max_iter
        f_tol = self._eval_tol
        momentum = self._eval_momentum

        # For the iterative loop, disable kv updates.
        mixer_kwargs["skip_kv_update"] = True

        diff_abs = torch.tensor(1e9, device=z.device)
        diff_rel = torch.tensor(1e9, device=z.device)

        for count in range(max_iter - 1):

            z_next = self.func(z, u, mixer_kwargs)
            diff_rel = relative_diff(z, z_next)
            diff_abs = absolute_diff(z, z_next)
            # update state with momentum
            z = (1 - momentum) * z + momentum * z_next

            # Use a lenient tolerance for the first token.
            tol = 0.2 if is_first_token else f_tol
            if diff_rel < tol:
                break

        # Re-enable key/value updates for the final call.
        mixer_kwargs["skip_kv_update"] = False
        final_output = self.func(z, u, mixer_kwargs)

        steps = torch.tensor(count + 1, dtype=torch.float32, device=z.device)
        log_dict = {
            "abs diff": diff_abs,
            "rel diff": diff_rel,
            "steps": steps,
        }

        # compute spectral radius if requested
        if self.eval_config.spectral_radius:
            mixer_kwargs["skip_kv_update"] = True
            func_for_sradius = partial(self.func, u=u, mixer_kwargs=mixer_kwargs)
            sradius = _spectral_radius(func_for_sradius, final_output)
            mixer_kwargs["skip_kv_update"] = False
            log_dict["spectral radius"] = sradius.mean().to(final_output)

        return final_output, log_dict

    def _sequential_forward(self, x: Tensor, zs: Tensor, mixer_kwargs: dict) -> tuple[Tensor, list[dict]]:
        """
        Run the DEQ model sequentially for each token in the input sequence.
        Args:
            x: input of embedding vectors (B, D)

        Returns:
            output: output of the DEQ model (B, L, D)
            log_dict: dictionary with logging information
        """
        assert self._eval_max_iter > 0, "max_iter must be at least one"
        output, log_dict = self._sequential_step(
            x, zs, mixer_kwargs, is_first_token=False
        )

        return output, log_dict
