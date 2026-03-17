"""
Standalone fixed-point solver with phantom gradient backward pass.

Replaces the TorchDEQ dependency with a minimal implementation focused on:
- Fixed-point iteration with momentum (beta)
- Phantom gradient for differentiable backward pass (tau)
- Variational dropout (fixed mask across solver iterations)
- Weight normalization via PyTorch built-in parametrizations
- Jacobian regularization (Hutchinson estimator)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


# ---------------------------------------------------------------------------
# Variational Dropout
# ---------------------------------------------------------------------------

class _VariationalDropoutNd(nn.Module):
    """
    Base class for variational dropout layers.

    A single dropout mask is generated once per training iteration and applied
    consistently across all solver steps, preserving dynamics during fixed-point
    iteration. Call ``reset_dropout(model)`` at the start of each iteration.
    """

    def __init__(self, dropout=0.5):
        super().__init__()
        if dropout < 0 or dropout > 1:
            raise ValueError(
                f"dropout probability must be between 0 and 1, got {dropout}"
            )
        self.dropout = dropout
        self.mask = None

    def reset_mask(self, x):
        raise NotImplementedError

    def forward(self, x):
        if not self.training or self.dropout == 0.0:
            return x
        if self.mask is None:
            self.reset_mask(x)
        mask = self.mask.expand_as(x)
        return mask * x


def reset_dropout(model):
    """Reset variational dropout masks for all layers in the model."""
    for module in model.modules():
        if isinstance(module, _VariationalDropoutNd):
            module.mask = None


# ---------------------------------------------------------------------------
# Weight Normalization (PyTorch built-in)
# ---------------------------------------------------------------------------

def apply_weight_norm(model):
    """Apply weight normalization to all nn.Linear layers in the model."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weight_norm(module, name="weight")


def remove_weight_norm(model):
    """Remove weight normalization from all nn.Linear layers in the model."""
    from torch.nn.utils.parametrize import remove_parametrizations

    for module in model.modules():
        if isinstance(module, nn.Linear):
            try:
                remove_parametrizations(module, "weight")
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Reset convenience
# ---------------------------------------------------------------------------

def reset_deq(model):
    """Reset state before each forward pass (dropout masks only).

    Weight normalization via PyTorch parametrizations auto-recomputes,
    so no explicit reset is needed for weights.
    """
    reset_dropout(model)


# ---------------------------------------------------------------------------
# Fixed-point iteration solver
# ---------------------------------------------------------------------------

def fixed_point_iter(func, x0, max_iter, tol, beta=1.0, stop_mode="rel"):
    """
    Fixed-point iteration with momentum.

    Computes: x_{k+1} = beta * f(x_k) + (1 - beta) * x_k

    Args:
        func: The function f for which we seek x* = f(x*).
        x0: Initial guess, shape (B, ...).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
        beta: Momentum/damping factor for the solver.
        stop_mode: ``'abs'`` or ``'rel'`` convergence criterion.

    Returns:
        (best_estimate, info) where info contains abs_lowest, rel_lowest, nstep.
    """
    bsz = x0.shape[0]

    best_abs = torch.full((bsz,), float("inf"), device=x0.device)
    best_rel = torch.full((bsz,), float("inf"), device=x0.device)
    best_x = x0
    best_step = torch.zeros(bsz, device=x0.device)

    x = x0
    for k in range(max_iter):
        fx = beta * func(x) + (1 - beta) * x

        # Residual metrics
        diff = fx - x
        abs_diff = diff.flatten(1).norm(dim=1)
        rel_diff = abs_diff / (fx.flatten(1).norm(dim=1) + 1e-9)

        # Track best per-sample
        metric = rel_diff if stop_mode == "rel" else abs_diff
        improved = metric < (best_rel if stop_mode == "rel" else best_abs)
        best_abs = torch.where(improved, abs_diff, best_abs)
        best_rel = torch.where(improved, rel_diff, best_rel)
        best_step = torch.where(improved, torch.tensor(k + 1.0, device=x0.device), best_step)
        # Update best estimate for improved samples
        best_x = torch.where(improved.unsqueeze(-1).expand_as(fx), fx, best_x)

        x = fx

        # Early stopping when all samples converged
        if metric.max() < tol:
            break

    info = {
        "abs_lowest": best_abs,
        "rel_lowest": best_rel,
        "nstep": best_step,
    }
    return best_x, info


# ---------------------------------------------------------------------------
# Phantom gradient
# ---------------------------------------------------------------------------

def phantom_grad(func, z_pred, n_steps, tau):
    """
    Phantom gradient: unroll ``n_steps`` of ``func`` with damping ``tau``,
    building a differentiable autograd graph for the backward pass.

    Args:
        func: The fixed-point function f.
        z_pred: Detached fixed-point estimate to start from.
        n_steps: Number of unrolling steps.
        tau: Phantom gradient damping factor.

    Returns:
        Final state with autograd graph attached.
    """
    for _ in range(n_steps):
        z_pred = tau * func(z_pred) + (1 - tau) * z_pred
    return z_pred


# ---------------------------------------------------------------------------
# FixedPointSolver (replaces DEQSliced / get_deq)
# ---------------------------------------------------------------------------

class FixedPointSolver(nn.Module):
    """
    Fixed-point solver with phantom gradient backward pass.

    During training: runs the solver to convergence without gradient tracking,
    then applies phantom gradient on the final converged state.

    During eval: runs the solver for ``eval_f_max_iter`` steps.

    Args:
        f_max_iter: Max forward iterations during training.
        f_tol: Convergence tolerance.
        beta: Solver momentum factor.
        tau: Phantom gradient damping factor.
        grad_steps: Number of phantom gradient unrolling steps.
        eval_factor: Multiplier for eval iterations (eval_iters = f_max_iter * eval_factor).
        eval_f_max_iter: Override for eval max iterations (0 = use eval_factor).
    """

    def __init__(self, f_max_iter, f_tol, beta, tau, grad_steps,
                 eval_factor=1.0, eval_f_max_iter=0):
        super().__init__()
        self.f_max_iter = f_max_iter
        self.f_tol = f_tol
        self.beta = beta
        self.tau = tau
        self.grad_steps = grad_steps
        self.eval_f_max_iter = (
            eval_f_max_iter if eval_f_max_iter > 0
            else int(f_max_iter * eval_factor)
        )

    def forward(self, func, z_init, sradius_mode=False):
        """
        Args:
            func: The fixed-point function f(z).
            z_init: Initial state, shape (B, L, D).
            sradius_mode: If True, compute spectral radius at eval time.

        Returns:
            (z_out_list, info): list containing the output tensor, and solver stats dict.
        """
        if self.training:
            with torch.no_grad():
                z_star, info = fixed_point_iter(
                    func, z_init, self.f_max_iter, self.f_tol, beta=self.beta
                )
            z_star = z_star.detach()
            z_out = phantom_grad(func, z_star, self.grad_steps, tau=self.tau)
            return [z_out], info
        else:
            with torch.no_grad():
                z_star, info = fixed_point_iter(
                    func, z_init, self.eval_f_max_iter, self.f_tol, beta=self.beta
                )
            if sradius_mode:
                info["sradius"] = _spectral_radius(func, z_star)
            else:
                info["sradius"] = torch.zeros(1, device=z_star.device)
            return [z_star], info


# ---------------------------------------------------------------------------
# Jacobian regularization & spectral radius
# ---------------------------------------------------------------------------

def jac_reg(f0, z0, vecs=1, create_graph=True):
    """
    Estimate tr(J^T J) via Hutchinson estimator.

    Args:
        f0: Output of f(z0).
        z0: Input to f.
        vecs: Number of random probe vectors.
        create_graph: Whether to create backward graph.

    Returns:
        Scalar tensor with the (shape-normalized) Jacobian loss.
    """
    result = 0
    for _ in range(vecs):
        v = torch.randn(*z0.shape).to(z0)
        vJ = torch.autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
        result += vJ.norm() ** 2
    return result / vecs / np.prod(z0.shape)


def _spectral_radius(func, z_star, n_iters=100):
    """Estimate spectral radius of the Jacobian at z_star via power method."""
    with torch.enable_grad():
        z = z_star.detach().requires_grad_()
        fz = func(z)

    evector = torch.randn_like(z_star)
    bsz = evector.shape[0]
    for i in range(n_iters):
        vTJ = torch.autograd.grad(
            fz, z, evector, retain_graph=(i < n_iters - 1), create_graph=False
        )[0]
        evalue = (
            (vTJ * evector).reshape(bsz, -1).sum(1)
            / (evector * evector).reshape(bsz, -1).sum(1)
        )
        evector = (
            vTJ.reshape(bsz, -1) / vTJ.reshape(bsz, -1).norm(dim=1, keepdim=True)
        ).reshape_as(z_star)
    return torch.abs(evalue)
