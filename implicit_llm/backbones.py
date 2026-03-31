"""
Common backbones for the implicit LLMs defined in
- implicit_llm/implicit_llama/modeling_llama.py
- implicit_llm/implicit_mamba2/modeling_mamba2.py
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Optional

import torch
from torch import Tensor

try:
    from torch.nn import RMSNorm
except ImportError:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm

import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from .modules.dropout import reset_dropout
from .solver import FixedPointSolver, apply_weight_norm

from .implicit import ImplicitMixin
from .modules.block import create_block


@dataclass
class ModelConfig:
    d_model: int | None = None
    d_intermediate: int | None = None
    n_layer: int | None = None
    vocab_size: int | None = None
    block_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int | None = None
    tie_embeddings: bool = True


@dataclass
class ImplicitBaseModelOutputWithPast(BaseModelOutputWithPast):
    implicit_metrics: Optional[Dict[str, Tensor]] | None = None
    jac_loss: Optional[Tensor] | None = None


class BaseModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_inner: int,
        pre_norm: bool,
        block_type: str,
        block_cfg: Optional[Dict[str, Any]] = None,
        rms_norm: bool = False,
        dropout: float = 0.0,
        mlp_type: str = "gated_mlp",
        norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert block_type in [
            "mamba2",
            "transformer",
        ], f"Unsupported block type: {block_type}. Supported types are 'mamba2' and 'transformer'."
        self.block_type = block_type
        self.layers = nn.ModuleList(
            [
                create_block(
                    block_type=block_type,
                    d_model=d_model,
                    d_intermediate=d_inner,
                    block_cfg=block_cfg,
                    dropout=dropout,
                    rms_norm=rms_norm,
                    pre_norm=pre_norm,
                    mlp_type=mlp_type,
                    residual_in_fp32=residual_in_fp32,
                    norm_epsilon=norm_epsilon,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        if not pre_norm:
            self.norm_f = nn.Identity()
        else:
            self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon, **factory_kwargs)

        # set some attributes which we need below
        self.input_projection_dim = self.get_input_proj_dim()
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_inner = d_inner
        self.block_cfg = block_cfg
        self.rms_norm = rms_norm
        self.d_intermediate = d_inner

    def get_input_proj_dim(self) -> int:
        dim = self.layers[0].mixer.d_in_proj
        return dim


def _init_weights(
    module,
    n_layer,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in [
                "out_proj.weight",
                "fc2.weight",
                "out_state_proj.weight",
                "inter_state_proj.weight",
            ]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class ExplicitModel(BaseModel):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_inner: int,
        pre_norm: bool,
        block_type: str = "mamba2",
        block_cfg: Optional[Dict[str, Any]] = None,
        rms_norm: bool = True,
        dropout: float = 0.0,
        initializer_cfg: Optional[Dict[str, Any]] = None,
        norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        **kwargs,
    ):
        kwargs.pop("deq_params", None)
        super().__init__(
            block_type=block_type,
            d_model=d_model,
            n_layer=n_layer,
            d_inner=d_inner,
            pre_norm=pre_norm,
            block_cfg=block_cfg,
            rms_norm=rms_norm,
            dropout=dropout,
            norm_epsilon=norm_epsilon,
            residual_in_fp32=residual_in_fp32,
            **kwargs,
        )
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_inner == 0 else 2,  # 2 if we have MLP
            )
        )

    def forward(self, hidden_states: Tensor, mixer_kwargs: Dict) -> BaseModelOutputWithPast:
        reset_dropout(self)

        for layer in self.layers:
            hidden_states = layer(hidden_states, injected_inputs=None, **mixer_kwargs)
        hidden_states = self.norm_f(hidden_states)

        output = ImplicitBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
            implicit_metrics=None,
            jac_loss=None,
        )
        return output


class ImplicitModel(BaseModel, ImplicitMixin):
    """
    Model class for implicit models, that is, models built on the DEQ paradigm.
    """

    def __init__(
        self,
        deq_params,
        d_model: int,
        n_layer: int,
        d_inner: int,
        pre_norm: bool,
        pretrain_steps: int = 0,
        pretrain_iter: int = 4,
        block_type: str = "mamba2",
        block_cfg: Optional[Dict[str, Any]] = None,
        rms_norm: bool = True,
        dropout: float = 0.0,
        init_gain: float = 0.5,
        initializer_cfg: Optional[Dict[str, Any]] = None,
        ema_alpha=None,
        norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        do_weight_norm: bool = True,
        init_z_mode: str = None,
        **kwargs,
    ):
        super().__init__(
            block_type=block_type,
            d_model=d_model,
            n_layer=n_layer,
            d_inner=d_inner,
            pre_norm=pre_norm,
            block_cfg=block_cfg,
            rms_norm=rms_norm,
            dropout=dropout,
            norm_epsilon=norm_epsilon,
            residual_in_fp32=residual_in_fp32,
            **kwargs,
        )
        if init_z_mode == "input":
            self.set_exp2implicit()
        else:
            self.init_z_mode = None

        self.do_weight_norm = do_weight_norm
        print(f"Default z_init is set to {self.init_z_mode}. For explicit 2 implicit conversion change it if needed.")

        # define input injection for Mamba model
        self.injection = nn.Linear(self.d_model, self.get_input_proj_dim())
        self.injection_norm = RMSNorm(self.d_model, eps=1e-5)

        # set the fixed-point solver
        self.deq_params = deq_params
        self.deq_args = {
            **deq_params.get("solver", {}),
            **deq_params.get("training", {}),
            **deq_params.get("regularization", {}),
        }
        self.deq = FixedPointSolver(
            f_max_iter=self.deq_args["f_max_iter"],
            f_tol=self.deq_args["f_tol"],
            beta=self.deq_args.get("beta", self.deq_args.get("tau", 0.8)),
            tau=self.deq_args.get("tau", 0.8),
            grad_steps=self.deq_args["grad"],
            eval_factor=self.deq_args.get("eval_factor", 1.0),
            eval_f_max_iter=self.deq_args.get("eval_f_max_iter", 0),
        )

        # get relevant attributes from config
        self.jac_loss_freq = self.deq_args.get("jac_loss_freq", 0.0)
        self.jac_loss_weight = self.deq_args.get("jac_loss_weight", 0.0)
        self.f_thres = self.deq.eval_f_max_iter
        self.f_tol = self.deq_args["f_tol"]
        self.beta = self.deq_args.get("beta", self.deq_args.get("tau", 0.8))
        self.sradius_mode = False

        # prepare weight normalization
        self.init_gain = init_gain
        if self.do_weight_norm:
            apply_weight_norm(self)
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_inner == 0 else 2,  # 2 if we have MLP
            )
        )

        # handle pretraining (fixed step unrolling instead of implicit solving)
        self.pretrain = True if pretrain_steps > 0 else False
        self.pretrain_steps = pretrain_steps
        self.pretrain_iter = pretrain_iter
        self.pretrain_counter = 0

        # initialize evaluation config
        self._init_eval_config()

    @contextmanager
    def spectral_radius_mode(self, enabled=True):
        """Temporarily set spectral radius computation for eval passes."""
        prev = self.sradius_mode
        self.sradius_mode = enabled
        try:
            yield
        finally:
            self.sradius_mode = prev

    def forward(self, hidden_states: Tensor, mixer_kwargs: Dict) -> ImplicitBaseModelOutputWithPast:
        reset_dropout(self)

        hidden_states, jac_loss, log_dict = self.implicit_forward(hidden_states, mixer_kwargs)

        output = ImplicitBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
            implicit_metrics=log_dict,
            jac_loss=jac_loss,
        )

        return output
