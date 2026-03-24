import math
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

from ..backbones import ExplicitModel, ImplicitModel, ModelConfig
from ..modules.heads import PartialCrossEntropyHead
from ..modules.mamba2 import Mamba2Cache
from ..utils import load_checkpoint
from .configuration_mamba2 import ImplicitMambaConfig


@dataclass
# Copied from transformers.models.mamba.modeling_mamba.MambaOutput with MAMBA->MAMBA2,Mamba->Mamba2
class Mamba2Output(ModelOutput):
    """
    Class for the MAMBA2 model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`Mamba2Cache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[Mamba2Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ImplicitCausalLMOutputWithPast(CausalLMOutputWithPast, Mamba2Output):
    implicit_metrics: Optional[Dict[str, Tensor]] | None = None
    jac_loss: Optional[Tensor] | None = None


class ImplicitMambaForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Model class for explicit and implicit Causal LM.
    """

    config_class = ImplicitMambaConfig
    _tied_weights_keys = ["criterion.decoder.weight"]

    def __init__(
        self,
        config: ImplicitMambaConfig,
    ):
        deq_params: Dict = config.deq_params
        backbone_type: str = config.backbone_type
        backbone_config: Dict = config.backbone_config
        n_tokens: int = config.n_tokens
        head_bias: bool = config.head_bias
        dropout: float = config.dropout
        weight_decay: float = config.weight_decay
        tokenizer: nn.Module = config.tokenizer
        pad_vocab: bool = config.pad_vocab
        pad_vocab_size_multiple: int = config.pad_vocab_size_multiple
        tie_embeddings: bool = config.tie_embeddings
        emb_init_std: float = config.emb_init_std
        load_from_pretrained_shell: bool = config.load_from_pretrained_shell
        keep_sequence_dim: bool = config.keep_sequence_dim
        save_output_ids: bool = config.save_output_ids
        config.tie_word_embeddings = tie_embeddings
        super().__init__(config)

        self.config = config
        self.gradient_checkpointing = False
        self.config._attn_implementation = "sdpa"
        backbone_cls = ExplicitModel if "explicit" in backbone_type else ImplicitModel
        backbone = backbone_cls(deq_params=deq_params, **backbone_config)
        self.config.vocab_size = n_tokens
        d_model = backbone.d_model
        if n_tokens % pad_vocab_size_multiple != 0 and pad_vocab:
            n_tokens += pad_vocab_size_multiple - (n_tokens % pad_vocab_size_multiple)

        self.backbone = backbone
        self.word_emb = nn.Embedding(n_tokens, d_model)
        self.criterion = PartialCrossEntropyHead(
            vocab_size=n_tokens,
            d_model=d_model,
            tokenizer=tokenizer,
            head_bias=head_bias,
        )
        self.emb_init_std = emb_init_std
        self.keep_sequence_dim = keep_sequence_dim
        self.save_output_ids = save_output_ids
        self.per_batch_sequence_losses = []
        self.per_batch_output_ids = []

        self.drop = nn.Dropout(p=dropout)
        self.weight_decay = weight_decay

        self.tie_embeddings = tie_embeddings
        self.pad_vocab = pad_vocab
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.config_attr = self.create_config_attributes()
        self.tokenizer = tokenizer
        self.load_from_pretrained_shell = load_from_pretrained_shell

        self.apply(self._init_weights)
        self.post_init()

    def get_input_embeddings(self):
        return self.word_emb

    def set_input_embeddings(self, value):
        self.word_emb = value

    def get_output_embeddings(self):
        return self.criterion.decoder

    def set_output_embeddings(self, value):
        self.criterion.decoder = value

    def create_config_attributes(self):
        return ModelConfig(
            d_model=self.backbone.d_model,
            d_intermediate=self.backbone.d_inner,
            n_layer=self.backbone.n_layer,
            block_cfg=self.backbone.block_cfg,
            rms_norm=self.backbone.rms_norm,
            tie_embeddings=self.tie_embeddings,
            pad_vocab_size_multiple=self.pad_vocab_size_multiple,
        )

    def set_exp2implicit(self):
        self.backbone.set_exp2implicit()

    def update_deq(self, deq_params):
        self.backbone.update_deq(deq_params)

    def sequential_evaluation(self, tau=1.0):  #  remove
        self.backbone.sequential_evaluation(tau)

    def simultaneous_evaluation(self):  #  remove
        self.backbone.simultaneous_evaluation()

    def keep_per_bacth_metrics(self, loss: torch.Tensor, logits: torch.Tensor):
        if self.keep_sequence_dim:
            self.per_batch_sequence_losses.append(loss.detach())
            loss = loss.mean()
        if self.save_output_ids:
            self.per_batch_output_ids.append(logits.max(dim=-1)[1].detach())
        return loss

    def prepare_inputs_for_generation(  # type: ignore
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Adapted Huggingface
        transformers/models/mamba/modeling_mamba.py
        """
        # Overwritten -- uses `cache_params` as opposed to `past_key_values`

        if use_cache:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            if cache_position[0] > 0:
                input_ids = input_ids[:, -1].unsqueeze(-1)

                if attention_mask is not None:
                    attention_mask = None

            else:
                # we initialize the `cache_position` to full size of `conv_states` at prefill stage
                # considering padding will be applied when input length is shorter, and truncation
                # will be applied when it is longer, so it will be equivalent to always have it match
                # the length of `cache_params.conv_states`, which is `config.conv_kernel`
                d_conv = self.backbone.layers[0].mixer.d_conv
                cache_position = torch.arange(0, d_conv, device=input_ids.device)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_params: Optional[Mamba2Cache] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Tuple[Tensor, Dict, Tensor]:
        # kwargs validation for our models
        assert output_hidden_states is False, "output_hidden_states is not supported"
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.word_emb(input_ids)

        d_conv = self.backbone.layers[0].mixer.d_conv
        if use_cache:
            if cache_params is None:
                cache_params = Mamba2Cache(
                    self.backbone.layers,
                    inputs_embeds.size(0),
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                cache_position = torch.arange(0, d_conv, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        inputs_embeds = self.drop(inputs_embeds)
        hidden_states = inputs_embeds
        # package key words as mixer_kwargs
        mixer_kwargs = {
            "cache_params": cache_params,
            "use_cache": use_cache,
            "cache_position": cache_position,
        }

        outputs = self.backbone(hidden_states, mixer_kwargs=mixer_kwargs)
        hidden_states = outputs[0]

        # for generation, we need to take care of the sequence dimension
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states.unsqueeze(1)

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        _, _, _, logits = self.criterion(hidden_states[:, slice_indices, :])  #  check this if it works correctly

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        output = ImplicitCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache_params=cache_params,
            last_hidden_state=outputs.last_hidden_state,
            implicit_metrics=outputs.implicit_metrics,
            jac_loss=outputs.jac_loss,
        )
        return output if return_dict else output.to_tuple()

    def _init_weights(self, m: nn.Module, initializer_range=0.02):
        classname = m.__class__.__name__
        if classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, std=initializer_range)

    @classmethod
    def from_config(
        cls,
        pretrained_model_name: Union[str, ImplicitMambaConfig],
        device=None,
        dtype=None,
        **kwargs,
    ) -> "ImplicitMambaForCausalLM":

        if isinstance(pretrained_model_name, str):
            config = ImplicitMambaConfig.from_pretrained(pretrained_model_name)
        elif isinstance(pretrained_model_name, ImplicitMambaConfig):
            config = pretrained_model_name
        else:
            raise ValueError("pretrained_model_name should be either a string or an ImplicitMambaConfig object")

        model = cls(config)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs) -> "ImplicitMambaForCausalLM":
        """
        Loads the model from a given directory (which must contain a config file and pytorch_model.bin).
        Extra keyword arguments will be passed to AutoConfig.from_pretrained and the model constructor.
        Recognized kwargs include: device, dtype.
        """
        from transformers import AutoConfig

        device = kwargs.pop("device_map", None)
        dtype = kwargs.pop("torch_dtype", None)

        # Load the configuration. Any extra kwargs here (like revision, etc.) will be forwarded.
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)

        state_dict = load_checkpoint(pretrained_model_name_or_path)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Some keys are missing in the state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Some unexpected keys in the state_dict: {unexpected_keys}")

        if device is not None:
            model.to(device)
        if dtype is not None:
            model.to(dtype)

        return model

    @classmethod
    def _from_pretrained_explicit(
        cls,
        pretrained_model_name_or_path: str,
        deq_params: Dict,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        """
        TODO: separate explicit and implicit model loading.
        Loads pretrained ImplicitLlamaForCausalLM (explicit version),
        converts to implicit variant,
        modifies config and saves the implicit model along with new config.
        """
        if "explicit" not in pretrained_model_name_or_path:
            raise ValueError("Pretrained model name or path must contain 'explicit' to load the explicit model.")
        print(f"Loading the pretrained model from: {pretrained_model_name_or_path}")
        explicit_model = ImplicitMambaForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            device_map=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        state_dict = explicit_model.state_dict()
        config = explicit_model.config.to_dict()

        print("Modifying model config for 'implicit_llama' model type...")

        config["deq_params"] = deq_params
        config["backbone_type"] = "implicit"
        config["backbone_config"]["do_weight_norm"] = False
        config["backbone_config"]["init_z_mode"] = "input"
        config["_name_or_path"] = f"Exp2Implicit_{pretrained_model_name_or_path}"

        # Produce implicit-compatible weights
        print("Converting explicit state_dict into implicit format...")
        d_model = config["backbone_config"]["d_model"]
        num_attention_heads = config["backbone_config"]["block_cfg"]["n_head"]
        num_key_value_heads = num_attention_heads
        head_dim = d_model // num_attention_heads
        input_projection_dim = (num_attention_heads + 2 * num_key_value_heads) * head_dim
        implicit_state_dict = cls.convert_explicit_to_implicit(
            state_dict, d_model=d_model, input_projection_dim=input_projection_dim
        )

        print("Initializing ImplicitLlamaForCausalLM with modified config...")
        implicit_config = ImplicitMambaConfig(**config)
        implicit_model = cls(implicit_config).to(device=device, dtype=torch_dtype)
        implicit_model.load_state_dict(implicit_state_dict, strict=True)

        return implicit_model

    @staticmethod
    def convert_explicit_to_implicit(
        state_dict: Dict[str, torch.Tensor],
        d_model: int,
        input_projection_dim: int,
        init_gain: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Converts state dict from explicit Qwen2 pretrained model to implicit-compatible model state dict.
        Initializes only additional weights (self.injection Linear and self.injection_norm RMSNorm).
        """

        import copy

        new_state_dict = copy.deepcopy(state_dict)

        # === Initialize injection Linear layer ===
        injection_layer = nn.Linear(d_model, input_projection_dim, bias=True)
        nn.init.xavier_normal_(injection_layer.weight, gain=init_gain)
        nn.init.constant_(injection_layer.bias, 0.0)

        # put initialized weights into new state dict
        new_state_dict["backbone.injection.weight"] = injection_layer.weight.detach()
        new_state_dict["backbone.injection.bias"] = injection_layer.bias.detach()
        print(
            "Initialized 'backbone.injection.weight' and 'backbone.injection.bias' with Xavier-normal and constant-0."
        )

        # === Initialize injection_norm (RMSNorm) layer ===
        injection_norm_layer = nn.RMSNorm(d_model, eps=1e-5)
        nn.init.constant_(injection_norm_layer.weight, 1.0)  # RMSNorm usually has no bias

        # put initialized norm weight into new state dict
        new_state_dict["backbone.injection_norm.weight"] = injection_norm_layer.weight.detach()
        print("Initialized 'backbone.injection_norm.weight' with constant-1.0.")

        return new_state_dict

    def _merge_configs(self, config: Dict, deq_params: Dict) -> ImplicitMambaConfig:
        """
        Merge implicit DEQ-specific hyperparameters into the existing model config.
        """
        print("Merging DEQ-specific parameters into configuration.")
        config["deq_params"] = deq_params
        return ImplicitMambaConfig(**config)

    # def save_pretrained(self, save_directory, **kwargs):
    #     """
    #     Minimal implementation of save_pretrained for CausalLMModel.
    #     Save the model and its configuration file to a directory.
    #     """
    #     # Ensure save_directory exists
    #     os.makedirs(save_directory, exist_ok=True)

    #     # Save the model's state_dict
    #     model_path = os.path.join(save_directory, 'pytorch_model.bin')
    #     torch.save(self.state_dict(), model_path)
    #     self.config.save_pretrained(save_directory)
