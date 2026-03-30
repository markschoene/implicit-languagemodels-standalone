"""
ImplicitLlamaForCausalLM class for implicit Llama models.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor, Tensor
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..backbones import ExplicitModel, ImplicitModel, ModelConfig
from ..modules.heads import PartialCrossEntropyHead
from ..utils import load_checkpoint
from .configuration_llama import ImplicitLlamaConfig


@dataclass
class ImplicitCausalLMOutputWithPast(CausalLMOutputWithPast):
    implicit_metrics: dict[str, Tensor] | None = None
    jac_loss: Tensor | None = None


class ImplicitLlamaForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Model class for explicit and implicit Causal LM.
    """

    config_class = ImplicitLlamaConfig
    _supports_sdpa = True
    _tied_weights_keys = ["criterion.decoder.weight"]

    def __init__(
        self,
        config: ImplicitLlamaConfig,
    ):
        deq_params: dict = config.deq_params
        backbone_type: str = config.backbone_type
        backbone_config: dict = config.backbone_config
        vocab_size: int = config.vocab_size if hasattr(config, "vocab_size") else config.n_tokens
        head_bias: bool = config.head_bias
        dropout: float = config.dropout
        weight_decay: float = config.weight_decay
        tokenizer: nn.Module = config.tokenizer
        pad_vocab: bool = config.pad_vocab
        pad_vocab_size_multiple: int = config.pad_vocab_size_multiple
        tie_embeddings: bool = config.tie_embeddings
        emb_init_std: float = config.emb_init_std
        keep_sequence_dim: bool = config.keep_sequence_dim
        save_output_ids: bool = config.save_output_ids
        config._attn_implementation = "sdpa"
        config.tie_word_embeddings = tie_embeddings
        super().__init__(config)

        self.config = config
        self.gradient_checkpointing = False
        backbone_cls = ExplicitModel if "explicit" in backbone_type else ImplicitModel
        backbone = backbone_cls(deq_params=deq_params, **backbone_config)
        self.config.vocab_size = vocab_size
        d_model = backbone.d_model
        if vocab_size % pad_vocab_size_multiple != 0 and pad_vocab:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.backbone = backbone
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.criterion = PartialCrossEntropyHead(
            vocab_size=vocab_size,
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

    def sequential_evaluation(self):
        self.backbone.sequential_evaluation()

    def simultaneous_evaluation(self):
        self.backbone.simultaneous_evaluation()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        if hasattr(self.backbone, 'eval_config') and self.backbone.eval_config.mode != "sequential":
            raise RuntimeError(
                "Generation requires sequential evaluation mode. "
                "Call model.backbone.sequential_evaluation() before generate()."
            )
        return super().prepare_inputs_for_generation(*args, **kwargs)

    def forward(
        self,
        input_ids: LongTensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: FloatTensor | None = None,
        labels: LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool | None = None,
        cache_position: LongTensor | None = None,
        logits_to_keep: int | Tensor = 0,
        **kwargs,
    ) -> tuple[Tensor] | ImplicitCausalLMOutputWithPast:
        # kwargs validation for our models
        assert output_attentions is False, "output_attentions is not supported"
        assert output_hidden_states is False, "output_hidden_states is not supported"
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.word_emb(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        inputs_embeds = self.drop(inputs_embeds)
        hidden_states = inputs_embeds
        # package key words as mixer_kwargs
        mixer_kwargs = {
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "past_key_value": past_key_values,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
            "cache_position": cache_position,
        }

        outputs = self.backbone(hidden_states, mixer_kwargs=mixer_kwargs)
        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        _, _, _, logits = self.criterion(
            hidden_states[:, slice_indices, :], just_decode=True
        )  #  check this if it works correctly

        loss = None
        if labels is not None:
            # note: labels are shifted by one position in _loss_function
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
            past_key_values=past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            implicit_metrics=outputs.implicit_metrics,
            jac_loss=outputs.jac_loss,
        )
        return output if return_dict else output.to_tuple()

    def _init_weights(self, m: nn.Module, initializer_range=0.02):
        classname = m.__class__.__name__
        if classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, std=initializer_range)

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    @classmethod
    def from_config(
        cls,
        pretrained_model_name: str | ImplicitLlamaConfig,
        device=None,
        dtype=None,
        **kwargs,
    ):
        if isinstance(pretrained_model_name, str):
            config = ImplicitLlamaConfig.from_pretrained(pretrained_model_name)
        elif isinstance(pretrained_model_name, ImplicitLlamaConfig):
            config = pretrained_model_name
        else:
            raise ValueError("pretrained_model_name should be either a string or an ImplicitLlamaConfig object")

        model = cls(config)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
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
