from dataclasses import asdict, dataclass, field
from typing import Any

from transformers import PretrainedConfig


@dataclass
class SolverConfig:
    f_max_iter: int
    f_tol: float
    eval_factor: float
    eval_f_max_iter: int


@dataclass
class TrainingConfig:
    grad: int
    tau: float
    beta: float


@dataclass
class RegularizationConfig:
    jac_loss_weight: float
    jac_loss_freq: float


@dataclass
class DEQConfig:
    solver: dict
    training: dict
    regularization: dict
    weight_norm: bool = True

    @staticmethod
    def from_dict(deq_dict):
        solver_dict = asdict(SolverConfig(**deq_dict["solver"]))
        training_dict = asdict(TrainingConfig(**deq_dict["training"]))
        regularization_dict = asdict(RegularizationConfig(**deq_dict["regularization"]))
        return DEQConfig(
            solver=solver_dict,
            training=training_dict,
            regularization=regularization_dict,
            weight_norm=deq_dict.get("weight_norm", True),
        )


@dataclass
class BackboneConfig:
    n_layer: int
    d_model: int
    d_inner: int
    pre_norm: bool
    rms_norm: bool
    dropout: float
    pretrain_iter: int
    pretrain_steps: int
    init_gain: float
    block_type: str
    block_cfg: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalLMConfig:
    tie_embeddings: bool
    pad_vocab: bool
    pad_vocab_size_multiple: int
    emb_init_std: float
    dropout: float
    weight_decay: float
    word_level: bool
    backbone: BackboneConfig
    data_info: Any = field(default_factory=dict)
    vocab_size: int | None = None

    @staticmethod
    def from_dict(config_dict) -> "CausalLMConfig":
        backbone_config = BackboneConfig(**config_dict["backbone"])

        config_dict["backbone"] = backbone_config

        if "deq" in config_dict["backbone"].deq_params:
            deq_config = DEQConfig(**config_dict["backbone"].deq_params["deq"])
            config_dict["backbone"].deq_params = deq_config

        return CausalLMConfig(**config_dict)


class ImplicitLlamaConfig(PretrainedConfig):
    model_type = "implicit_llama3"

    def __init__(
        self,
        deq_params: dict = {},
        backbone_type: str = "implicit",
        backbone_config: dict = {},
        vocab_size: int = 50277,
        head_bias: bool = False,
        d_embed: int | None = None,
        average_eval: bool = True,
        dropout: float = 0.05,
        weight_decay: float = 0.0,
        data_info: dict = {},
        dataset_name: str | None = None,
        pad_vocab: bool = False,
        pad_vocab_size_multiple: int = 8,
        tie_embeddings: bool = True,
        emb_init_std: float = 0.01,
        keep_sequence_dim: bool = False,
        save_output_ids: bool = False,
        **kwargs,
    ):
        self.deq_params = deq_params
        self.vocab_size = vocab_size
        self.backbone_type = backbone_type
        self.backbone_config = backbone_config
        self.head_bias = head_bias
        self.d_embed = d_embed
        self.average_eval = average_eval
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.data_info = data_info
        self.dataset_name = dataset_name
        self.pad_vocab = pad_vocab
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.tie_embeddings = tie_embeddings
        self.emb_init_std = emb_init_std
        self.keep_sequence_dim = keep_sequence_dim
        self.save_output_ids = save_output_ids
        self.tokenizer = "EleutherAI/gpt-neox-20b"

        super().__init__(**kwargs)


if __name__ == "__main__":
    pretrained_model_config_name = "mamba2-1.3b-wt"
    my_config = ImplicitLlamaConfig.from_pretrained(pretrained_model_config_name)
    my_config.save_pretrained(f"{pretrained_model_config_name}")
