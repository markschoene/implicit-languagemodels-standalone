import pytest
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

import implicit_llm


CONFIG_PATHS = [
    "hf_models/mamba2-130m-explicit",
    "hf_models/mamba2-130m-implicit",
    "hf_models/llama3-130m-explicit",
    "hf_models/llama3-130m-implicit",
]


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_num_hidden_layers_matches_backbone(config_path):
    config = AutoConfig.from_pretrained(config_path)
    assert config.num_hidden_layers == config.backbone_config["n_layer"]


@pytest.mark.parametrize("config_cls", ["ImplicitLlamaConfig", "ImplicitMambaConfig"])
def test_token_ids_default_to_none(config_cls):
    cls = getattr(implicit_llm, config_cls)
    config = cls()
    assert config.eos_token_id is None
    assert config.pad_token_id is None


@pytest.mark.parametrize("config_cls", ["ImplicitLlamaConfig", "ImplicitMambaConfig"])
def test_token_ids_overridable(config_cls):
    cls = getattr(implicit_llm, config_cls)
    config = cls(eos_token_id=42, pad_token_id=7)
    assert config.eos_token_id == 42
    assert config.pad_token_id == 7


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_dynamic_cache_can_be_created(config_path):
    """DynamicCache requires num_hidden_layers -- this was the original crash."""
    config = AutoConfig.from_pretrained(config_path)
    cache = DynamicCache(config=config)
    assert cache is not None
