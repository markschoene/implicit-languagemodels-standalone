import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

import implicit_llm
from implicit_llm import ImplicitLlamaForCausalLM, ImplicitMambaForCausalLM

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def make_model(backbone, model_type, device="cpu"):
    """Create a model from config with random weights."""
    config_path = f"hf_models/{backbone}-130m-{model_type}"
    config = AutoConfig.from_pretrained(config_path)
    if backbone == "mamba2":
        model = ImplicitMambaForCausalLM(config)
    else:
        model = ImplicitLlamaForCausalLM(config)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


@pytest.fixture
def random_input_ids(device):
    return torch.randint(0, 240, (1, 16)).to(device)


# --- Mamba models (require CUDA) ---

@pytest.fixture(scope="module")
def mamba_explicit_model(device):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for mamba2")
    return make_model("mamba2", "explicit", device)


@pytest.fixture(scope="module")
def mamba_implicit_model(device):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for mamba2")
    return make_model("mamba2", "implicit", device)


# --- Llama models (CPU compatible) ---

@pytest.fixture(scope="module")
def llama_explicit_model(device):
    return make_model("llama3", "explicit", device)


@pytest.fixture(scope="module")
def llama_implicit_model(device):
    return make_model("llama3", "implicit", device)
