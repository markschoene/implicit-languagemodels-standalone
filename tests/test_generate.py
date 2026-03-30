import pytest
import torch

from tests.conftest import make_model, requires_cuda


GENERATION_MODELS = [
    pytest.param(("mamba2", "explicit"), marks=requires_cuda),
    pytest.param(("mamba2", "implicit"), marks=requires_cuda),
    ("llama3", "explicit"),
    ("llama3", "implicit"),
]


@pytest.fixture(params=GENERATION_MODELS)
def generation_model(request, device):
    backbone, model_type = request.param
    model = make_model(backbone, model_type, device)
    if model_type != "explicit":
        model.backbone.sequential_evaluation()
    return model


class TestGenerate:
    def test_generate_extends_sequence(self, generation_model):
        input_ids = torch.randint(0, 240, (1, 8)).to(next(generation_model.parameters()).device)
        with torch.no_grad():
            out = generation_model.generate(
                input_ids, max_length=input_ids.shape[1] + 5, do_sample=False
            )
        assert out.shape[1] > input_ids.shape[1]

    def test_generate_greedy_is_deterministic(self, generation_model):
        input_ids = torch.randint(0, 240, (1, 8)).to(next(generation_model.parameters()).device)
        with torch.no_grad():
            out1 = generation_model.generate(
                input_ids, max_length=input_ids.shape[1] + 5, do_sample=False
            )
            out2 = generation_model.generate(
                input_ids, max_length=input_ids.shape[1] + 5, do_sample=False
            )
        assert torch.equal(out1, out2)

    def test_generate_tokens_in_vocab_range(self, generation_model):
        input_ids = torch.randint(0, 240, (1, 8)).to(next(generation_model.parameters()).device)
        with torch.no_grad():
            out = generation_model.generate(
                input_ids, max_length=input_ids.shape[1] + 5, do_sample=False
            )
        assert (out >= 0).all()
        assert (out < generation_model.config.vocab_size).all()
