import pytest
import torch

from implicit_llm import sequential_forward
from tests.conftest import make_model, requires_cuda


BACKBONES = ["llama3", pytest.param("mamba2", marks=requires_cuda)]


@pytest.fixture(params=BACKBONES)
def implicit_model(request, device):
    return make_model(request.param, "implicit", device)


@pytest.fixture(params=BACKBONES)
def explicit_model(request, device):
    return make_model(request.param, "explicit", device)


class TestSimultaneousEvaluation:
    def test_simultaneous_logits_shape(self, implicit_model, random_input_ids):
        """Simultaneous forward produces logits of shape (B, L, V)."""
        with torch.no_grad():
            out = implicit_model(input_ids=random_input_ids, use_cache=False)
        B, L = random_input_ids.shape
        assert out.logits.shape[:2] == (B, L)
        assert out.logits.shape[2] >= implicit_model.config.vocab_size


class TestSequentialForward:
    def test_sequential_forward_logits_shape(self, implicit_model, random_input_ids):
        """sequential_forward() returns logits of correct shape."""
        out = sequential_forward(implicit_model, random_input_ids)
        B, L = random_input_ids.shape
        assert out.logits.shape[:2] == (B, L)
        assert out.logits.shape[2] >= implicit_model.config.vocab_size

    def test_sequential_forward_returns_metrics(self, implicit_model, random_input_ids):
        """sequential_forward() returns implicit_metrics with convergence info."""
        out = sequential_forward(implicit_model, random_input_ids)
        assert out.implicit_metrics is not None
        for key in ["abs diff", "rel diff", "steps"]:
            assert key in out.implicit_metrics


class TestExplicitModelForward:
    def test_explicit_model_forward_shape(self, explicit_model, random_input_ids):
        """Explicit model forward returns correct logits shape without eval mode."""
        with torch.no_grad():
            out = explicit_model(input_ids=random_input_ids, use_cache=False)
        B, L = random_input_ids.shape
        assert out.logits.shape[:2] == (B, L)
        assert out.logits.shape[2] >= explicit_model.config.vocab_size
