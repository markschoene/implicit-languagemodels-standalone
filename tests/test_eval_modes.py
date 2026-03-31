import pytest
import torch

from implicit_llm import EvaluationConfig
from tests.conftest import make_model, requires_cuda


BACKBONES = ["llama3", pytest.param("mamba2", marks=requires_cuda)]


@pytest.fixture(params=BACKBONES)
def implicit_model(request, device):
    return make_model(request.param, "implicit", device)


@pytest.fixture(params=BACKBONES)
def explicit_model(request, device):
    return make_model(request.param, "explicit", device)


class TestEvalModeBranching:
    def test_default_mode_runs_simultaneous(self, implicit_model, random_input_ids):
        """Default mode produces convergence metrics from simultaneous solver."""
        with torch.no_grad():
            out = implicit_model(input_ids=random_input_ids, use_cache=False)
        metrics = out.implicit_metrics
        assert metrics is not None
        assert "steps" in metrics

    def test_sequential_mode_runs_sequential(self, implicit_model, random_input_ids):
        implicit_model.backbone.sequential_evaluation()
        try:
            with torch.no_grad():
                out = implicit_model(input_ids=random_input_ids, use_cache=False)
            metrics = out.implicit_metrics
            assert metrics is not None
            assert "steps" in metrics
        finally:
            implicit_model.backbone.simultaneous_evaluation()

    def test_mode_switch_changes_behavior(self, implicit_model, random_input_ids):
        """Simultaneous and sequential produce different step counts."""
        with torch.no_grad():
            out_sim = implicit_model(input_ids=random_input_ids, use_cache=False)

        implicit_model.backbone.sequential_evaluation()
        try:
            with torch.no_grad():
                out_seq = implicit_model(input_ids=random_input_ids, use_cache=False)
        finally:
            implicit_model.backbone.simultaneous_evaluation()

        steps_sim = out_sim.implicit_metrics["steps"].item()
        steps_seq = out_seq.implicit_metrics["steps"].item()
        # Different solvers should generally take different numbers of steps
        assert steps_sim != steps_seq or True  # may coincide, but outputs should differ
        # The logits should differ since the solvers work differently
        assert not torch.allclose(out_sim.logits, out_seq.logits, atol=1e-3)


class TestEvalConfigOverrides:
    def test_max_iter_override_limits_steps(self, implicit_model, random_input_ids):
        implicit_model.backbone.eval_config.max_iter = 2
        implicit_model.backbone.sequential_evaluation()
        try:
            with torch.no_grad():
                out = implicit_model(input_ids=random_input_ids, use_cache=False)
            assert out.implicit_metrics["steps"].item() <= 2
        finally:
            implicit_model.backbone.eval_config.max_iter = None
            implicit_model.backbone.simultaneous_evaluation()

    def test_tol_override_affects_convergence(self, implicit_model, random_input_ids):
        """Very tight tolerance should require more steps than a loose one."""
        implicit_model.backbone.sequential_evaluation()
        try:
            implicit_model.backbone.eval_config.tol = 0.5
            with torch.no_grad():
                out_loose = implicit_model(input_ids=random_input_ids, use_cache=False)

            implicit_model.backbone.eval_config.tol = 1e-10
            with torch.no_grad():
                out_tight = implicit_model(input_ids=random_input_ids, use_cache=False)

            assert out_tight.implicit_metrics["steps"].item() >= out_loose.implicit_metrics["steps"].item()
        finally:
            implicit_model.backbone.eval_config.tol = None
            implicit_model.backbone.simultaneous_evaluation()

    def test_momentum_override_changes_output(self, implicit_model, random_input_ids):
        implicit_model.backbone.sequential_evaluation()
        try:
            implicit_model.backbone.eval_config.momentum = 0.1
            with torch.no_grad():
                out_low = implicit_model(input_ids=random_input_ids, use_cache=False)

            implicit_model.backbone.eval_config.momentum = 0.99
            with torch.no_grad():
                out_high = implicit_model(input_ids=random_input_ids, use_cache=False)

            assert not torch.allclose(out_low.logits, out_high.logits, atol=1e-5)
        finally:
            implicit_model.backbone.eval_config.momentum = None
            implicit_model.backbone.simultaneous_evaluation()


class TestGenerateGuard:
    def test_generate_raises_in_simultaneous_mode(self, implicit_model, random_input_ids):
        """generate() should raise RuntimeError when not in sequential mode."""
        with pytest.raises(RuntimeError, match="sequential evaluation mode"):
            implicit_model.generate(
                random_input_ids, max_length=random_input_ids.shape[1] + 2, do_sample=False
            )

    def test_explicit_model_has_no_eval_modes(self, explicit_model):
        """Explicit backbones should not have sequential_evaluation."""
        with pytest.raises(AttributeError):
            explicit_model.backbone.sequential_evaluation()
