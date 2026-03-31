"""
Evaluation configuration for implicit models.
"""

from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """
    Controls eval-time behavior of implicit (DEQ) models.

    Training always uses the simultaneous (parallel) DEQ solver.
    This config only governs inference/evaluation behavior.

    Args:
        mode: "simultaneous" for parallel fixed-point solving over the full
              sequence, or "sequential" for token-by-token fixed-point iteration
              (required for autoregressive generation).
        max_iter: Override solver max iterations. None uses model default.
        tol: Override convergence tolerance. None uses model default.
        momentum: Damping factor for the fixed-point iteration
                  z = (1 - momentum) * z + momentum * f(z).
                  None uses model default (beta from DEQ params).
        spectral_radius: If True, compute spectral radius at fixed points
                         via power iteration on the Jacobian.
    """

    mode: str = "simultaneous"
    max_iter: int | None = None
    tol: float | None = None
    momentum: float | None = None
    spectral_radius: bool = False
