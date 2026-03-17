"""
Variational dropout for implicit models.

A single dropout mask is generated once per training iteration and applied
consistently across all fixed-point solver steps, preserving dynamics during
convergence. Call ``reset_dropout(model)`` at the start of each iteration.
"""

import torch
import torch.nn as nn


class VariationalDropout1d(nn.Module):
    """
    Applies Variational Dropout to 1d input tensors.

    During training, randomly zeros out the entire channel/feature dimension
    with probability ``dropout`` using a Bernoulli mask. The same mask is reused
    across all fixed-point solver iterations within a single training step.
    Call ``reset_dropout(model)`` at the start of each new training step.

    Args:
        dropout (float): Probability of an element being zeroed. Default: 0.5.
        token_first (bool): If True, expects input shape (B, L, D),
            otherwise (B, D, L). Default: True.

    Shape:
        - Input: (B, L, D) or (B, D, L).
        - Output: same shape as input.
    """

    def __init__(self, dropout=0.5, token_first=True):
        super().__init__()
        if dropout < 0 or dropout > 1:
            raise ValueError(
                f"dropout probability must be between 0 and 1, got {dropout}"
            )
        self.dropout = dropout
        self.token_first = token_first
        self.mask = None

    def _reset_mask(self, x):
        if self.token_first:
            B, _, D = x.shape
            m = torch.zeros(B, 1, D).bernoulli_(1 - self.dropout)
        else:
            B, D, _ = x.shape
            m = torch.zeros(B, D, 1).bernoulli_(1 - self.dropout)
        self.mask = (m / (1 - self.dropout)).requires_grad_(False).to(x)

    def forward(self, x):
        if not self.training or self.dropout == 0.0:
            return x
        if self.mask is None:
            self._reset_mask(x)
        return self.mask.expand_as(x) * x


def reset_dropout(model):
    """Reset variational dropout masks for all layers in the model."""
    for module in model.modules():
        if isinstance(module, VariationalDropout1d):
            module.mask = None
