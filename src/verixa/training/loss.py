"""Loss functions for synthetic media detection training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothBCEWithLogitsLoss(nn.Module):
    """Binary cross entropy with logits and label smoothing.

    For label y in {0, 1} and smoothing factor alpha:
        y_smooth = y * (1 - alpha) + 0.5 * alpha
    """

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smooth_targets = targets * (1.0 - self.alpha) + 0.5 * self.alpha
        return F.binary_cross_entropy_with_logits(logits, smooth_targets)
