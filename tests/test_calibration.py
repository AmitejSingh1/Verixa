"""Unit tests for development calibration and test-time augmentation (TTA)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from verixa.evaluation.calibration import (
    find_development_thresholds,
    fit_temperature_scaling,
    predict_with_tta,
)


class DummyBinaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x)).flatten(1)


def test_predict_with_tta_shapes() -> None:
    model = DummyBinaryModel()
    model.eval()
    x = torch.randn(4, 3, 224, 224)

    # Test without TTA
    logits_no_tta, probs_no_tta = predict_with_tta(model, x, use_hflip=False)
    assert logits_no_tta.shape == (4,)
    assert probs_no_tta.shape == (4,)
    assert (probs_no_tta >= 0.0).all() and (probs_no_tta <= 1.0).all()

    # Test with horizontal-flip TTA
    logits_tta, probs_tta = predict_with_tta(model, x, use_hflip=True)
    assert logits_tta.shape == (4,)
    assert probs_tta.shape == (4,)
    assert (probs_tta >= 0.0).all() and (probs_tta <= 1.0).all()


def test_find_development_thresholds() -> None:
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.70, 0.80, 0.90, 0.95, 0.99])

    calib = find_development_thresholds(y_true, y_prob)
    assert "best_accuracy_threshold" in calib
    assert "best_f1_threshold" in calib
    assert "operational_fpr_thresholds" in calib

    th_acc = calib["best_accuracy_threshold"]
    assert 0.05 < th_acc < 0.70


def test_fit_temperature_scaling() -> None:
    # Simulated overconfident logits
    logits = np.array([-5.0, -4.0, -3.5, 3.0, 4.5, 6.0])
    y_true = np.array([0, 0, 0, 1, 1, 1])

    T = fit_temperature_scaling(logits, y_true)
    assert isinstance(T, float)
    assert 0.1 < T < 10.0

