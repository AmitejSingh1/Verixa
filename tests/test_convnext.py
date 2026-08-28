"""Unit tests for Phase 2 ConvNeXt model, metrics, and dataset logic."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from verixa.evaluation.metrics import calculate_binary_metrics
from verixa.models.convnext import ConvNeXtBinaryClassifier, freeze_backbone_stages


def test_convnext_binary_classifier_forward_shape() -> None:
    # Use uninitialized weights for fast testing
    model = ConvNeXtBinaryClassifier(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 1)


def test_freeze_backbone_stages() -> None:
    model = ConvNeXtBinaryClassifier(pretrained=False)
    freeze_backbone_stages(model, freeze_up_to_stage=2)

    # Features 0..5 should have requires_grad=False
    for idx, child in enumerate(model.features.children()):
        if idx <= 5:
            assert all(not p.requires_grad for p in child.parameters())
        else:
            assert all(p.requires_grad for p in child.parameters())

    # Classification head must have requires_grad=True
    assert all(p.requires_grad for p in model.head.parameters())


def test_calculate_binary_metrics_perfect() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = calculate_binary_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["confusion_matrix"]["true_positives"] == 2
    assert metrics["confusion_matrix"]["true_negatives"] == 2


def test_calculate_binary_metrics_imperfect() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.8, 0.1, 0.2, 0.9])  # 1 false positive, 1 false negative
    metrics = calculate_binary_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"]["false_positives"] == 1
    assert metrics["confusion_matrix"]["false_negatives"] == 1


def test_calculate_binary_metrics_empty_raises() -> None:
    with pytest.raises(ValueError, match="Cannot calculate metrics"):
        calculate_binary_metrics([], [])

