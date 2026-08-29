"""Unit tests for Hybrid RGB+FFT classifier and stage freezing."""
from __future__ import annotations

import torch

from verixa.models.hybrid import HybridRGBFFTClassifier, freeze_hybrid_backbone_stages


def test_hybrid_forward_shape() -> None:
    model = HybridRGBFFTClassifier(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 1)
    assert not torch.isnan(logits).any()


def test_hybrid_gradient_flow() -> None:
    model = HybridRGBFFTClassifier(pretrained=False)
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    assert x.grad is not None


def test_freeze_hybrid_backbone_stages() -> None:
    model = HybridRGBFFTClassifier(pretrained=False)
    freeze_hybrid_backbone_stages(model, freeze_up_to_stage=2)

    # 1. Stem and early stages should be frozen
    for p in model.spatial_backbone[0].parameters():
        assert not p.requires_grad

    # 2. Stage 3 should be trainable
    for p in model.spatial_backbone[7].parameters():
        assert p.requires_grad

    # 3. Spectral branch should be trainable
    for p in model.spectral_branch.parameters():
        assert p.requires_grad

    # 4. Fusion head should be trainable
    for p in model.head.parameters():
        assert p.requires_grad


def test_hybrid_parameter_budget() -> None:
    model = HybridRGBFFTClassifier(pretrained=False)
    total_params = sum(p.numel() for p in model.parameters())
    # Hybrid should be ~30.6M parameters (well under 2B limit)
    assert 25_000_000 < total_params < 40_000_000

