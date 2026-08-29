"""Unit tests for custom training loss functions."""
import torch

from verixa.training.loss import SmoothBCEWithLogitsLoss


def test_smooth_bce_loss_forward_and_backward() -> None:
    criterion = SmoothBCEWithLogitsLoss(alpha=0.05)
    logits = torch.randn(4, 1, requires_grad=True)
    targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])

    loss = criterion(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_smooth_bce_loss_alpha_zero_matches_standard() -> None:
    criterion_smooth = SmoothBCEWithLogitsLoss(alpha=0.0)
    criterion_std = torch.nn.BCEWithLogitsLoss()

    logits = torch.randn(8, 1)
    targets = torch.tensor([[0.0], [1.0]] * 4)

    loss_smooth = criterion_smooth(logits, targets)
    loss_std = criterion_std(logits, targets)

    assert torch.allclose(loss_smooth, loss_std, atol=1e-6)
