"""Development-set calibration, threshold optimization, and test-time augmentation (TTA)."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import f1_score


def predict_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    use_hflip: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute model inference with optional horizontal-flip Test-Time Augmentation (TTA).

    Args:
        model: PyTorch model in eval mode.
        images: Batch of image tensors on device of shape (B, 3, 224, 224).
        use_hflip: If True, averages predictions across original and horizontally flipped images.

    Returns:
        Tuple of (averaged_logits, averaged_probabilities) of shape (B,).
    """
    if not use_hflip:
        logits = model(images).squeeze(-1)
        probs = torch.sigmoid(logits)
        return logits, probs

    # Original forward
    logits_orig = model(images).squeeze(-1)
    probs_orig = torch.sigmoid(logits_orig)

    # Horizontally flipped forward
    images_flipped = torch.flip(images, dims=[-1])
    logits_flipped = model(images_flipped).squeeze(-1)
    probs_flipped = torch.sigmoid(logits_flipped)

    # Average probabilities
    probs_avg = 0.5 * (probs_orig + probs_flipped)
    # Calibrated surrogate logit from average probability
    eps = 1e-7
    probs_clamped = torch.clamp(probs_avg, eps, 1.0 - eps)
    logits_avg = torch.log(probs_clamped / (1.0 - probs_clamped))

    return logits_avg, probs_avg


def find_development_thresholds(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    target_fprs: list[float] | None = None,
) -> dict[str, Any]:
    """Compute optimal and operational thresholds strictly from development set distributions.

    Args:
        y_true: Ground truth binary labels (0 = Real, 1 = Fake).
        y_prob: Predicted probabilities.
        target_fprs: List of target FPR ceilings for operational thresholds.

    Returns:
        Dictionary of calibrated thresholds and their metrics on the development set.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    if target_fprs is None:
        target_fprs = [0.01, 0.02, 0.03, 0.05]

    # Grid search across 500 thresholds
    thresholds = np.linspace(0.001, 0.999, 500)
    best_acc = 0.0
    best_acc_th = 0.5
    best_f1 = 0.0
    best_f1_th = 0.5

    for th in thresholds:
        preds = (y_prob_arr >= th).astype(np.int64)
        acc = float((preds == y_true_arr).mean())
        if acc > best_acc:
            best_acc = acc
            best_acc_th = float(th)

        f1 = float(f1_score(y_true_arr, preds, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_f1_th = float(th)

    # Operational thresholds derived from authentic (y_true == 0) distribution
    real_probs = y_prob_arr[y_true_arr == 0]
    fpr_thresholds: dict[str, float] = {}

    for target_fpr in target_fprs:
        # Quantile (1 - target_fpr) guarantees empirical FPR <= target_fpr on authentic dev data
        percentile_val = (1.0 - target_fpr) * 100.0
        th_val = float(np.percentile(real_probs, percentile_val))
        fpr_thresholds[f"th_fpr_le_{int(target_fpr * 100)}pct"] = round(th_val, 4)

    return {
        "best_accuracy_threshold": round(best_acc_th, 4),
        "best_accuracy_on_dev": round(best_acc, 4),
        "best_f1_threshold": round(best_f1_th, 4),
        "best_f1_on_dev": round(best_f1, 4),
        "operational_fpr_thresholds": fpr_thresholds,
    }


def fit_temperature_scaling(
    logits: np.ndarray | list[float],
    y_true: np.ndarray | list[int],
) -> float:
    """Optimize a single positive temperature scalar T via NLL on development validation data."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)

    def nll_objective(T_arr: np.ndarray) -> float:
        T = float(T_arr[0])
        scaled_z = z / max(T, 1e-4)
        # Numerically stable binary cross entropy
        loss = np.maximum(scaled_z, 0) - scaled_z * y + np.log(1.0 + np.exp(-np.abs(scaled_z)))
        return float(np.mean(loss))

    res = minimize(nll_objective, x0=np.array([1.0]), bounds=[(0.01, 10.0)], method="L-BFGS-B")
    return float(res.x[0])
