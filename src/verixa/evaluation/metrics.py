"""Evaluation metrics calculator for binary synthetic image detection."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def calculate_binary_metrics(
    y_true: np.ndarray | list[float],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute binary classification metrics: Accuracy, AUROC, FPR, and confusion matrix.

    Args:
        y_true: Array of ground-truth binary labels (0 = REAL, 1 = AI-GENERATED).
        y_prob: Array of predicted probabilities for class 1 (P(AI-GENERATED)).
        threshold: Decision threshold for discrete classification (default 0.5).

    Returns:
        Dict containing accuracy, auroc, fpr_at_threshold, fpr_at_95_tpr, and
        confusion matrix counts.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    if len(y_true_arr) == 0:
        raise ValueError("Cannot calculate metrics on an empty array.")

    # Threshold-based predictions
    y_pred = (y_prob_arr >= threshold).astype(np.int64)

    # Confusion matrix
    # labels=[0, 1] ensures 2x2 shape even if only one class is present
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    accuracy = float((tp + tn) / max(1, len(y_true_arr)))
    fpr_at_threshold = float(fp / max(1, fp + tn))
    fnr_at_threshold = float(fn / max(1, fn + tp))

    # AUROC and ROC curve
    # roc_auc_score requires at least one positive and one negative sample
    unique_labels = np.unique(y_true_arr)
    if len(unique_labels) > 1:
        auroc = float(roc_auc_score(y_true_arr, y_prob_arr))
        fpr_arr, tpr_arr, _ = roc_curve(y_true_arr, y_prob_arr)
        # Find FPR at TPR >= 0.95
        idx_95 = np.where(tpr_arr >= 0.95)[0]
        fpr_at_95_tpr = float(fpr_arr[idx_95[0]]) if len(idx_95) > 0 else 1.0
    else:
        auroc = 0.5
        fpr_at_95_tpr = 1.0

    return {
        "total_samples": int(len(y_true_arr)),
        "threshold": float(threshold),
        "accuracy": round(accuracy, 4),
        "auroc": round(auroc, 4),
        "fpr": round(fpr_at_threshold, 4),
        "fnr": round(fnr_at_threshold, 4),
        "fpr_at_95_tpr": round(fpr_at_95_tpr, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
