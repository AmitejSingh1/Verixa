"""Fresh canonical threshold analysis for Hybrid V1.

Authoritative source of truth:
Evaluates models/convnext_tiny_hybrid_fft.pt strictly on the 6,001-image
validation split of data/manifests/merged_manifest.csv.
Computes raw score statistics, percentiles, target-FPR operational thresholds,
fixed thresholds, and multi-condition robustness across candidate thresholds.
Outputs to reports/threshold_calibration.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

from verixa.models.loader import load_model_from_checkpoint
from verixa.training.augmentations import EVAL_DISTORTION_SUITES, get_distortion_eval_transform
from verixa.training.dataset import create_eval_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute fresh canonical threshold calibration report for Hybrid V1."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/convnext_tiny_hybrid_fft.pt"),
        help="Path to trained Hybrid V1 model checkpoint (.pt).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/merged_manifest.csv"),
        help="Path to development manifest CSV containing validation split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count (0 for main process, avoids Windows IPC memory limits).",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/threshold_calibration.json"),
        help="Path to save canonical threshold calibration JSON.",
    )
    return parser.parse_args()


def run_split_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute forward inference returning (y_true, y_logits, y_probs)."""
    all_targets: list[int] = []
    all_logits: list[float] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].long()
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images).squeeze(-1)
                probs = torch.sigmoid(logits)

            all_targets.extend(targets.tolist())
            all_logits.extend(logits.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return (
        np.array(all_targets, dtype=np.int64),
        np.array(all_logits, dtype=np.float64),
        np.array(all_probs, dtype=np.float64),
    )


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute detailed classification metrics at an exact decision threshold."""
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = float((tp + tn) / max(1, len(y_true)))
    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))
    tpr = float(tp / max(1, fn + tp))
    precision = float(tp / max(1, tp + fp))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(accuracy, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "tpr_synthetic_recall": round(tpr, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    print("=================================================================")
    print(" Verixa — Fresh Canonical Threshold Calibration (Hybrid V1)")
    print("=================================================================")
    print(f" Model Checkpoint: {args.model_path}")
    print(f" Manifest:         {args.manifest}")
    print(f" Device:           {device}")
    print(f" Report Out:       {args.report_out}")
    print("=================================================================")

    # 1. Load model and verify checkpoint exists
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    model = load_model_from_checkpoint(args.model_path, device=device)
    model.eval()

    # 2. Fresh Inference strictly on Clean Development Validation Split
    print("\nExecuting fresh inference on clean validation split (N=6,001)...")
    clean_loader = create_eval_dataloader(
        manifest_path=args.manifest,
        split="val",
        transform=get_distortion_eval_transform("clean"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    y_true_clean, logits_clean, probs_clean = run_split_inference(
        model=model, loader=clean_loader, device=device
    )

    total_samples = len(y_true_clean)
    num_authentic = int((y_true_clean == 0).sum())
    num_synthetic = int((y_true_clean == 1).sum())

    clean_auroc = float(roc_auc_score(y_true_clean, probs_clean))
    fpr_curve, tpr_curve, _ = roc_curve(y_true_clean, probs_clean)
    idx_95 = np.where(tpr_curve >= 0.95)[0]
    fpr_at_95_tpr = float(fpr_curve[idx_95[0]]) if len(idx_95) > 0 else 1.0

    print(f"Total Validation Samples: {total_samples:,}")
    print(f"  - Authentic (Class 0):  {num_authentic:,}")
    print(f"  - Synthetic (Class 1):  {num_synthetic:,}")
    print(f"  - Clean AUROC:          {clean_auroc*100:.2f}%")
    print(f"  - Clean FPR@95% TPR:    {fpr_at_95_tpr*100:.2f}%")

    # 3. Raw Score Statistics by Class
    real_probs = probs_clean[y_true_clean == 0]
    fake_probs = probs_clean[y_true_clean == 1]
    real_logits = logits_clean[y_true_clean == 0]
    fake_logits = logits_clean[y_true_clean == 1]

    percentiles_to_calc = [
        1.0, 2.0, 2.5, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 97.0, 97.5, 98.0, 99.0
    ]

    score_statistics = {
        "authentic": {
            "count": num_authentic,
            "mean_probability": round(float(np.mean(real_probs)), 6),
            "std_probability": round(float(np.std(real_probs)), 6),
            "min_probability": round(float(np.min(real_probs)), 6),
            "max_probability": round(float(np.max(real_probs)), 6),
            "median_probability": round(float(np.median(real_probs)), 6),
            "mean_logit": round(float(np.mean(real_logits)), 4),
            "std_logit": round(float(np.std(real_logits)), 4),
            "percentiles_probability": {
                f"p{p:g}": round(float(np.percentile(real_probs, p)), 6)
                for p in percentiles_to_calc
            },
        },
        "synthetic": {
            "count": num_synthetic,
            "mean_probability": round(float(np.mean(fake_probs)), 6),
            "std_probability": round(float(np.std(fake_probs)), 6),
            "min_probability": round(float(np.min(fake_probs)), 6),
            "max_probability": round(float(np.max(fake_probs)), 6),
            "median_probability": round(float(np.median(fake_probs)), 6),
            "mean_logit": round(float(np.mean(fake_logits)), 4),
            "std_logit": round(float(np.std(fake_logits)), 4),
            "percentiles_probability": {
                f"p{p:g}": round(float(np.percentile(fake_probs, p)), 6)
                for p in percentiles_to_calc
            },
        },
    }

    # 4. Determine Threshold Candidates
    # Operational Target-FPR thresholds (derived from 1 - FPR quantile of authentic)
    th_fpr10 = float(np.percentile(real_probs, 99.0))
    th_fpr20 = float(np.percentile(real_probs, 98.0))
    th_fpr25 = float(np.percentile(real_probs, 97.5))
    th_fpr30 = float(np.percentile(real_probs, 97.0))
    th_fpr50 = float(np.percentile(real_probs, 95.0))

    # B. Optimal empirical thresholds on clean dev val
    grid_ths = np.linspace(0.001, 0.999, 1000)
    best_acc, best_acc_th = 0.0, 0.5
    best_f1, best_f1_th = 0.0, 0.5

    for th in grid_ths:
        preds = (probs_clean >= th).astype(np.int64)
        acc = float((preds == y_true_clean).mean())
        if acc > best_acc:
            best_acc = acc
            best_acc_th = float(th)

        f1 = float(f1_score(y_true_clean, preds, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_f1_th = float(th)

    evaluated_threshold_configs = [
        ("th_default_050", 0.50, "Standard default decision boundary"),
        ("th_fixed_025", 0.25, "Fixed conservative threshold (0.25)"),
        ("th_fixed_010", 0.10, "Fixed sensitive threshold (0.10)"),
        ("th_fixed_005", 0.05, "Fixed high-recall threshold (0.05)"),
        ("th_dev_fpr_target_1pct", th_fpr10, "Operational threshold targeting <= 1.0% FPR on dev"),
        ("th_dev_fpr_target_2pct", th_fpr20, "Operational threshold targeting <= 2.0% FPR on dev"),
        ("th_dev_fpr_target_2.5pct", th_fpr25, "Operational threshold targeting <= 2.5% FPR"),
        ("th_dev_fpr_target_3pct", th_fpr30, "Operational threshold targeting <= 3.0% FPR on dev"),
        ("th_dev_fpr_target_5pct", th_fpr50, "Operational threshold targeting <= 5.0% FPR on dev"),
        ("th_dev_optimal_accuracy", best_acc_th, "Optimal accuracy threshold on clean dev val"),
        ("th_dev_optimal_f1", best_f1_th, "Optimal F1 threshold on clean dev val"),
    ]

    threshold_results: dict[str, Any] = {}
    print("\n--- Canonical Threshold Performance on Clean Dev Val ---")
    hdr = (
        f"{'Identifier':<26} | {'Threshold':<9} | {'Accuracy':<8} | {'FPR':<7} | "
        f"{'TPR/Recall':<10} | {'Prec':<7} | {'F1':<7}"
    )
    print(hdr)
    print("-" * 88)

    for name, th_val, desc in evaluated_threshold_configs:
        m = compute_metrics_at_threshold(y_true_clean, probs_clean, threshold=th_val)
        m["description"] = desc
        threshold_results[name] = m
        print(
            f"{name:<26} | {m['threshold']:<9.6f} | {m['accuracy']*100:<7.2f}% | "
            f"{m['fpr']*100:<6.2f}% | {m['tpr_synthetic_recall']*100:<9.2f}% | "
            f"{m['precision']*100:<6.2f}% | {m['f1']*100:<6.2f}%"
        )

    print("-" * 88)

    # 5. Evaluate Candidate Thresholds Across Complete 17-Condition Suite
    print("\nEvaluating candidate thresholds across all 17 development conditions...")
    candidate_keys_to_eval = [
        "th_default_050",
        "th_fixed_025",
        "th_fixed_010",
        "th_fixed_005",
        "th_dev_fpr_target_2pct",
        "th_dev_fpr_target_2.5pct",
        "th_dev_fpr_target_3pct",
        "th_dev_fpr_target_5pct",
    ]

    suite_conditions = ["clean"] + list(EVAL_DISTORTION_SUITES.keys())
    multi_condition_robustness: dict[str, Any] = {c: {} for c in suite_conditions}

    for cond_idx, condition in enumerate(suite_conditions, start=1):
        if condition == "clean":
            y_t, y_p = y_true_clean, probs_clean
        else:
            tf = get_distortion_eval_transform(condition)
            loader = create_eval_dataloader(
                manifest_path=args.manifest,
                split="val",
                transform=tf,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            y_t, _, y_p = run_split_inference(model=model, loader=loader, device=device)
            del loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cond_auroc = float(roc_auc_score(y_t, y_p))

        cond_dict: dict[str, Any] = {"auroc": round(cond_auroc, 4)}
        for c_key in candidate_keys_to_eval:
            th_val = threshold_results[c_key]["threshold"]
            m_cond = compute_metrics_at_threshold(y_t, y_p, threshold=th_val)
            cond_dict[c_key] = {
                "threshold": th_val,
                "accuracy": m_cond["accuracy"],
                "fpr": m_cond["fpr"],
                "tpr": m_cond["tpr_synthetic_recall"],
                "f1": m_cond["f1"],
            }

        multi_condition_robustness[condition] = cond_dict
        print(
            f"  [{cond_idx:>2}/17] {condition:<18} | AUROC: {cond_auroc*100:5.2f}% | "
            f"Acc(0.50): {cond_dict['th_default_050']['accuracy']*100:5.2f}% | "
            f"Acc(0.25): {cond_dict['th_fixed_025']['accuracy']*100:5.2f}% | "
            f"Acc(0.05): {cond_dict['th_fixed_005']['accuracy']*100:5.2f}%"
        )

    # 6. Aggregate Transformed Statistics for Candidate Thresholds
    trans_conds = [c for c in suite_conditions if c != "clean"]
    candidate_summary: dict[str, Any] = {}

    for c_key in candidate_keys_to_eval:
        th_val = threshold_results[c_key]["threshold"]
        clean_acc = multi_condition_robustness["clean"][c_key]["accuracy"]
        clean_fpr = multi_condition_robustness["clean"][c_key]["fpr"]
        clean_tpr = multi_condition_robustness["clean"][c_key]["tpr"]
        clean_f1 = multi_condition_robustness["clean"][c_key]["f1"]

        trans_accs = [multi_condition_robustness[c][c_key]["accuracy"] for c in trans_conds]
        trans_fprs = [multi_condition_robustness[c][c_key]["fpr"] for c in trans_conds]
        trans_tprs = [multi_condition_robustness[c][c_key]["tpr"] for c in trans_conds]
        trans_f1s = [multi_condition_robustness[c][c_key]["f1"] for c in trans_conds]

        worst_cond = min(
            trans_conds,
            key=lambda c: multi_condition_robustness[c][c_key]["accuracy"],
        )
        worst_acc = multi_condition_robustness[worst_cond][c_key]["accuracy"]
        comp_acc = multi_condition_robustness["composite_severe"][c_key]["accuracy"]
        comp_fpr = multi_condition_robustness["composite_severe"][c_key]["fpr"]

        candidate_summary[c_key] = {
            "threshold": th_val,
            "description": threshold_results[c_key]["description"],
            "clean_accuracy": clean_acc,
            "clean_fpr": clean_fpr,
            "clean_synthetic_recall": clean_tpr,
            "clean_f1": clean_f1,
            "mean_transformed_accuracy": round(float(np.mean(trans_accs)), 4),
            "mean_transformed_fpr": round(float(np.mean(trans_fprs)), 4),
            "mean_transformed_synthetic_recall": round(float(np.mean(trans_tprs)), 4),
            "mean_transformed_f1": round(float(np.mean(trans_f1s)), 4),
            "worst_case_condition": worst_cond,
            "worst_case_accuracy": worst_acc,
            "composite_severe_accuracy": comp_acc,
            "composite_severe_fpr": comp_fpr,
        }

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"\nCompleted canonical threshold analysis in {elapsed:.2f}s.")

    # 7. Assemble and Write Canonical JSON Report
    canonical_report: dict[str, Any] = {
        "metadata": {
            "title": "Verixa — Canonical Threshold Calibration Report (Hybrid V1)",
            "model_path": str(args.model_path),
            "manifest": str(args.manifest),
            "split": "val",
            "total_samples": total_samples,
            "authentic_samples": num_authentic,
            "synthetic_samples": num_synthetic,
            "clean_auroc": round(clean_auroc, 4),
            "clean_fpr_at_95_tpr": round(fpr_at_95_tpr, 4),
            "device": str(device),
            "elapsed_seconds": elapsed,
            "methodology": (
                "Recomputed strictly from raw inference on models/convnext_tiny_hybrid_fft.pt "
                "across the 6,001-image validation split of merged_manifest.csv. "
                "Operational thresholds parameterized by empirical authentic quantiles "
                "(1 - target_fpr). Evaluated across complete 17-condition development suite."
            ),
        },
        "score_statistics": score_statistics,
        "clean_threshold_performance": threshold_results,
        "multi_condition_robustness": multi_condition_robustness,
        "candidate_summary": candidate_summary,
    }

    payload = json.dumps(canonical_report, indent=2, sort_keys=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(payload, encoding="utf-8")
    print(
        f"Saved canonical threshold calibration report ({len(payload):,} bytes) "
        f"to: {args.report_out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
