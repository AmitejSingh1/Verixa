"""CLI script to evaluate development-set threshold calibration and horizontal-flip TTA.

Evaluates ONLY on the 6,001-image development validation suite across all 17 conditions.
Does NOT access or touch the held-out benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from verixa.evaluation.calibration import (
    find_development_thresholds,
    fit_temperature_scaling,
    predict_with_tta,
)
from verixa.evaluation.metrics import calculate_binary_metrics
from verixa.models.loader import load_model_from_checkpoint
from verixa.training.augmentations import EVAL_DISTORTION_SUITES, get_distortion_eval_transform
from verixa.training.dataset import create_eval_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate development threshold calibration and TTA on validation split."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/convnext_tiny_hybrid_fft.pt"),
        help="Path to model checkpoint (.pt).",
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
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/hybrid_calibrated_tta_eval.json"),
        help="Path to save evaluation report JSON.",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=Path("reports/final_hybrid_robustness_report.json"),
        help="Path to baseline evaluation JSON for delta comparison.",
    )
    return parser.parse_args()


def evaluate_split_tta(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_hflip: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference with TTA on dataloader, returning (y_true, y_logits, y_prob)."""
    all_targets: list[int] = []
    all_logits: list[float] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].long()
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                logits, probs = predict_with_tta(model, images, use_hflip=use_hflip)

            all_targets.extend(targets.tolist())
            all_logits.extend(logits.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return (
        np.array(all_targets, dtype=np.int64),
        np.array(all_logits, dtype=np.float64),
        np.array(all_probs, dtype=np.float64),
    )


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=================================================================")
    print(" Verixa — Development-Only Calibration & Horizontal-Flip TTA")
    print("=================================================================")
    print(f" Model Checkpoint: {args.model_path}")
    print(f" Manifest:         {args.manifest}")
    print(f" Device:           {device}")
    print(f" Report Out:       {args.report_out}")
    print("=================================================================")

    model = load_model_from_checkpoint(args.model_path, device=device)
    model.eval()

    # -------------------------------------------------------------------
    # Step 1: Calibration strictly on Clean Development Validation Split
    # -------------------------------------------------------------------
    print("\n--- Step 1: Deriving Calibrated Thresholds on Clean Dev Val ---")
    clean_loader = create_eval_dataloader(
        manifest_path=args.manifest,
        split="val",
        transform=get_distortion_eval_transform("clean"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    y_true_clean, logits_clean, probs_clean = evaluate_split_tta(
        model, clean_loader, device=device, use_hflip=True
    )

    clean_metrics_std = calculate_binary_metrics(y_true_clean, probs_clean, threshold=0.5)
    print(
        f"Clean TTA (th=0.5): Acc={clean_metrics_std['accuracy']*100:.2f}% | "
        f"AUROC={clean_metrics_std['auroc']*100:.2f}% | FPR={clean_metrics_std['fpr']*100:.2f}%"
    )

    dev_thresholds = find_development_thresholds(
        y_true_clean, probs_clean, target_fprs=[0.01, 0.02, 0.025, 0.03, 0.05]
    )
    temp_T = fit_temperature_scaling(logits_clean, y_true_clean)

    opt_acc_th = dev_thresholds["best_accuracy_threshold"]
    opt_f1_th = dev_thresholds["best_f1_threshold"]
    op_fpr3_th = dev_thresholds["operational_fpr_thresholds"]["th_fpr_le_3pct"]
    op_fpr5_th = dev_thresholds["operational_fpr_thresholds"]["th_fpr_le_5pct"]

    best_acc_pct = dev_thresholds["best_accuracy_on_dev"] * 100
    best_f1_pct = dev_thresholds["best_f1_on_dev"] * 100
    print(f"  - Optimal Accuracy Threshold: th={opt_acc_th:.4f} (Acc={best_acc_pct:.2f}%)")
    print(f"  - Optimal F1 Threshold:       th={opt_f1_th:.4f} (F1={best_f1_pct:.2f}%)")
    print(f"  - Operational FPR <= 3% Thresh: th={op_fpr3_th:.4f}")
    print(f"  - Operational FPR <= 5% Thresh: th={op_fpr5_th:.4f}")
    print(f"  - Fitted Temperature T:       T={temp_T:.4f}")

    # -------------------------------------------------------------------
    # Step 2: Full 17-Condition Suite Evaluation with TTA + Thresholds
    # -------------------------------------------------------------------
    print("\n--- Step 2: Evaluating All 17 Development Conditions with TTA ---")
    suite_keys = ["clean"] + list(EVAL_DISTORTION_SUITES.keys())
    results: dict[str, Any] = {}

    hdr = (
        f"{'Condition':<18} | {'AUROC (TTA)':<11} | {'Acc (th=0.5)':<12} | "
        f"{'Acc (Calib)':<12} | {'FPR (th=0.5)':<12} | {'FPR (Calib)':<11}"
    )
    print(hdr)
    print("-" * 88)

    for condition in suite_keys:
        transform = get_distortion_eval_transform(condition)
        loader = create_eval_dataloader(
            manifest_path=args.manifest,
            split="val",
            transform=transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        y_true, y_logits, y_probs = evaluate_split_tta(
            model, loader, device=device, use_hflip=True
        )

        # Standard metrics at 0.5 threshold with TTA
        m_std = calculate_binary_metrics(y_true, y_probs, threshold=0.5)
        # Calibrated metrics at optimal accuracy threshold with TTA
        m_calib = calculate_binary_metrics(y_true, y_probs, threshold=opt_acc_th)
        # Operational metrics at FPR <= 3% threshold with TTA
        m_fpr3 = calculate_binary_metrics(y_true, y_probs, threshold=op_fpr3_th)

        results[condition] = {
            "tta_auroc": m_std["auroc"],
            "tta_acc_th05": m_std["accuracy"],
            "tta_fpr_th05": m_std["fpr"],
            "tta_acc_calib": m_calib["accuracy"],
            "tta_fpr_calib": m_calib["fpr"],
            "tta_acc_fpr3": m_fpr3["accuracy"],
            "tta_fpr_fpr3": m_fpr3["fpr"],
            "metrics_std": m_std,
            "metrics_calib": m_calib,
        }

        print(
            f"{condition:<18} | {m_std['auroc']*100:<10.2f}% | "
            f"{m_std['accuracy']*100:<11.2f}% | {m_calib['accuracy']*100:<11.2f}% | "
            f"{m_std['fpr']*100:<11.2f}% | {m_calib['fpr']*100:<10.2f}%"
        )

    print("-" * 88)

    # -------------------------------------------------------------------
    # Step 3: Comparative Aggregates vs. Standard Hybrid Baseline
    # -------------------------------------------------------------------
    trans_keys = [c for c in suite_keys if c != "clean"]

    mean_trans_acc_std = float(np.mean([results[c]["tta_acc_th05"] for c in trans_keys]))
    mean_trans_acc_calib = float(np.mean([results[c]["tta_acc_calib"] for c in trans_keys]))
    mean_trans_auroc = float(np.mean([results[c]["tta_auroc"] for c in trans_keys]))

    worst_k_std = min(suite_keys, key=lambda k: results[k]["tta_acc_th05"])
    worst_acc_std = results[worst_k_std]["tta_acc_th05"]

    worst_k_calib = min(suite_keys, key=lambda k: results[k]["tta_acc_calib"])
    worst_acc_calib = results[worst_k_calib]["tta_acc_calib"]

    print("\n=================================================================")
    print(" Summary Across All 17 Development Conditions (with TTA):")
    print("=================================================================")
    print(f" Clean AUROC (TTA):           {results['clean']['tta_auroc']*100:.2f}%")
    print(f" Clean Accuracy (th=0.5):     {results['clean']['tta_acc_th05']*100:.2f}%")
    print(f" Clean Accuracy (Calib):      {results['clean']['tta_acc_calib']*100:.2f}%")
    print(f" Mean Transformed AUROC:      {mean_trans_auroc*100:.2f}%")
    print(f" Mean Transformed Acc (th=0.5): {mean_trans_acc_std*100:.2f}%")
    print(f" Mean Transformed Acc (Calib):  {mean_trans_acc_calib*100:.2f}%")
    print(f" Worst-Case Floor (th=0.5):   {worst_k_std} ({worst_acc_std*100:.2f}%)")
    print(f" Worst-Case Floor (Calib):    {worst_k_calib} ({worst_acc_calib*100:.2f}%)")

    comparison: dict[str, Any] = {}
    if args.compare_to.exists():
        base_data = json.loads(args.compare_to.read_text(encoding="utf-8"))
        base_res = base_data.get("results", base_data)

        ref_clean_acc = base_res.get("clean", {}).get("accuracy", 0.0)
        ref_clean_auroc = base_res.get("clean", {}).get("auroc", 0.0)
        ref_trans_acc = float(np.mean([base_res[c]["accuracy"] for c in trans_keys]))
        ref_trans_auroc = float(np.mean([base_res[c]["auroc"] for c in trans_keys]))

        print("\n=================================================================")
        print(" Delta vs. Non-TTA Hybrid Model:")
        print("=================================================================")
        d_clean_auroc = (results["clean"]["tta_auroc"] - ref_clean_auroc) * 100
        d_clean_acc = (results["clean"]["tta_acc_th05"] - ref_clean_acc) * 100
        d_trans_auroc = (mean_trans_auroc - ref_trans_auroc) * 100
        d_trans_acc = (mean_trans_acc_std - ref_trans_acc) * 100
        print(f" Clean AUROC Delta:           {d_clean_auroc:+.2f}%")
        print(f" Clean Accuracy Delta:        {d_clean_acc:+.2f}%")
        print(f" Mean Transformed AUROC Delta:{d_trans_auroc:+.2f}%")
        print(f" Mean Transformed Acc Delta:  {d_trans_acc:+.2f}%")
        print("=================================================================")

        comparison = {
            "ref_clean_accuracy": ref_clean_acc,
            "ref_clean_auroc": ref_clean_auroc,
            "ref_mean_trans_acc": round(ref_trans_acc, 4),
            "ref_mean_trans_auroc": round(ref_trans_auroc, 4),
            "delta_clean_accuracy": round(results["clean"]["tta_acc_th05"] - ref_clean_acc, 4),
            "delta_clean_auroc": round(results["clean"]["tta_auroc"] - ref_clean_auroc, 4),
            "delta_mean_trans_acc": round(mean_trans_acc_std - ref_trans_acc, 4),
            "delta_mean_trans_auroc": round(mean_trans_auroc - ref_trans_auroc, 4),
        }

    output_payload: dict[str, Any] = {
        "model_path": str(args.model_path),
        "manifest": str(args.manifest),
        "calibration": {
            "temperature_T": round(temp_T, 4),
            "thresholds": dev_thresholds,
        },
        "results": results,
        "summary": {
            "clean_auroc_tta": results["clean"]["tta_auroc"],
            "clean_acc_tta_th05": results["clean"]["tta_acc_th05"],
            "clean_acc_tta_calib": results["clean"]["tta_acc_calib"],
            "mean_transformed_auroc_tta": round(mean_trans_auroc, 4),
            "mean_transformed_acc_tta_th05": round(mean_trans_acc_std, 4),
            "mean_transformed_acc_tta_calib": round(mean_trans_acc_calib, 4),
            "worst_case_th05": {"condition": worst_k_std, "accuracy": worst_acc_std},
            "worst_case_calib": {"condition": worst_k_calib, "accuracy": worst_acc_calib},
        },
        "comparison": comparison,
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(output_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nSaved calibrated TTA evaluation report to: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
