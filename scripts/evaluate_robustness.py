"""CLI script to evaluate a model checkpoint across all 6 required distortion suites.

Computes clean baseline performance and degradation metrics across all 15 distortion
severities, with optional head-to-head comparison against another model report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from verixa.models.convnext import ConvNeXtBinaryClassifier
from verixa.training.augmentations import EVAL_DISTORTION_SUITES, get_distortion_eval_transform
from verixa.training.dataset import create_eval_dataloader
from verixa.training.trainer import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model checkpoint on clean validation and all distortion suites."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the model checkpoint .pt file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/merged_manifest.csv"),
        help="Path to manifest CSV containing validation split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size.",
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
        required=True,
        help="Path to save evaluation report JSON.",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="Optional path to baseline distortion eval JSON to compute deltas.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="all",
        help="Comma-separated list of conditions to evaluate, or 'all'.",
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model architecture and weights from checkpoint dictionary."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    arch = config.get("architecture", "convnext_tiny")

    if arch == "fft_standalone":
        from verixa.models.fft import FFTClassifier

        use_grayscale = config.get("use_grayscale", True)
        model: nn.Module = FFTClassifier(use_grayscale=use_grayscale)
    elif arch == "hybrid_rgb_fft":
        from verixa.models.hybrid import HybridRGBFFTClassifier

        model = HybridRGBFFTClassifier(pretrained=False, use_grayscale_fft=True)
    else:
        model = ConvNeXtBinaryClassifier(pretrained=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=================================================================")
    print(" Verixa — Robustness & Distortion Suite Evaluation")
    print("=================================================================")
    print(f" Model Checkpoint: {args.model_path}")
    print(f" Manifest:         {args.manifest}")
    print(f" Device:           {device}")
    print(f" Report Out:       {args.report_out}")
    print("=================================================================\n")

    model = load_model_from_checkpoint(args.model_path, device=device)
    criterion = nn.BCEWithLogitsLoss()

    if args.conditions == "all":
        suite_keys = ["clean"] + list(EVAL_DISTORTION_SUITES.keys())
    else:
        suite_keys = [c.strip() for c in args.conditions.split(",") if c.strip()]
    results: dict[str, Any] = {}

    clean_acc = 0.0
    clean_auroc = 0.0

    hdr = (
        f"{'Condition':<20} | {'Loss':<7} | {'Accuracy':<9} | "
        f"{'AUROC':<9} | {'FPR':<7} | {'FPR@95%':<8}"
    )
    print(hdr)
    print("-" * 75)

    for idx, condition in enumerate(suite_keys, start=1):
        transform = get_distortion_eval_transform(condition)
        loader = create_eval_dataloader(
            manifest_path=args.manifest,
            split="val",
            transform=transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        avg_loss, metrics = evaluate_model(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            epoch=idx,
            total_epochs=len(suite_keys),
        )

        acc = metrics["accuracy"]
        auroc = metrics["auroc"]
        fpr = metrics["fpr"]
        fpr95 = metrics["fpr_at_95_tpr"]

        if condition == "clean":
            clean_acc = acc
            clean_auroc = auroc
            delta_acc = 0.0
            delta_auroc = 0.0
        else:
            delta_acc = round(clean_acc - acc, 4)
            delta_auroc = round(clean_auroc - auroc, 4)

        metrics["delta_acc_from_clean"] = delta_acc
        metrics["delta_auroc_from_clean"] = delta_auroc
        results[condition] = metrics

        print(
            f"{condition:<20} | {avg_loss:<7.4f} | {acc*100:<8.2f}% | "
            f"{auroc*100:<8.2f}% | {fpr*100:<6.2f}% | {fpr95*100:<7.2f}%"
        )

    print("-" * 75)

    # Compute model aggregate statistics across evaluated conditions
    transformed_keys = [k for k in suite_keys if k != "clean"]
    mean_trans_acc = (
        float(np.mean([results[k]["accuracy"] for k in transformed_keys]))
        if transformed_keys else clean_acc
    )
    mean_trans_auroc = (
        float(np.mean([results[k]["auroc"] for k in transformed_keys]))
        if transformed_keys else clean_auroc
    )
    worst_case_k = min(suite_keys, key=lambda k: results[k]["accuracy"])
    worst_case_acc = results[worst_case_k]["accuracy"]

    print(f"\nModel Summary Across {len(suite_keys)} Condition(s):")
    print(f"  Clean Accuracy:          {clean_acc*100:.2f}% | AUROC: {clean_auroc*100:.2f}%")
    if transformed_keys:
        print(f"  Mean Transformed Acc:    {mean_trans_acc*100:.2f}%")
        print(f"  Mean Transformed AUROC:  {mean_trans_auroc*100:.2f}%")
        print(f"  Worst-Case Condition:    {worst_case_k} ({worst_case_acc*100:.2f}%)")

    # Comparison and Decision Checkpoint analysis if comparison report is provided
    comparison: dict[str, Any] = {}
    if args.compare_to is not None and args.compare_to.exists():
        baseline_report = json.loads(args.compare_to.read_text(encoding="utf-8"))
        base_results = baseline_report.get("results", baseline_report)

        print("\n=================================================================")
        print(" Comparative Analysis vs. Reference Model")
        print("=================================================================")

        clean_base_acc = base_results.get("clean", {}).get("accuracy", 0.0)
        clean_drop = clean_base_acc - clean_acc

        common_trans = [k for k in transformed_keys if k in base_results]
        ref_trans_acc = (
            float(np.mean([base_results[k]["accuracy"] for k in common_trans]))
            if common_trans else 0.0
        )
        cur_trans_acc = (
            float(np.mean([results[k]["accuracy"] for k in common_trans]))
            if common_trans else 0.0
        )
        trans_acc_gain = cur_trans_acc - ref_trans_acc

        if common_trans:
            ref_worst_k = min(common_trans, key=lambda k: base_results[k]["accuracy"])
            ref_worst_acc = base_results[ref_worst_k]["accuracy"]
        else:
            ref_worst_k = "N/A"
            ref_worst_acc = 0.0

        comparison = {
            "clean_reference_acc": clean_base_acc,
            "clean_current_acc": clean_acc,
            "clean_drop": round(clean_drop, 4),
            "mean_trans_acc_ref": round(ref_trans_acc, 4),
            "mean_trans_acc_cur": round(cur_trans_acc, 4),
            "mean_trans_acc_gain": round(trans_acc_gain, 4),
            "worst_case_cur": {"condition": worst_case_k, "accuracy": worst_case_acc},
            "worst_case_ref": {"condition": ref_worst_k, "accuracy": ref_worst_acc},
        }

        print(f" Clean Accuracy Delta:    {clean_acc - clean_base_acc:+.2f}%")
        print(f" Mean Transformed Delta:  {trans_acc_gain:+.2f}%")
        print(
            f" Worst-Case Accuracy:     {worst_case_acc*100:.2f}% "
            f"(Ref: {ref_worst_acc*100:.2f}%)"
        )
        print("=================================================================")

    output_payload = {
        "model_path": str(args.model_path),
        "manifest": str(args.manifest),
        "device": str(device),
        "results": results,
        "comparison": comparison,
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(output_payload, indent=2, sort_keys=True)
    args.report_out.write_text(report_json, encoding="utf-8")
    print(f"\nSaved distortion evaluation report to: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
