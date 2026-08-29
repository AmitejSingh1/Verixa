"""CLI script to evaluate a model checkpoint across all 6 required distortion suites.

Computes clean baseline performance and degradation metrics across all 15 distortion
severities, with optional head-to-head comparison against another model report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model architecture and weights from checkpoint dictionary."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
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

    suite_keys = ["clean"] + list(EVAL_DISTORTION_SUITES.keys())
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

    # Comparison and Decision Checkpoint #1 analysis if baseline report is provided
    comparison: dict[str, Any] = {}
    if args.compare_to is not None and args.compare_to.exists():
        baseline_report = json.loads(args.compare_to.read_text(encoding="utf-8"))
        base_results = baseline_report.get("results", baseline_report)

        print("\n=================================================================")
        print(" Comparative Analysis vs. Baseline (Decision Checkpoint #1)")
        print("=================================================================")

        jpeg30_base_acc = base_results.get("jpeg_q30", {}).get("accuracy", 0.0)
        jpeg30_rob_acc = results.get("jpeg_q30", {}).get("accuracy", 0.0)
        jpeg30_delta = jpeg30_rob_acc - jpeg30_base_acc

        blur2_base_acc = base_results.get("blur_sigma2.0", {}).get("accuracy", 0.0)
        blur2_rob_acc = results.get("blur_sigma2.0", {}).get("accuracy", 0.0)
        blur2_delta = blur2_rob_acc - blur2_base_acc

        clean_base_acc = base_results.get("clean", {}).get("accuracy", 0.0)
        clean_drop = clean_base_acc - clean_acc

        jpeg30_pass = jpeg30_delta >= 0.15
        blur2_pass = blur2_delta >= 0.15
        clean_pass = clean_drop < 0.03

        checkpoint_1_pass = (jpeg30_pass or blur2_pass) and clean_pass

        comparison = {
            "clean_baseline_acc": clean_base_acc,
            "clean_robust_acc": clean_acc,
            "clean_drop": round(clean_drop, 4),
            "jpeg_q30_baseline_acc": jpeg30_base_acc,
            "jpeg_q30_robust_acc": jpeg30_rob_acc,
            "jpeg_q30_improvement": round(jpeg30_delta, 4),
            "blur_sigma2_baseline_acc": blur2_base_acc,
            "blur_sigma2_robust_acc": blur2_rob_acc,
            "blur_sigma2_improvement": round(blur2_delta, 4),
            "clean_drop_acceptable (<3%)": clean_pass,
            "severe_distortion_gain (>=15%)": (jpeg30_pass or blur2_pass),
            "checkpoint_1_passed": checkpoint_1_pass,
        }

        print(f" Clean Accuracy Drop:     {clean_drop*100:+.2f}%  (Target: < 3.0%)")
        print(f" JPEG Q=30 Gain:          {jpeg30_delta*100:+.2f}% (Target: >= +15.0%)")
        print(f" Blur sigma=2.0 Gain:     {blur2_delta*100:+.2f}% (Target: >= +15.0%)")
        status_str = "PASSED (Fallback Designated)" if checkpoint_1_pass else "REVIEW REQUIRED"
        print(f" Decision Checkpoint #1:  {status_str}")
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
