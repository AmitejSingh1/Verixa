"""CLI script for single-pass held-out benchmark evaluation.

Evaluates either the Primary Hybrid model or Fallback model on the isolated
held-out benchmark dataset (cocoval2017 authentic + dalle3 synthetic),
computing ground-truth metrics and generating submission predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from verixa.evaluation.metrics import calculate_binary_metrics
from verixa.models.loader import load_model_from_checkpoint
from verixa.training.dataset import DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD


class HeldOutBenchmarkDataset(Dataset):
    """Dataset for held-out benchmark images with ground truth labels inferred from directory."""

    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            tensor = self.transform(rgb_img)
        return tensor, label, str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-pass evaluation on the isolated held-out benchmark."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("benchmark_dataset"),
        help="Path to benchmark_dataset containing dalle3/ and cocoval2017/ subdirectories.",
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
        help="DataLoader worker subprocesses.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary classification.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/held_out_benchmark_eval.json"),
        help="Output JSON evaluation report path.",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("predictions_held_out.csv"),
        help="Output CSV path for submission predictions.",
    )
    return parser.parse_args()


def discover_benchmark_samples(benchmark_dir: Path) -> list[tuple[Path, int]]:
    """Discover authentic (cocoval2017) and synthetic (dalle3) images."""
    coco_dir = benchmark_dir / "cocoval2017"
    dalle_dir = benchmark_dir / "dalle3"

    if not coco_dir.exists() or not dalle_dir.exists():
        msg = f"Benchmark directory {benchmark_dir} must contain 'cocoval2017' and 'dalle3'."
        raise FileNotFoundError(msg)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    # Authentic images = label 0
    coco_files = sorted([p for p in coco_dir.rglob("*") if p.suffix.lower() in valid_exts])
    # Synthetic images = label 1
    dalle_files = sorted([p for p in dalle_dir.rglob("*") if p.suffix.lower() in valid_exts])

    samples: list[tuple[Path, int]] = []
    for p in coco_files:
        samples.append((p, 0))
    for p in dalle_files:
        samples.append((p, 1))

    return samples


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=================================================================")
    print(" Verixa — Held-Out Benchmark Single-Pass Evaluation")
    print("=================================================================")
    print(f" Model Checkpoint: {args.model_path}")
    print(f" Benchmark Dir:    {args.benchmark_dir}")
    print(f" Report Out:       {args.report_out}")
    print(f" Predictions Out:  {args.predictions_out}")
    print(f" Device:           {device}")
    print("=================================================================")

    # 1. Discover samples
    samples = discover_benchmark_samples(args.benchmark_dir)
    coco_count = sum(1 for _, lbl in samples if lbl == 0)
    dalle_count = sum(1 for _, lbl in samples if lbl == 1)
    total_count = len(samples)

    print(f"Discovered {total_count:,} total benchmark images:")
    print(f"  - Authentic (cocoval2017): {coco_count:,}")
    print(f"  - Synthetic (dalle3):      {dalle_count:,}")

    # 2. Setup model & data
    model = load_model_from_checkpoint(args.model_path, device=device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_IMAGENET_MEAN, std=DEFAULT_IMAGENET_STD),
        ]
    )

    dataset = HeldOutBenchmarkDataset(samples, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Single-pass evaluation loop
    all_targets: list[int] = []
    all_probs: list[float] = []
    all_preds: list[int] = []
    all_paths: list[str] = []

    total_batches = len(dataloader)
    start_time = time.perf_counter()

    print("\nExecuting single-pass inference across benchmark...")

    with torch.no_grad():
        for batch_idx, (images, targets, paths) in enumerate(dataloader, start=1):
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().tolist()

            if isinstance(probs, float):
                probs = [probs]

            targets_list = targets.tolist()
            preds_list = [1 if p >= args.threshold else 0 for p in probs]

            all_targets.extend(targets_list)
            all_probs.extend(probs)
            all_preds.extend(preds_list)
            all_paths.extend(paths)

            # In-place ASCII progress
            pct = int((batch_idx / total_batches) * 100)
            filled = int(30 * (batch_idx / total_batches))
            bar = "#" * filled + "-" * (30 - filled)
            print(
                f"\rBatch [{batch_idx:>4}/{total_batches}] [{bar}] {pct:>3}%",
                end="",
                flush=True,
            )

    elapsed_s = time.perf_counter() - start_time
    fps = total_count / max(elapsed_s, 1e-4)
    print(f"\nInference complete in {elapsed_s:.2f}s ({fps:.1f} images/s).")

    # 4. Compute metrics
    binary_metrics = calculate_binary_metrics(
        y_true=all_targets,
        y_prob=all_probs,
        threshold=args.threshold,
    )

    # Per-class accuracy
    auth_correct = sum(
        1 for t, p in zip(all_targets, all_preds, strict=True) if t == 0 and p == 0
    )
    synth_correct = sum(
        1 for t, p in zip(all_targets, all_preds, strict=True) if t == 1 and p == 1
    )
    auth_acc = auth_correct / max(coco_count, 1)
    synth_acc = synth_correct / max(dalle_count, 1)

    print("\n=================================================================")
    print(" Held-Out Benchmark Evaluation Results")
    print("=================================================================")
    print(f" Overall Accuracy:        {binary_metrics['accuracy']*100:.2f}%")
    print(f" Overall AUROC:           {binary_metrics['auroc']*100:.2f}%")
    print(f" False Positive Rate:     {binary_metrics['fpr']*100:.2f}%")
    print(f" False Negative Rate:     {binary_metrics['fnr']*100:.2f}%")
    print(f" FPR at 95% Recall:       {binary_metrics['fpr_at_95_tpr']*100:.2f}%")
    print(f" Authentic (COCO) Acc:    {auth_acc*100:.2f}% ({auth_correct:,}/{coco_count:,})")
    print(f" Synthetic (DALL-E) Acc:  {synth_acc*100:.2f}% ({synth_correct:,}/{dalle_count:,})")
    cm = binary_metrics["confusion_matrix"]
    print(
        f" Confusion Matrix:        TP={cm['true_positives']:,} | TN={cm['true_negatives']:,} "
        f"| FP={cm['false_positives']:,} | FN={cm['false_negatives']:,}"
    )
    print("=================================================================")

    # 5. Save report JSON
    report_payload: dict[str, Any] = {
        "model_path": str(args.model_path),
        "benchmark_dir": str(args.benchmark_dir),
        "total_images": total_count,
        "authentic_count": coco_count,
        "synthetic_count": dalle_count,
        "elapsed_seconds": round(elapsed_s, 2),
        "throughput_images_per_s": round(fps, 1),
        "threshold": args.threshold,
        "metrics": binary_metrics,
        "per_class": {
            "authentic_coco_accuracy": round(auth_acc, 4),
            "synthetic_dalle_accuracy": round(synth_acc, 4),
        },
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved evaluation report to: {args.report_out}")

    # 6. Save submission predictions CSV
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.predictions_out, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "probability", "prediction"])
        writer.writeheader()
        for p, prob, pred in zip(all_paths, all_probs, all_preds, strict=True):
            writer.writerow(
                {
                    "filepath": p,
                    "probability": round(prob, 6),
                    "prediction": pred,
                }
            )
    print(f"Saved submission predictions to: {args.predictions_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
