"""CLI script to generate submission prediction CSVs for arbitrary evaluation sets.

Computes calibrated probabilities and binary classifications using any trained
checkpoint (Hybrid RGB+FFT, Robust RGB, or Baseline).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from verixa.models.loader import load_model_from_checkpoint
from verixa.training.dataset import DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prediction CSV for an image manifest or image directory."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to manifest CSV containing filepath column.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional path to directory of evaluation images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary classification.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split filter (e.g. 'val', 'test', 'train').",
    )
    return parser.parse_args()


def get_image_paths(
    manifest_path: Path | None,
    image_dir: Path | None,
    split: str | None = None,
) -> list[Path]:
    """Retrieve list of image paths from either a manifest CSV or image directory."""
    if manifest_path is not None and manifest_path.exists():
        paths: list[Path] = []
        with open(manifest_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split is not None and "split" in row and row["split"] != split:
                    continue
                raw_path = row.get("image_path") or row.get("filepath")
                if raw_path is None:
                    raise KeyError("Manifest must contain 'image_path' or 'filepath' column.")
                paths.append(Path(raw_path))
        return paths

    if image_dir is not None and image_dir.exists():
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        return [p for p in image_dir.rglob("*") if p.suffix.lower() in valid_exts]

    raise ValueError("Must provide either a valid --manifest or --image-dir.")


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=================================================================")
    print(" Verixa — Competition Submission Prediction Generator")
    print("=================================================================")
    print(f" Model Checkpoint: {args.model_path}")
    print(f" Output File:      {args.output}")
    print(f" Device:           {device}")
    print("=================================================================")

    image_paths = get_image_paths(args.manifest, args.image_dir, split=args.split)
    print(f"Discovered {len(image_paths):,} images for inference.")

    model = load_model_from_checkpoint(args.model_path, device=device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_IMAGENET_MEAN, std=DEFAULT_IMAGENET_STD),
        ]
    )

    results: list[dict[str, str | float | int]] = []
    total_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for b_idx in range(total_batches):
            start = b_idx * args.batch_size
            end = min(start + args.batch_size, len(image_paths))
            batch_paths = image_paths[start:end]

            tensors: list[torch.Tensor] = []
            for p in batch_paths:
                with Image.open(p) as img:
                    rgb_img = img.convert("RGB")
                    tensors.append(transform(rgb_img))

            batch_tensor = torch.stack(tensors).to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(batch_tensor).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().tolist()

            if isinstance(probs, float):
                probs = [probs]

            for p, prob in zip(batch_paths, probs, strict=True):
                pred_label = 1 if prob >= args.threshold else 0
                results.append(
                    {
                        "filepath": str(p),
                        "probability": round(prob, 6),
                        "prediction": pred_label,
                    }
                )

            if (b_idx + 1) % max(1, total_batches // 10) == 0 or (b_idx + 1) == total_batches:
                pct = (b_idx + 1) / total_batches * 100
                print(f"Processed {end:,}/{len(image_paths):,} images ({pct:.1f}%)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "probability", "prediction"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSuccessfully generated submission file: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
