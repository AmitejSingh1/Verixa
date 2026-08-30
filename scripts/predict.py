"""Production inference CLI for Verixa AI-generated image detection.

Supports single-image analysis, batch directory processing, and manifest CSVs
using the champion Hybrid RGB+FFT model (convnext_tiny_hybrid_fft.pt).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from verixa.models.loader import load_model_from_checkpoint
from verixa.training.dataset import DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def get_inference_transform() -> transforms.Compose:
    """Standard ImageNet evaluation transform with bicubic resizing to 224x224."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_IMAGENET_MEAN, std=DEFAULT_IMAGENET_STD),
        ]
    )


def predict_single_image(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    transform: transforms.Compose,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Execute single-image inference returning probability, prediction, and confidence."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            tensor = transform(rgb_img).unsqueeze(0).to(device)
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Failed to decode image {image_path}: {e}") from e

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(tensor).squeeze(-1)
            prob = float(torch.sigmoid(logits).item())

    pred_label = 1 if prob >= threshold else 0
    class_name = "AI-Generated" if pred_label == 1 else "Authentic"
    confidence = prob if pred_label == 1 else (1.0 - prob)

    return {
        "filepath": str(image_path),
        "prediction": pred_label,
        "class_name": class_name,
        "probability_synthetic": round(prob, 6),
        "confidence_pct": round(confidence * 100.0, 2),
        "threshold": threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verixa — Production AI-Generated Image Detector CLI"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=Path,
        help="Path to a single image file for inference.",
    )
    input_group.add_argument(
        "--image-dir",
        type=Path,
        help="Path to a directory containing images to evaluate in batch.",
    )
    input_group.add_argument(
        "--manifest",
        type=Path,
        help="Path to a CSV manifest containing an 'image_path' or 'filepath' column.",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/convnext_tiny_hybrid_fft.pt"),
        help="Path to trained model checkpoint (.pt). Default: models/convnext_tiny_hybrid_fft.pt",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for AI-generated classification (default: 0.5, locked).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for directory or manifest inference (default: 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Compute device ('cuda' or 'cpu'). Default: auto-detect.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save batch predictions (.csv or .json).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode: suppress non-essential output.",
    )
    return parser.parse_args()


def collect_images_from_source(args: argparse.Namespace) -> list[Path]:
    """Collect valid image paths from directory or manifest."""
    if args.image_dir is not None:
        if not args.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
        paths = [
            p
            for p in args.image_dir.rglob("*")
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and p.is_file()
        ]
        return sorted(paths)

    if args.manifest is not None:
        if not args.manifest.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {args.manifest}")
        paths = []
        with open(args.manifest, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_path = row.get("image_path") or row.get("filepath")
                if raw_path:
                    paths.append(Path(raw_path))
        return paths

    return []


def main() -> int:
    args = parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.model_path.exists():
        print(f"Error: Model checkpoint not found at {args.model_path}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("=================================================================")
        print(" Verixa — AI-Generated Image Detection Inference CLI")
        print("=================================================================")
        print(f" Model Checkpoint: {args.model_path}")
        print(f" Compute Device:   {device}")
        print(f" Decision Cut-Off: {args.threshold:.4f}")
        print("=================================================================")

    # Load model
    t0 = time.perf_counter()
    model = load_model_from_checkpoint(args.model_path, device=device)
    model.eval()
    transform = get_inference_transform()

    # 1. Single Image Mode
    if args.image is not None:
        try:
            res = predict_single_image(
                image_path=args.image,
                model=model,
                device=device,
                transform=transform,
                threshold=args.threshold,
            )
        except Exception as e:
            print(f"Error processing image {args.image}: {e}", file=sys.stderr)
            return 1

        elapsed = (time.perf_counter() - t0) * 1000.0

        if not args.quiet:
            print("\nAnalysis Result:")
            print(f"  File:           {res['filepath']}")
            print(f"  Classification: {res['class_name'].upper()} (Class {res['prediction']})")
            print(f"  Probability:    {res['probability_synthetic']:.4f} (Synthetic score)")
            print(f"  Confidence:     {res['confidence_pct']:.1f}%")
            print(f"  Inference Time: {elapsed:.1f} ms")
        else:
            print(f"{res['class_name']},{res['probability_synthetic']:.6f}")

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.suffix.lower() == ".json":
                args.output.write_text(json.dumps([res], indent=2), encoding="utf-8")
            else:
                with open(args.output, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(res.keys()))
                    writer.writeheader()
                    writer.writerow(res)

        return 0

    # 2. Batch Mode (Directory or Manifest)
    image_paths = collect_images_from_source(args)
    if not image_paths:
        print("Error: No valid images discovered in specified source.", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Discovered {len(image_paths):,} image(s) for batch evaluation.")

    results: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []
    total_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for b_idx in range(total_batches):
            start = b_idx * args.batch_size
            end = min(start + args.batch_size, len(image_paths))
            batch_paths = image_paths[start:end]

            valid_paths: list[Path] = []
            tensors: list[torch.Tensor] = []

            for p in batch_paths:
                try:
                    with Image.open(p) as img:
                        rgb_img = img.convert("RGB")
                        tensors.append(transform(rgb_img))
                        valid_paths.append(p)
                except Exception as e:
                    errors.append((str(p), str(e)))

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors).to(device)
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(batch_tensor).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().tolist()

            if isinstance(probs, float):
                probs = [probs]

            for p, prob in zip(valid_paths, probs, strict=True):
                pred_label = 1 if prob >= args.threshold else 0
                class_name = "AI-Generated" if pred_label == 1 else "Authentic"
                conf = prob if pred_label == 1 else (1.0 - prob)
                results.append(
                    {
                        "filepath": str(p),
                        "prediction": pred_label,
                        "class_name": class_name,
                        "probability": round(prob, 6),
                        "confidence_pct": round(conf * 100.0, 2),
                    }
                )

            if not args.quiet and (b_idx + 1) % max(1, total_batches // 10) == 0:
                pct = (b_idx + 1) / total_batches * 100
                print(f"Processed {end:,}/{len(image_paths):,} images ({pct:.1f}%)...")

    elapsed = time.perf_counter() - t0
    num_fake = sum(1 for r in results if r["prediction"] == 1)
    num_real = len(results) - num_fake
    throughput = len(results) / max(0.001, elapsed)

    if not args.quiet:
        print("\n=================================================================")
        print(" Batch Evaluation Summary")
        print("=================================================================")
        print(f" Total Processed:      {len(results):,}")
        print(f" AI-Generated Flags:   {num_fake:,} ({num_fake/max(1, len(results))*100:.1f}%)")
        print(f" Authentic Flags:      {num_real:,} ({num_real/max(1, len(results))*100:.1f}%)")
        if errors:
            print(f" Corrupted / Skipped:  {len(errors):,}")
        print(f" Total Time:           {elapsed:.2f}s ({throughput:.1f} img/s)")
        print("=================================================================")

    # Write output if specified
    out_path = args.output or Path("predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    else:
        fieldnames = ["filepath", "probability", "prediction", "class_name", "confidence_pct"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    if not args.quiet:
        print(f"Saved predictions to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
