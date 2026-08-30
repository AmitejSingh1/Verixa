"""High-level Python inference API for Verixa AI-generated image detection."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from verixa.models.loader import load_model_from_checkpoint
from verixa.training.dataset import DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def get_default_transform() -> transforms.Compose:
    """Standard ImageNet evaluation transform (bicubic 224x224 resize and normalization)."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_IMAGENET_MEAN, std=DEFAULT_IMAGENET_STD),
        ]
    )


class VerixaPredictor:
    """Self-contained predictor for single-image and batch AI-generated image detection."""

    def __init__(
        self,
        checkpoint_path: Path | str = "models/convnext_tiny_hybrid_fft.pt",
        device: str | torch.device | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = threshold

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        self.model = load_model_from_checkpoint(self.checkpoint_path, device=self.device)
        self.model.eval()
        self.transform = get_default_transform()

    def predict_image(self, image_input: Path | str | Image.Image) -> dict[str, Any]:
        """Classify a single image, returning class, probability, and confidence."""
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            try:
                with Image.open(img_path) as img:
                    rgb_img = img.convert("RGB")
                    tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)
            except (UnidentifiedImageError, OSError) as e:
                raise ValueError(f"Failed to decode image {img_path}: {e}") from e
            source_desc = str(img_path)
        elif isinstance(image_input, Image.Image):
            rgb_img = image_input.convert("RGB")
            tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)
            source_desc = "<PIL.Image>"
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                logits = self.model(tensor).squeeze(-1)
                prob = float(torch.sigmoid(logits).item())

        pred_label = 1 if prob >= self.threshold else 0
        class_name = "AI-Generated" if pred_label == 1 else "Authentic"
        conf = prob if pred_label == 1 else (1.0 - prob)

        return {
            "filepath": source_desc,
            "prediction": pred_label,
            "class_name": class_name,
            "probability_synthetic": round(prob, 6),
            "confidence_pct": round(conf * 100.0, 2),
            "threshold": self.threshold,
        }

    def predict_batch(
        self,
        image_paths: list[Path | str],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """Classify a list of image paths in batches with GPU acceleration."""
        results: list[dict[str, Any]] = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size

        with torch.no_grad():
            for b_idx in range(total_batches):
                start = b_idx * batch_size
                end = min(start + batch_size, len(image_paths))
                batch_paths = [Path(p) for p in image_paths[start:end]]

                valid_paths: list[Path] = []
                tensors: list[torch.Tensor] = []

                for p in batch_paths:
                    try:
                        with Image.open(p) as img:
                            tensors.append(self.transform(img.convert("RGB")))
                            valid_paths.append(p)
                    except Exception:
                        continue

                if not tensors:
                    continue

                batch_tensor = torch.stack(tensors).to(self.device)
                with torch.amp.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                    logits = self.model(batch_tensor).squeeze(-1)
                    probs = torch.sigmoid(logits).cpu().tolist()

                if isinstance(probs, float):
                    probs = [probs]

                for p, prob in zip(valid_paths, probs, strict=True):
                    pred_label = 1 if prob >= self.threshold else 0
                    class_name = "AI-Generated" if pred_label == 1 else "Authentic"
                    conf = prob if pred_label == 1 else (1.0 - prob)
                    results.append(
                        {
                            "filepath": str(p),
                            "prediction": pred_label,
                            "class_name": class_name,
                            "probability_synthetic": round(prob, 6),
                            "confidence_pct": round(conf * 100.0, 2),
                            "threshold": self.threshold,
                        }
                    )

        return results

    def save_predictions_to_csv(
        self,
        results: list[dict[str, Any]],
        output_csv: Path | str,
    ) -> None:
        """Write prediction records to a standard CSV file."""
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "filepath",
            "prediction",
            "class_name",
            "probability_synthetic",
            "confidence_pct",
            "threshold",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
