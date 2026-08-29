"""PyTorch Dataset and DataLoader implementations for Verixa."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from verixa.data.manifest import read_manifest

DEFAULT_IMAGENET_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_clean_transforms() -> transforms.Compose:
    """Return standard ImageNet normalization transform for clean images."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_IMAGENET_MEAN, std=DEFAULT_IMAGENET_STD),
        ]
    )


class VerixaDataset(Dataset):
    """Dataset loading 224x224 JPEGs from a Verixa manifest CSV for a specific split."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.rows = rows
        self.transform = transform if transform is not None else get_clean_transforms()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = Path(row["image_path"])

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                tensor = self.transform(img)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image at {image_path}") from exc

        label = float(row["label"])

        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "image_path": str(image_path),
            "source_dataset": row.get("source_dataset", ""),
            "original_id": row.get("original_id", ""),
            "sha256": row.get("sha256", ""),
        }


def create_dataloaders(
    manifest_path: Path,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 1337,
    train_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    val_transform: Callable[[Image.Image], torch.Tensor] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from a merged manifest with assigned splits."""
    all_rows = read_manifest(manifest_path)
    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_rows = [r for r in all_rows if r["split"] == "val"]

    if not train_rows or not val_rows:
        raise ValueError(
            f"Manifest {manifest_path} must have 'train' and 'val' splits assigned before training."
        )

    t_train = train_transform if train_transform is not None else get_clean_transforms()
    t_val = val_transform if val_transform is not None else get_clean_transforms()

    train_dataset = VerixaDataset(train_rows, transform=t_train)
    val_dataset = VerixaDataset(val_rows, transform=t_val)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def create_eval_dataloader(
    manifest_path: Path,
    split: str = "val",
    transform: Callable[[Image.Image], torch.Tensor] | None = None,
    batch_size: int = 32,
    num_workers: int = 2,
) -> DataLoader:
    """Create an evaluation DataLoader for a specific split with an arbitrary transform."""
    all_rows = read_manifest(manifest_path)
    eval_rows = [r for r in all_rows if r["split"] == split]

    if not eval_rows:
        raise ValueError(f"No samples found with split='{split}' in {manifest_path}")

    dataset = VerixaDataset(eval_rows, transform=transform or get_clean_transforms())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

