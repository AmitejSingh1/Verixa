"""CLI script to train the Phase 2 RGB ConvNeXt-Tiny baseline model."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from verixa.models.convnext import ConvNeXtBinaryClassifier, freeze_backbone_stages
from verixa.training.dataset import create_dataloaders
from verixa.training.trainer import fit_convnext_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Phase 2 RGB ConvNeXt-Tiny baseline model on clean images."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/merged_manifest.csv"),
        help="Path to the merged manifest CSV with train/val splits.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--freeze-stages",
        type=int,
        default=2,
        help="Freeze early stages up to this index (0: stem, 1: stage 0-1, 2: stage 0-2).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping patience in epochs without validation AUROC improvement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker subprocesses.",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("models/convnext_tiny_baseline.pt"),
        help="Output path for best checkpoint.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/baseline_clean_eval.json"),
        help="Output path for evaluation report JSON.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    print("=================================================================")
    print(" Verixa — Phase 2: RGB ConvNeXt-Tiny Baseline Model Training")
    print("=================================================================")
    print(f" Manifest:       {args.manifest}")
    print(f" Max Epochs:     {args.epochs}")
    print(f" Patience:       {args.patience} epochs")
    print(f" Batch size:     {args.batch_size}")
    print(f" Learning rate:  {args.lr}")
    print(f" Frozen stages:  features[0..{args.freeze_stages}]")
    print(f" Random seed:    {args.seed}")
    print(f" Checkpoint out: {args.checkpoint_out}")
    print(f" Report out:     {args.report_out}")
    print("=================================================================")

    # 1. Dataloaders
    train_loader, val_loader = create_dataloaders(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(
        f"Loaded {len(train_loader.dataset):,} train samples, "
        f"{len(val_loader.dataset):,} val samples."
    )

    # 2. Model & Freezing
    model = ConvNeXtBinaryClassifier(pretrained=True)
    freeze_backbone_stages(model, freeze_up_to_stage=args.freeze_stages)

    # 3. Assemble full run configuration for checkpoint reproducibility
    run_config = {
        "architecture": "convnext_tiny",
        "pretrained": True,
        "dropout": 0.3,
        "freeze_stages": args.freeze_stages,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "manifest": str(args.manifest),
        "image_size": [224, 224],
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
    }

    # 4. Train & Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = fit_convnext_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_path=args.checkpoint_out,
        config=run_config,
        patience=args.patience,
    )

    # 5. Save report
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nTraining complete. Saved clean evaluation report to: {args.report_out}")
    print(f"Saved best model checkpoint to: {args.checkpoint_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

