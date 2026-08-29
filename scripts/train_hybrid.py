"""CLI script to train the Hybrid RGB + FFT model on the frozen 30K dataset.

Uses identical transformation-aware training (p=0.8), identical seed (1337),
identical hyperparameters, and identical 30K train/val split as Phase 3.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from verixa.models.hybrid import HybridRGBFFTClassifier, freeze_hybrid_backbone_stages
from verixa.training.augmentations import get_robust_training_transforms
from verixa.training.dataset import create_dataloaders
from verixa.training.trainer import fit_convnext_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Hybrid RGB+FFT model with transformation-aware augmentations."
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
        help="Freeze early stages of spatial backbone (features[0..2]).",
    )
    parser.add_argument(
        "--aug-prob",
        type=float,
        default=0.8,
        help="Transformation-aware augmentation probability during training.",
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
        default=Path("models/convnext_tiny_hybrid_fft.pt"),
        help="Output path for best checkpoint.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/hybrid_fft_clean_eval.json"),
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
    print(" Verixa — Phase 4: Hybrid RGB+FFT Model Training")
    print("=================================================================")
    print(f" Manifest:       {args.manifest}")
    print(" Architecture:   HybridRGBFFTClassifier (ConvNeXt-Tiny + 2D FFT CNN)")
    print(f" Augmentation:   Transformation-Aware (p={args.aug_prob})")
    print(f" Max Epochs:     {args.epochs}")
    print(f" Patience:       {args.patience} epochs")
    print(f" Batch size:     {args.batch_size}")
    print(f" Learning rate:  {args.lr}")
    print(f" Frozen stages:  spatial_backbone[0..{args.freeze_stages}]")
    print(f" Random seed:    {args.seed}")
    print(f" Checkpoint out: {args.checkpoint_out}")
    print(f" Report out:     {args.report_out}")
    print("=================================================================")

    # 1. Dataloaders with identical Phase 3 transformation-aware augmentations
    train_transform = get_robust_training_transforms(p=args.aug_prob, seed=args.seed)
    train_loader, val_loader = create_dataloaders(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_transform=train_transform,
    )
    print(
        f"Loaded {len(train_loader.dataset):,} train samples, "
        f"{len(val_loader.dataset):,} val samples."
    )

    # 2. Hybrid Model & Freezing
    model = HybridRGBFFTClassifier(pretrained=True, dropout=0.3, use_grayscale_fft=True)
    freeze_hybrid_backbone_stages(model, freeze_up_to_stage=args.freeze_stages)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. Assemble full run configuration
    run_config = {
        "architecture": "hybrid_rgb_fft",
        "spatial_backbone": "convnext_tiny",
        "spectral_branch": "fft_cnn_512d",
        "fusion": "concatenation_1280d",
        "pretrained_spatial": True,
        "dropout": 0.3,
        "freeze_stages": args.freeze_stages,
        "robust_augment": True,
        "aug_prob": args.aug_prob,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "manifest": str(args.manifest),
        "trainable_parameters": trainable_params,
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
        model_name="HybridRGBFFTClassifier",
    )

    # 5. Save report
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nTraining complete. Saved clean evaluation report to: {args.report_out}")
    print(f"Saved best model checkpoint to: {args.checkpoint_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
