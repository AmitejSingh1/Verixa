"""CLI script for training Hybrid V2 RGB+FFT model with representation enhancements.

Key improvements over V1:
1. Horizontal-flip training invariance (RandomHorizontalFlip p=0.5).
2. Label-smoothed loss (alpha=0.05) to prevent logit overconfidence.
3. Unfrozen ConvNeXt Stage 2 with discriminative learning rate (2e-5 vs 1e-4).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from verixa.models.hybrid import HybridRGBFFTClassifier, freeze_hybrid_backbone_stages
from verixa.training.augmentations import get_robust_training_transforms
from verixa.training.dataset import create_dataloaders
from verixa.training.loss import SmoothBCEWithLogitsLoss
from verixa.training.trainer import fit_convnext_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Hybrid V2 RGB+FFT with Stage 2 unfreezing and label smoothing."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/merged_manifest.csv"),
        help="Path to development manifest CSV.",
    )
    parser.add_argument(
        "--aug-prob",
        type=float,
        default=0.8,
        help="Probability of applying transformation-aware augmentations.",
    )
    parser.add_argument(
        "--hflip-prob",
        type=float,
        default=0.5,
        help="Probability of random horizontal flip augmentation.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing factor alpha for BCE loss.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--stage2-lr",
        type=float,
        default=2e-5,
        help="Learning rate for ConvNeXt Stage 2.",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-4,
        help="Learning rate for Stage 3, Head, and FFT branch.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW.",
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
        default=Path("models/convnext_tiny_hybrid_v2.pt"),
        help="Output path for best checkpoint.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/hybrid_v2_clean_eval.json"),
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
    print(" Verixa — Training Hybrid V2 Candidate Model")
    print("=================================================================")
    print(f" Manifest:          {args.manifest}")
    print(" Architecture:      HybridRGBFFTClassifier (ConvNeXt-Tiny + 2D FFT CNN)")
    print(f" Distortion Aug:    Transformation-Aware (p={args.aug_prob})")
    print(f" Horizontal Flip:   RandomHorizontalFlip (p={args.hflip_prob})")
    print(f" Label Smoothing:   alpha={args.label_smoothing}")
    print(f" ConvNeXt Stage 2:  Trainable (lr={args.stage2_lr})")
    print(f" Stage 3 + Head:    Trainable (lr={args.head_lr})")
    print(f" Max Epochs:        {args.epochs}")
    print(f" Patience:          {args.patience} epochs")
    print(f" Batch size:        {args.batch_size}")
    print(f" Random seed:       {args.seed}")
    print(f" Checkpoint out:    {args.checkpoint_out}")
    print(f" Report out:        {args.report_out}")
    print("=================================================================")

    # 1. Dataloaders with transformation-aware augmentations AND horizontal flip
    train_transform = get_robust_training_transforms(
        p=args.aug_prob,
        seed=args.seed,
        horizontal_flip_p=args.hflip_prob,
    )
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

    # 2. Hybrid Model & Unfreeze Stage 2 (freeze_up_to_stage=1 freezes Stem + Stage 0 + Stage 1)
    model = HybridRGBFFTClassifier(pretrained=True, dropout=0.3, use_grayscale_fft=True)
    freeze_hybrid_backbone_stages(model, freeze_up_to_stage=1)

    # Parameter groups with discriminative learning rates
    stage2_params = [
        p
        for i, child in enumerate(model.spatial_backbone.children())
        if i in (4, 5)
        for p in child.parameters()
        if p.requires_grad
    ]
    stage3_params = [
        p
        for i, child in enumerate(model.spatial_backbone.children())
        if i in (6, 7)
        for p in child.parameters()
        if p.requires_grad
    ]
    head_fft_params = [
        p for p in model.spectral_branch.parameters() if p.requires_grad
    ] + [p for p in model.head.parameters() if p.requires_grad]

    param_groups = [
        {"params": stage2_params, "lr": args.stage2_lr, "weight_decay": args.weight_decay},
        {"params": stage3_params, "lr": args.head_lr, "weight_decay": args.weight_decay},
        {"params": head_fft_params, "lr": args.head_lr, "weight_decay": args.weight_decay},
    ]

    optimizer = AdamW(param_groups)
    criterion = SmoothBCEWithLogitsLoss(alpha=args.label_smoothing)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel Parameter Breakdown:")
    print(f"  - Total Parameters:       {total_params:,}")
    print(f"  - Total Trainable:        {trainable_params:,}")
    s2_cnt = sum(p.numel() for p in stage2_params)
    s3_cnt = sum(p.numel() for p in stage3_params)
    hf_cnt = sum(p.numel() for p in head_fft_params)
    print(f"  - Stage 2 Params:         {s2_cnt:,} (lr={args.stage2_lr})")
    print(f"  - Stage 3 Params:         {s3_cnt:,} (lr={args.head_lr})")
    print(f"  - Head + FFT Params:      {hf_cnt:,} (lr={args.head_lr})")

    # 3. Assemble full run configuration embedded into checkpoint
    run_config = {
        "architecture": "hybrid_rgb_fft",
        "version": "v2",
        "spatial_backbone": "convnext_tiny",
        "spectral_branch": "fft_cnn_512d",
        "fusion": "concatenation_1280d",
        "pretrained_spatial": True,
        "dropout": 0.3,
        "freeze_stages": 1,
        "unfrozen_stage_2": True,
        "stage2_lr": args.stage2_lr,
        "head_lr": args.head_lr,
        "label_smoothing": args.label_smoothing,
        "hflip_prob": args.hflip_prob,
        "robust_augment": True,
        "aug_prob": args.aug_prob,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "manifest": str(args.manifest),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }

    # 4. Train & Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.perf_counter()

    results = fit_convnext_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.head_lr,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_path=args.checkpoint_out,
        config=run_config,
        patience=args.patience,
        model_name="HybridRGBFFT-V2",
        optimizer=optimizer,
        criterion=criterion,
    )

    t_elapsed = time.perf_counter() - t_start
    print(f"\nTraining finished in {t_elapsed:.1f}s ({t_elapsed/60:.2f} min).")

    # 5. Save report JSON
    report_payload = {
        "config": run_config,
        "training_time_seconds": round(t_elapsed, 2),
        "best_epoch": results.get("best_epoch"),
        "best_metrics": results.get("best_val_metrics", results.get("best_metrics")),
        "peak_vram_mb": results.get("peak_vram", results.get("peak_vram_mb")),
        "history": results.get("history"),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved training report to: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
