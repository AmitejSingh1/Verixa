"""CLI script for training Hybrid V3 RGB+FFT ablation model.

Controlled ablation over Hybrid V1:
1. Exact Hybrid V1 architecture (ConvNeXt-Tiny + 2D FFT CNN, 1280d concatenation).
2. ConvNeXt stages 0..2 strictly frozen (stage 3, Head, and FFT branch trainable).
3. Exact V1 hyperparameters (lr=1e-4, weight_decay=0.01, batch_size=32, patience=4, aug_prob=0.8).
4. Added label smoothing alpha=0.05.
5. Added RandomHorizontalFlip p=0.5.
6. Target checkpoint: models/convnext_tiny_hybrid_v3.pt.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from verixa.models.hybrid import HybridRGBFFTClassifier, freeze_hybrid_backbone_stages
from verixa.training.augmentations import get_robust_training_transforms
from verixa.training.dataset import create_dataloaders
from verixa.training.loss import SmoothBCEWithLogitsLoss
from verixa.training.trainer import fit_convnext_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Hybrid V3 RGB+FFT ablation model with frozen backbone and smoothing."
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
        "--freeze-stages",
        type=int,
        default=2,
        help="Freeze ConvNeXt spatial backbone up to stage index (0..2 frozen).",
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
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for trainable parameters.",
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
        default=Path("models/convnext_tiny_hybrid_v3.pt"),
        help="Output path for best checkpoint.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/hybrid_v3_clean_eval.json"),
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
    print(" Verixa — Training Hybrid V3 Clean Ablation Model")
    print("=================================================================")
    print(f" Manifest:          {args.manifest}")
    print(" Architecture:      HybridRGBFFTClassifier (ConvNeXt-Tiny + 2D FFT CNN)")
    print(f" Distortion Aug:    Transformation-Aware (p={args.aug_prob})")
    print(f" Horizontal Flip:   RandomHorizontalFlip (p={args.hflip_prob})")
    print(f" Label Smoothing:   alpha={args.label_smoothing}")
    print(f" Frozen Stages:     spatial_backbone[0..{args.freeze_stages}] (Identical to V1)")
    print(f" Learning Rate:     {args.lr} (Identical to V1)")
    print(f" Weight Decay:      {args.weight_decay} (Identical to V1)")
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

    # 2. Hybrid Model & Freezing (freeze_up_to_stage=2: stages 0..2 frozen, exactly as V1)
    model = HybridRGBFFTClassifier(pretrained=True, dropout=0.3, use_grayscale_fft=True)
    freeze_hybrid_backbone_stages(model, freeze_up_to_stage=args.freeze_stages)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel Parameter Breakdown:")
    print(f"  - Total Parameters:       {total_params:,}")
    print(f"  - Total Trainable:        {trainable_params:,} (Stage 3 + Head + FFT Branch)")
    print(f"  - Frozen Parameters:      {total_params - trainable_params:,} (Stages 0..2)")

    # 3. Label-smoothed loss criterion
    criterion = SmoothBCEWithLogitsLoss(alpha=args.label_smoothing)

    # 4. Assemble run configuration embedded into checkpoint
    run_config = {
        "architecture": "hybrid_rgb_fft",
        "version": "v3",
        "spatial_backbone": "convnext_tiny",
        "spectral_branch": "fft_cnn_512d",
        "fusion": "concatenation_1280d",
        "pretrained_spatial": True,
        "dropout": 0.3,
        "freeze_stages": args.freeze_stages,
        "unfrozen_stage_2": False,
        "learning_rate": args.lr,
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

    # 5. Train & Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.perf_counter()

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
        model_name="HybridRGBFFT-V3",
        criterion=criterion,
    )

    t_elapsed = time.perf_counter() - t_start
    print(f"\nTraining finished in {t_elapsed:.1f}s ({t_elapsed/60:.2f} min).")

    # 6. Save report JSON
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
