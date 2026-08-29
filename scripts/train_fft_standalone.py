"""CLI script to train the standalone FFT-only model on the frozen 30K dataset."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from verixa.models.fft import FFTClassifier
from verixa.training.dataset import create_dataloaders
from verixa.training.trainer import fit_convnext_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train standalone FFT-only model on 2D Fourier magnitude spectra."
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
        default=5e-4,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer.",
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
        default=Path("models/fft_standalone.pt"),
        help="Output path for best checkpoint.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/fft_standalone_clean_eval.json"),
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
    print(" Verixa — Phase 4: Standalone FFT-Only Spectral Model Training")
    print("=================================================================")
    print(f" Manifest:       {args.manifest}")
    print(" Architecture:   FFTClassifier (2D Log-Magnitude Spectrum CNN)")
    print(f" Max Epochs:     {args.epochs}")
    print(f" Patience:       {args.patience} epochs")
    print(f" Batch size:     {args.batch_size}")
    print(f" Learning rate:  {args.lr}")
    print(f" Random seed:    {args.seed}")
    print(f" Checkpoint out: {args.checkpoint_out}")
    print(f" Report out:     {args.report_out}")
    print("=================================================================")

    # 1. Dataloaders (Clean input images; FFT is extracted on GPU inside the model)
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

    # 2. Standalone Spectral Model
    model = FFTClassifier(use_grayscale=True, dropout=0.3)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. Assemble run configuration
    run_config = {
        "architecture": "fft_standalone",
        "num_params": trainable_params,
        "use_grayscale": True,
        "spectrum": "2D_fftshift_log1p_standardized",
        "max_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "manifest": str(args.manifest),
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
        model_name="FFTClassifier",
    )

    # 5. Save report
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nTraining complete. Saved clean evaluation report to: {args.report_out}")
    print(f"Saved best model checkpoint to: {args.checkpoint_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
