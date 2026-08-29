"""Training loop, validation, and VRAM monitoring for ConvNeXt-Tiny baseline."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from verixa.evaluation.metrics import calculate_binary_metrics


def get_peak_vram_mb() -> dict[str, float]:
    """Return peak allocated and reserved VRAM on the primary CUDA device in MB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
    }


def _render_live_progress(
    epoch: int,
    total_epochs: int,
    stage: str,
    step: int,
    total_steps: int,
    loss: float,
    start_time: float,
    bar_width: int = 16,
) -> None:
    """Render an in-place ASCII progress bar on a single terminal line."""
    elapsed = max(1e-5, time.time() - start_time)
    it_per_sec = step / elapsed
    pct = step / max(1, total_steps)
    filled = int(bar_width * pct)
    empty = bar_width - filled
    bar = "#" * filled + "-" * empty
    line = (
        f"\rEpoch {epoch}/{total_epochs} | {stage} [{bar}] {pct*100:3.0f}% | "
        f"{step}/{total_steps} | loss={loss:.4f} | {it_per_sec:.1f} it/s"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def _clear_live_line() -> None:
    """Clear the current terminal line."""
    sys.stdout.write("\r" + " " * 95 + "\r")
    sys.stdout.flush()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 1,
) -> float:
    """Run one training epoch with mixed precision and in-place ASCII progress bar."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_steps = len(loader)
    t_start = time.time()

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        use_amp = device.type == "cuda"
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1

        _render_live_progress(
            epoch=epoch,
            total_epochs=total_epochs,
            stage="Train",
            step=step,
            total_steps=total_steps,
            loss=loss_val,
            start_time=t_start,
        )

    _clear_live_line()
    avg_loss = round(total_loss / max(1, num_batches), 4)
    print(f"Epoch {epoch}/{total_epochs} | Train complete | loss={avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 1,
) -> tuple[float, dict[str, Any]]:
    """Evaluate model on a dataset and return validation loss and complete metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_steps = len(loader)
    all_labels: list[float] = []
    all_probs: list[float] = []
    t_start = time.time()

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        use_amp = device.type == "cuda"
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_labels.extend(labels.squeeze(1).cpu().numpy().tolist())
        all_probs.extend(probs.tolist())

        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1

        _render_live_progress(
            epoch=epoch,
            total_epochs=total_epochs,
            stage="Val  ",
            step=step,
            total_steps=total_steps,
            loss=loss_val,
            start_time=t_start,
        )

    _clear_live_line()
    avg_loss = round(total_loss / max(1, num_batches), 4)
    metrics = calculate_binary_metrics(
        y_true=np.array(all_labels),
        y_prob=np.array(all_probs),
        threshold=0.5,
    )
    metrics["val_loss"] = avg_loss

    acc_pct = metrics["accuracy"] * 100
    auroc_pct = metrics["auroc"] * 100
    fpr_pct = metrics["fpr"] * 100
    print(
        f"Epoch {epoch}/{total_epochs} | Val complete   | loss={avg_loss:.4f} | "
        f"Acc={acc_pct:.2f}% | AUROC={auroc_pct:.2f}% | FPR={fpr_pct:.2f}%"
    )
    return avg_loss, metrics


def fit_convnext_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    device: torch.device | None = None,
    checkpoint_path: Path | None = None,
    config: dict[str, Any] | None = None,
    patience: int | None = 4,
    model_name: str = "ConvNeXt-Tiny",
) -> dict[str, Any]:
    """Execute complete binary classifier training and validation loop.

    Returns:
        Summary dict of training history, best metrics, and peak VRAM.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_auroc = 0.0
    epochs_without_improvement = 0
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    print(f"Starting {model_name} training on device: {device}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )
        scheduler.step()

        val_loss, val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )
        duration = round(time.time() - t0, 2)
        vram = get_peak_vram_mb()

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_auroc": val_metrics["auroc"],
            "val_fpr": val_metrics["fpr"],
            "duration_s": duration,
            "vram": vram,
        }
        history.append(epoch_record)

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            epochs_without_improvement = 0
            best_metrics = dict(val_metrics)
            best_metrics["epoch"] = epoch
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": best_metrics,
                        "history": history,
                        "vram": vram,
                        "config": config or {},
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                print(
                    f"\nEarly stopping triggered: no AUROC improvement for {patience} "
                    f"consecutive epochs (best epoch: {best_metrics.get('epoch', 1)} "
                    f"with AUROC {best_val_auroc:.4f})."
                )
                break

    peak_vram = get_peak_vram_mb()

    return {
        "epochs": epochs,
        "best_epoch": best_metrics.get("epoch", epochs),
        "best_val_metrics": best_metrics,
        "final_epoch_metrics": history[-1] if history else {},
        "history": history,
        "peak_vram": peak_vram,
        "device": str(device),
    }
