"""Model checkpoint loader supporting all Verixa model architectures."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from verixa.models.convnext import ConvNeXtBinaryClassifier
from verixa.models.fft import FFTClassifier
from verixa.models.hybrid import HybridRGBFFTClassifier


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model architecture and weights from checkpoint dictionary."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    arch = config.get("architecture", "convnext_tiny")

    if arch == "fft_standalone":
        use_grayscale = config.get("use_grayscale", True)
        model: nn.Module = FFTClassifier(use_grayscale=use_grayscale)
    elif arch == "hybrid_rgb_fft":
        model = HybridRGBFFTClassifier(pretrained=False, use_grayscale_fft=True)
    else:
        model = ConvNeXtBinaryClassifier(pretrained=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
