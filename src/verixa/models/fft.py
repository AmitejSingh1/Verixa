"""Fast Fourier Transform (FFT) 2D magnitude spectrum extractor and standalone classifier."""
from __future__ import annotations

import torch
import torch.nn as nn


class FFTSpectrumExtractor(nn.Module):
    """Differentiable 2D FFT Log-Magnitude Spectrum Extractor.

    Computes centered 2D discrete Fourier transform on input images,
    compresses dynamic range via log1p, and standardizes values to zero mean, unit variance.
    """

    def __init__(self, use_grayscale: bool = True) -> None:
        super().__init__()
        self.use_grayscale = use_grayscale
        if use_grayscale:
            self.register_buffer(
                "luma_weights",
                torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1),
            )
        else:
            self.luma_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2D centered log-magnitude spectrum.

        Args:
            x: Input image tensor of shape (B, 3, H, W) or (B, 1, H, W).

        Returns:
            Normalized spectrum tensor of shape (B, 1, H, W) or (B, 3, H, W).
        """
        if self.use_grayscale and x.shape[1] == 3:
            x = (x * self.luma_weights).sum(dim=1, keepdim=True)

        fft = torch.fft.fft2(x, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.abs(fft_shifted)
        log_mag = torch.log1p(mag)

        mean = log_mag.mean(dim=(-2, -1), keepdim=True)
        std = log_mag.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (log_mag - mean) / std


class FFTClassifier(nn.Module):
    """Standalone Spectral CNN operating exclusively on 2D FFT Magnitude Spectrum."""

    def __init__(
        self,
        use_grayscale: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        in_ch = 1 if use_grayscale else 3
        self.use_grayscale = use_grayscale
        self.extractor = FFTSpectrumExtractor(use_grayscale=use_grayscale)

        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2),
            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, 1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional spectral feature vector for fusion in downstream hybrid model."""
        spectrum = self.extractor(x)
        feat = self.features(spectrum)
        return torch.flatten(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting binary logit from 2D FFT spectrum."""
        spectrum = self.extractor(x)
        feat = self.features(spectrum)
        return self.head(feat)

