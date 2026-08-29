"""Hybrid RGB + FFT Dual-Branch Binary Classifier.

Combines a spatial ConvNeXt-Tiny branch (768-d) and a spectral 2D FFT CNN branch (512-d)
via direct feature concatenation (1,280-d) into a projection and classification head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from verixa.models.fft import FFTClassifier


class HybridRGBFFTClassifier(nn.Module):
    """Dual-branch spatial-spectral architecture with direct concatenation fusion.

    - Branch 1 (Spatial): Pretrained ConvNeXt-Tiny backbone -> 768-d spatial embedding.
    - Fusion Head: Concatenate (1,280-d) -> LayerNorm -> Linear(1280, 256) -> GELU
      -> Dropout -> Linear(256, 1).
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        use_grayscale_fft: bool = True,
    ) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=weights)

        # 1. Spatial Branch (ConvNeXt-Tiny features)
        self.spatial_backbone = base.features
        self.spatial_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )

        # 2. Spectral Branch (2D FFT Magnitude Spectrum CNN)
        self.spectral_branch = FFTClassifier(
            use_grayscale=use_grayscale_fft,
            dropout=dropout,
        )

        # 3. Simple Concatenation Fusion Head (768 + 512 = 1,280)
        fusion_in_features = 768 + 512
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_in_features, eps=1e-6),
            nn.Linear(fusion_in_features, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting spatial and spectral embeddings and classifying.

        Args:
            x: Input image tensor of shape (B, 3, 224, 224).

        Returns:
            Raw scalar logits of shape (B, 1).
        """
        # Spatial branch: (B, 3, 224, 224) -> (B, 768, 7, 7) -> (B, 768)
        spatial_maps = self.spatial_backbone(x)
        spatial_feat = self.spatial_pool(spatial_maps)

        # Spectral branch: (B, 3, 224, 224) -> (B, 512)
        spectral_feat = self.spectral_branch.extract_features(x)

        # Concatenate: (B, 1280)
        combined = torch.cat([spatial_feat, spectral_feat], dim=1)

        # Classification logits: (B, 1)
        logits = self.head(combined)
        return logits


def freeze_hybrid_backbone_stages(
    model: HybridRGBFFTClassifier,
    freeze_up_to_stage: int = 2,
) -> None:
    """Freeze early convolutional stages of the spatial ConvNeXt backbone in the hybrid model.

    Preserves stage 3 of ConvNeXt, the entire spectral FFT branch, and the fusion head
    as trainable parameters.
    """
    feature_cutoffs = {0: 1, 1: 3, 2: 5}
    cutoff = feature_cutoffs.get(freeze_up_to_stage, 5)

    # 1. Freeze spatial backbone up to specified cutoff
    for i, child in enumerate(model.spatial_backbone.children()):
        if i <= cutoff:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True

    # 2. Spectral FFT branch remains fully trainable
    for param in model.spectral_branch.parameters():
        param.requires_grad = True

    # 3. Fusion head remains fully trainable
    for param in model.head.parameters():
        param.requires_grad = True
