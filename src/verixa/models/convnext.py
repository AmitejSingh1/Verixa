"""ConvNeXt-Tiny binary classifier definition and stage-freezing utilities."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ConvNeXtBinaryClassifier(nn.Module):
    """Pretrained ConvNeXt-Tiny backbone with a custom binary classification head.

    Head architecture:
        Global Average Pooling (768-d) -> LayerNorm(768) -> Linear(768, 256) -> GELU
        -> Dropout(p=0.3) -> Linear(256, 1)

    Output:
        Raw scalar logit per image for numerically stable BCEWithLogitsLoss.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=weights)

        # Retain convolutional stages (stages 0 to 7 in torchvision: 4 downsamplers + 4 stages)
        self.features = base.features

        # Custom binary classification head
        in_features = 768
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, 3, 224, 224)

        Returns:
            Logits tensor of shape (B, 1)
        """
        feats = self.features(x)
        logits = self.head(feats)
        return logits


def freeze_backbone_stages(
    model: ConvNeXtBinaryClassifier,
    freeze_up_to_stage: int = 2,
) -> None:
    """Freeze early convolutional stages of ConvNeXt-Tiny.

    Torchvision ConvNeXt stages layout in `features`:
        0: downsampling stem
        1: stage 0 blocks
        2: downsampling layer 1
        3: stage 1 blocks
        4: downsampling layer 2
        5: stage 2 blocks
        6: downsampling layer 3
        7: stage 3 blocks

    Args:
        model: ConvNeXtBinaryClassifier instance.
        freeze_up_to_stage: Index up to which stages are frozen:
            0: freeze stem + stage 0 (features[0..1])
            1: freeze up to stage 1 (features[0..3])
            2: freeze up to stage 2 (features[0..5]) -> stage 3 + head remain trainable
    """
    feature_cutoffs = {0: 1, 1: 3, 2: 5}
    cutoff = feature_cutoffs.get(freeze_up_to_stage, 5)

    for i, child in enumerate(model.features.children()):
        if i <= cutoff:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True

