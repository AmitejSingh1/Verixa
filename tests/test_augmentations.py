"""Unit tests for Verixa transformation-aware augmentations and benchmark suites."""
from __future__ import annotations

import pytest
import torch
from PIL import Image

from verixa.training.augmentations import (
    EVAL_DISTORTION_SUITES,
    RobustAugmentation,
    apply_center_crop,
    apply_color_jitter,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_resize,
    get_distortion_eval_transform,
    get_robust_training_transforms,
)


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a 224x224 RGB test image with gradient patterns."""
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    return img


def test_apply_jpeg_compression(sample_image: Image.Image) -> None:
    compressed = apply_jpeg_compression(sample_image, quality=30)
    assert compressed.size == (224, 224)
    assert compressed.mode == "RGB"


def test_apply_gaussian_blur(sample_image: Image.Image) -> None:
    blurred = apply_gaussian_blur(sample_image, sigma=2.0)
    assert blurred.size == (224, 224)
    assert blurred.mode == "RGB"


def test_apply_resize(sample_image: Image.Image) -> None:
    resized = apply_resize(sample_image, scale=0.25)
    assert resized.size == (224, 224)
    assert resized.mode == "RGB"


def test_apply_gaussian_noise(sample_image: Image.Image) -> None:
    noisy = apply_gaussian_noise(sample_image, sigma=0.10)
    assert noisy.size == (224, 224)
    assert noisy.mode == "RGB"


def test_apply_color_jitter(sample_image: Image.Image) -> None:
    jittered = apply_color_jitter(sample_image, brightness=1.2, contrast=0.8, saturation=1.1)
    assert jittered.size == (224, 224)
    assert jittered.mode == "RGB"


def test_apply_center_crop(sample_image: Image.Image) -> None:
    cropped = apply_center_crop(sample_image, crop_fraction=0.8)
    assert cropped.size == (224, 224)
    assert cropped.mode == "RGB"


def test_robust_augmentation_always_active(sample_image: Image.Image) -> None:
    aug = RobustAugmentation(p=1.0, seed=1337)
    transformed = aug(sample_image)
    assert transformed.size == (224, 224)
    assert transformed.mode == "RGB"


def test_robust_augmentation_inactive(sample_image: Image.Image) -> None:
    aug = RobustAugmentation(p=0.0, seed=1337)
    transformed = aug(sample_image)
    assert transformed == sample_image


def test_get_robust_training_transforms(sample_image: Image.Image) -> None:
    pipeline = get_robust_training_transforms(p=0.8, seed=1337)
    tensor = pipeline(sample_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


@pytest.mark.parametrize("suite_name", list(EVAL_DISTORTION_SUITES.keys()) + ["clean"])
def test_get_distortion_eval_transform_all_suites(
    sample_image: Image.Image,
    suite_name: str,
) -> None:
    t = get_distortion_eval_transform(suite_name)
    tensor = t(sample_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_get_distortion_eval_transform_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unknown distortion"):
        get_distortion_eval_transform("non_existent_distortion")
