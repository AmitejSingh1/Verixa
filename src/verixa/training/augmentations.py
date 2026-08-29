"""Transformation-aware augmentation engine and evaluation distortion suites for Verixa.

Covers all 6 competition-mandated transformation families:
1. JPEG Compression:    quality in [30, 90]
2. Gaussian Blur:       sigma in [0.5, 2.0]
3. Resize (Down/Up):    scale in [0.25, 0.50]
4. Gaussian Noise:      sigma in [0.02, 0.10]
5. Color Jitter:        brightness/contrast/saturation in [0.8, 1.2]
6. Center Crop:         fraction in [0.75, 0.85] resized to 224x224
"""
from __future__ import annotations

import io
import random
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

from verixa.training.dataset import DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD

TARGET_IMAGE_SIZE = (224, 224)


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Simulate JPEG compression artifacts at a given quality factor (1-100)."""
    quality = max(1, min(100, int(quality)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur with radius/sigma."""
    if sigma <= 0.0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_resize(image: Image.Image, scale: float) -> Image.Image:
    """Downsample by scale factor using bilinear interpolation, then upscale back."""
    orig_w, orig_h = image.size
    down_w = max(1, int(orig_w * scale))
    down_h = max(1, int(orig_h * scale))
    downscaled = image.resize((down_w, down_h), resample=Image.Resampling.BILINEAR)
    return downscaled.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)


def apply_gaussian_noise(
    image: Image.Image,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Add zero-mean Gaussian noise scaled by sigma (0.0 to 1.0 range)."""
    if sigma <= 0.0:
        return image
    arr = np.array(image, dtype=np.float32) / 255.0
    if rng is not None:
        noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    else:
        noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).astype(np.uint8))


def apply_color_jitter(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> Image.Image:
    """Apply brightness, contrast, and saturation jitter using PIL ImageEnhance."""
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    if saturation != 1.0:
        image = ImageEnhance.Color(image).enhance(saturation)
    return image


def apply_center_crop(image: Image.Image, crop_fraction: float = 0.8) -> Image.Image:
    """Crop center percentage of image dimensions and resize back to original size."""
    orig_w, orig_h = image.size
    crop_w = max(1, int(orig_w * crop_fraction))
    crop_h = max(1, int(orig_h * crop_fraction))
    left = (orig_w - crop_w) // 2
    top = (orig_h - crop_h) // 2
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((orig_w, orig_h), resample=Image.Resampling.BICUBIC)


class RobustAugmentation:
    """Randomized transformation pipeline for transformation-aware training.

    With probability `p`, applies 1 to 2 randomly sampled transformations from the 6
    mandated families with uniformly sampled continuous parameters.
    """

    def __init__(
        self,
        p: float = 0.8,
        seed: int | None = None,
    ) -> None:
        self.p = p
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def _sample_transformation(self, transform_type: str, image: Image.Image) -> Image.Image:
        if transform_type == "jpeg":
            q = self.rng.randint(30, 90)
            return apply_jpeg_compression(image, quality=q)
        elif transform_type == "blur":
            sigma = self.rng.uniform(0.5, 2.0)
            return apply_gaussian_blur(image, sigma=sigma)
        elif transform_type == "resize":
            scale = self.rng.uniform(0.25, 0.50)
            return apply_resize(image, scale=scale)
        elif transform_type == "noise":
            sigma = self.rng.uniform(0.02, 0.10)
            return apply_gaussian_noise(image, sigma=sigma, rng=self.np_rng)
        elif transform_type == "jitter":
            b = self.rng.uniform(0.80, 1.20)
            c = self.rng.uniform(0.80, 1.20)
            s = self.rng.uniform(0.80, 1.20)
            return apply_color_jitter(image, brightness=b, contrast=c, saturation=s)
        elif transform_type == "crop":
            return apply_center_crop(image, crop_fraction=0.80)
        return image

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.rng.random() > self.p:
            return image

        candidate_types = ["jpeg", "blur", "resize", "noise", "jitter", "crop"]
        num_transforms = self.rng.choice([1, 2])
        chosen = self.rng.sample(candidate_types, k=num_transforms)

        augmented = image
        for t_type in chosen:
            augmented = self._sample_transformation(t_type, augmented)

        if augmented.size != TARGET_IMAGE_SIZE:
            augmented = augmented.resize(TARGET_IMAGE_SIZE, resample=Image.Resampling.BILINEAR)

        return augmented


def get_robust_training_transforms(p: float = 0.8, seed: int | None = 1337) -> transforms.Compose:
    """Return a torchvision Compose pipeline combining RobustAugmentation with ImageNet norm."""
    return transforms.Compose(
        [
            RobustAugmentation(p=p, seed=seed),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_IMAGENET_MEAN, std=DEFAULT_IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Benchmark Distortion Evaluation Suites (Exact 15 Distorted Conditions + Clean)
# ---------------------------------------------------------------------------
EVAL_DISTORTION_NAMES: list[str] = [
    "jpeg_q90",
    "jpeg_q70",
    "jpeg_q50",
    "jpeg_q30",
    "blur_sigma0.5",
    "blur_sigma1.0",
    "blur_sigma2.0",
    "resize_scale0.50",
    "resize_scale0.25",
    "noise_sigma0.02",
    "noise_sigma0.05",
    "noise_sigma0.10",
    "jitter_pm10",
    "jitter_pm20",
    "crop_fraction0.80",
    "composite_severe",
]


class DiscreteDistortionTransform:
    """Pickle-safe callable transform for discrete benchmark distortion conditions.

    Compatible with Windows multiprocessing DataLoader workers (spawn start method).
    """

    def __init__(self, distortion_name: str) -> None:
        if distortion_name not in EVAL_DISTORTION_NAMES:
            raise ValueError(
                f"Unknown distortion '{distortion_name}'. Expected one of: {EVAL_DISTORTION_NAMES}"
            )
        self.distortion_name = distortion_name

    def __call__(self, img: Image.Image) -> Image.Image:
        name = self.distortion_name
        if name == "jpeg_q90":
            return apply_jpeg_compression(img, quality=90)
        elif name == "jpeg_q70":
            return apply_jpeg_compression(img, quality=70)
        elif name == "jpeg_q50":
            return apply_jpeg_compression(img, quality=50)
        elif name == "jpeg_q30":
            return apply_jpeg_compression(img, quality=30)
        elif name == "blur_sigma0.5":
            return apply_gaussian_blur(img, sigma=0.5)
        elif name == "blur_sigma1.0":
            return apply_gaussian_blur(img, sigma=1.0)
        elif name == "blur_sigma2.0":
            return apply_gaussian_blur(img, sigma=2.0)
        elif name == "resize_scale0.50":
            return apply_resize(img, scale=0.50)
        elif name == "resize_scale0.25":
            return apply_resize(img, scale=0.25)
        elif name == "noise_sigma0.02":
            return apply_gaussian_noise(img, sigma=0.02)
        elif name == "noise_sigma0.05":
            return apply_gaussian_noise(img, sigma=0.05)
        elif name == "noise_sigma0.10":
            return apply_gaussian_noise(img, sigma=0.10)
        elif name == "jitter_pm10":
            return apply_color_jitter(img, brightness=1.1, contrast=0.9, saturation=1.1)
        elif name == "jitter_pm20":
            return apply_color_jitter(img, brightness=1.2, contrast=0.8, saturation=1.2)
        elif name == "crop_fraction0.80":
            return apply_center_crop(img, crop_fraction=0.80)
        elif name == "composite_severe":
            img = apply_jpeg_compression(img, quality=50)
            img = apply_gaussian_blur(img, sigma=1.0)
            return apply_resize(img, scale=0.50)
        raise ValueError(f"Unknown distortion: {name}")


EVAL_DISTORTION_SUITES: dict[str, Callable[[Image.Image], Image.Image]] = {
    name: DiscreteDistortionTransform(name) for name in EVAL_DISTORTION_NAMES
}


def get_distortion_eval_transform(distortion_name: str) -> transforms.Compose:
    """Return a transform applying a specific benchmark distortion followed by ImageNet norm."""
    if distortion_name == "clean":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=DEFAULT_IMAGENET_MEAN, std=DEFAULT_IMAGENET_STD),
            ]
        )

    distortion_fn = DiscreteDistortionTransform(distortion_name)
    return transforms.Compose(
        [
            distortion_fn,
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_IMAGENET_MEAN, std=DEFAULT_IMAGENET_STD),
        ]
    )

