"""Unit tests for Fast Fourier Transform (FFT) spectrum extractor and classifier."""
from __future__ import annotations

import torch

from verixa.models.fft import FFTClassifier, FFTSpectrumExtractor


def test_fft_spectrum_extractor_grayscale_shape() -> None:
    extractor = FFTSpectrumExtractor(use_grayscale=True)
    x = torch.randn(2, 3, 224, 224)
    spec = extractor(x)
    assert spec.shape == (2, 1, 224, 224)
    assert not torch.isnan(spec).any()
    assert not torch.isinf(spec).any()


def test_fft_spectrum_extractor_rgb_shape() -> None:
    extractor = FFTSpectrumExtractor(use_grayscale=False)
    x = torch.randn(2, 3, 224, 224)
    spec = extractor(x)
    assert spec.shape == (2, 3, 224, 224)


def test_fft_spectrum_extractor_backward() -> None:
    extractor = FFTSpectrumExtractor(use_grayscale=True)
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    spec = extractor(x)
    loss = spec.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_fft_classifier_forward_shape() -> None:
    model = FFTClassifier(use_grayscale=True)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (4, 1)


def test_fft_classifier_extract_features() -> None:
    model = FFTClassifier(use_grayscale=True)
    x = torch.randn(3, 3, 224, 224)
    features = model.extract_features(x)
    assert features.shape == (3, 512)


def test_fft_classifier_parameter_budget() -> None:
    model = FFTClassifier(use_grayscale=True)
    total_params = sum(p.numel() for p in model.parameters())
    # Lightweight spectral CNN should be < 5M parameters
    assert 1_000_000 < total_params < 3_000_000

