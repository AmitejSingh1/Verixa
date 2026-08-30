"""Unit tests for production inference functions and CLI interfaces."""
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from scripts.predict import get_inference_transform, predict_single_image


class DummyBinaryClassifier(nn.Module):
    """Mock binary classifier outputting fixed logits for testing."""

    def __init__(self, logit_value: float = 2.5):
        super().__init__()
        self.logit_value = logit_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.full((batch_size, 1), self.logit_value, dtype=torch.float32)


def test_get_inference_transform() -> None:
    """Verify inference transform produces (3, 224, 224) normalized tensor."""
    tf = get_inference_transform()
    img = Image.new("RGB", (300, 400), color="blue")
    tensor = tf(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_predict_single_image_synthetic(tmp_path: Path) -> None:
    """Verify predict_single_image flags high logits as AI-Generated."""
    img_path = tmp_path / "test_fake.jpg"
    Image.new("RGB", (224, 224), color="red").save(img_path)

    model = DummyBinaryClassifier(logit_value=3.0)  # sigmoid(3.0) ~ 0.9526
    tf = get_inference_transform()
    device = torch.device("cpu")

    res = predict_single_image(img_path, model, device, tf, threshold=0.5)

    assert res["prediction"] == 1
    assert res["class_name"] == "AI-Generated"
    assert res["probability_synthetic"] > 0.9
    assert res["confidence_pct"] > 90.0
    assert res["threshold"] == 0.5


def test_predict_single_image_authentic(tmp_path: Path) -> None:
    """Verify predict_single_image flags negative logits as Authentic."""
    img_path = tmp_path / "test_real.png"
    Image.new("RGB", (100, 100), color="green").save(img_path)

    model = DummyBinaryClassifier(logit_value=-4.0)  # sigmoid(-4.0) ~ 0.018
    tf = get_inference_transform()
    device = torch.device("cpu")

    res = predict_single_image(img_path, model, device, tf, threshold=0.5)

    assert res["prediction"] == 0
    assert res["class_name"] == "Authentic"
    assert res["probability_synthetic"] < 0.05
    assert res["confidence_pct"] > 95.0


def test_predict_single_image_missing_raises(tmp_path: Path) -> None:
    """Verify FileNotFoundError is raised for non-existent image path."""
    model = DummyBinaryClassifier()
    tf = get_inference_transform()
    device = torch.device("cpu")

    with pytest.raises(FileNotFoundError):
        predict_single_image(tmp_path / "non_existent.jpg", model, device, tf)


def test_predict_single_image_corrupted_raises(tmp_path: Path) -> None:
    """Verify ValueError is raised for corrupted image file."""
    corrupted = tmp_path / "bad.jpg"
    corrupted.write_bytes(b"not an image data")

    model = DummyBinaryClassifier()
    tf = get_inference_transform()
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="Failed to decode image"):
        predict_single_image(corrupted, model, device, tf)


def test_verixa_predictor_batch_and_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify VerixaPredictor batch inference and CSV saving."""
    from verixa.inference import VerixaPredictor

    # Mock load_model_from_checkpoint
    monkeypatch.setattr(
        "verixa.inference.load_model_from_checkpoint",
        lambda *args, **kwargs: DummyBinaryClassifier(logit_value=2.0),
    )

    # Create dummy checkpoint file
    dummy_ckpt = tmp_path / "dummy.pt"
    dummy_ckpt.write_bytes(b"dummy")

    predictor = VerixaPredictor(checkpoint_path=dummy_ckpt, device="cpu", threshold=0.5)

    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.jpg"
    Image.new("RGB", (50, 50), color="red").save(img1)
    Image.new("RGB", (50, 50), color="blue").save(img2)

    res_single = predictor.predict_image(img1)
    assert res_single["prediction"] == 1
    assert res_single["class_name"] == "AI-Generated"

    res_batch = predictor.predict_batch([img1, img2], batch_size=2)
    assert len(res_batch) == 2

    out_csv = tmp_path / "preds.csv"
    predictor.save_predictions_to_csv(res_batch, out_csv)
    assert out_csv.exists()
    content = out_csv.read_text(encoding="utf-8")
    assert "filepath,prediction,class_name" in content
    assert "AI-Generated" in content

