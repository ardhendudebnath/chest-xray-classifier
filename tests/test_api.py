"""The served endpoints, including what they do with no model and bad input."""

import importlib
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.dataset import CLASSES


def load_app(monkeypatch, checkpoint_path):
    """Re-import app.main so it picks up CXR_CHECKPOINT from the environment.

    The path is read at import time into a module constant, so setting the
    variable after the first import would have no effect.
    """
    monkeypatch.setenv("CXR_CHECKPOINT", str(checkpoint_path))
    import app.main

    return importlib.reload(app.main)


@pytest.fixture
def client(monkeypatch, checkpoint):
    module = load_app(monkeypatch, checkpoint)
    with TestClient(module.app) as test_client:
        yield test_client


@pytest.fixture
def png_bytes(xray_image):
    path, _ = xray_image
    return path.read_bytes()


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["backbone"] == "resnet18"
    assert body["classes"] == CLASSES


def test_health_is_honest_when_no_checkpoint_exists(monkeypatch, tmp_path):
    """The service must still start, so /health can say what is wrong."""
    module = load_app(monkeypatch, tmp_path / "missing.pt")

    with TestClient(module.app) as test_client:
        body = test_client.get("/health").json()
        assert body["status"] == "no_model"
        assert body["model_loaded"] is False

        response = test_client.post(
            "/predict", files={"file": ("x.png", b"not an image", "image/png")}
        )
        assert response.status_code == 503


def test_predict_returns_a_ranked_distribution(client, png_bytes):
    response = client.post("/predict", files={"file": ("xray.png", png_bytes, "image/png")})
    assert response.status_code == 200
    body = response.json()

    assert body["prediction"] in CLASSES
    assert set(body["probabilities"]) == set(CLASSES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["confidence"] == pytest.approx(
        max(body["probabilities"].values()), abs=1e-4
    )
    assert body["disclaimer"]


def test_predict_flags_low_confidence(client, png_bytes):
    """An untrained model is near-uniform, which is exactly the case the flag
    exists for: a 3-class softmax names a class even for a photo of a cat."""
    body = client.post(
        "/predict", files={"file": ("xray.png", png_bytes, "image/png")}
    ).json()

    assert body["low_confidence"] == (body["confidence"] < 0.6)


def test_a_jpeg_is_accepted_too(client, xray_image):
    _, image = xray_image
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")

    response = client.post(
        "/predict", files={"file": ("xray.jpg", buffer.getvalue(), "image/jpeg")}
    )
    assert response.status_code == 200


def test_a_non_image_upload_gets_a_clear_400(client):
    response = client.post(
        "/predict", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_an_empty_upload_is_rejected(client):
    response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400


def test_an_oversized_upload_is_rejected_before_decoding(client, monkeypatch):
    import app.main

    monkeypatch.setattr(app.main, "MAX_UPLOAD_BYTES", 1024)
    response = client.post(
        "/predict", files={"file": ("big.png", b"\x00" * 2048, "image/png")}
    )
    assert response.status_code == 413


def test_explain_returns_a_png_with_the_prediction_in_the_headers(client, png_bytes):
    response = client.post("/explain", files={"file": ("xray.png", png_bytes, "image/png")})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-prediction"] in CLASSES
    assert 0.0 <= float(response.headers["x-confidence"]) <= 1.0
    assert response.headers["x-disclaimer"]
    # With no class_name, the explained class is the predicted one.
    assert response.headers["x-explained-class"] == response.headers["x-prediction"]

    overlay = Image.open(io.BytesIO(response.content))
    assert overlay.size == (224, 224)


def test_explain_separates_the_explained_class_from_the_prediction(client, png_bytes):
    """Asking "why not pneumonia?" must not overwrite what the model actually
    said, and each confidence has to belong to the class printed beside it."""
    response = client.post(
        "/explain",
        params={"class_name": "PNEUMONIA"},
        files={"file": ("xray.png", png_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.headers["x-explained-class"] == "PNEUMONIA"
    assert response.headers["x-prediction"] in CLASSES

    predicted = client.post(
        "/predict", files={"file": ("xray.png", png_bytes, "image/png")}
    ).json()
    assert response.headers["x-prediction"] == predicted["prediction"]
    assert float(response.headers["x-confidence"]) == pytest.approx(
        predicted["confidence"], abs=1e-3
    )
    assert float(response.headers["x-explained-confidence"]) == pytest.approx(
        predicted["probabilities"]["PNEUMONIA"], abs=1e-3
    )


def test_explain_rejects_an_unknown_class(client, png_bytes):
    response = client.post(
        "/explain",
        params={"class_name": "TUBERCULOSIS"},
        files={"file": ("xray.png", png_bytes, "image/png")},
    )
    assert response.status_code == 400
