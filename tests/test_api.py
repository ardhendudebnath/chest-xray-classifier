"""The served endpoints, including what they do with no model and bad input."""

import importlib
import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.dataset import CLASSES


def load_app(monkeypatch, checkpoint_path, ood_path=None):
    """Re-import app.main so it picks up its paths from the environment.

    Both are read at import time into module constants, so setting the
    variables after the first import would have no effect.

    ood_path defaults to somewhere that does not exist, rather than to the
    module default: that default is `checkpoints/ood.pt`, which is a real file
    in a working checkout, and letting it load would make these tests depend on
    whether someone had run src.ood.
    """
    monkeypatch.setenv("CXR_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setenv(
        "CXR_OOD_STATS", str(ood_path or checkpoint_path.parent / "no-such-ood.pt")
    )
    import app.main

    return importlib.reload(app.main)


@pytest.fixture
def client(monkeypatch, checkpoint):
    """No OOD statistics. The classifier still serves; the check reports null."""
    module = load_app(monkeypatch, checkpoint)
    with TestClient(module.app) as test_client:
        yield test_client


def write_stats(path, checkpoint_path, thresholds):
    """Stats for this checkpoint with the cutoffs pinned to chosen values.

    Real fitted cutoffs would make these tests depend on what untrained weights
    happen to do to synthetic images. What is under test here is that the
    verdict reaches the caller, so the verdict is made deterministic: 0.0
    flags everything, a huge number flags nothing.
    """
    import numpy as np

    from src.dataset import CLASSES
    from src.model import fingerprint_state_dict, load_checkpoint
    from src.ood import save_stats

    model, _ = load_checkpoint(checkpoint_path, "cpu")
    dimension = model.fc.in_features

    return save_stats(
        path,
        {
            "means": torch.zeros(len(CLASSES), dimension, dtype=torch.float64),
            "precision": torch.from_numpy(np.eye(dimension)),
            "classes": list(CLASSES),
            "backbone": "resnet18",
            "feature_dim": dimension,
            "shrinkage": 0.5,
            "fingerprint": fingerprint_state_dict(model.state_dict()),
            "thresholds": list(thresholds),
            "percentile": 95.0,
        },
    )


@pytest.fixture
def client_in_distribution(monkeypatch, checkpoint, tmp_path):
    path = write_stats(tmp_path / "pass.pt", checkpoint, [1e12] * 3)
    module = load_app(monkeypatch, checkpoint, path)
    with TestClient(module.app) as test_client:
        yield test_client


@pytest.fixture
def client_rejects_everything(monkeypatch, checkpoint, tmp_path):
    path = write_stats(tmp_path / "reject.pt", checkpoint, [0.0] * 3)
    module = load_app(monkeypatch, checkpoint, path)
    with TestClient(module.app) as test_client:
        yield test_client


@pytest.fixture
def png_bytes(xray_image):
    path, _ = xray_image
    return path.read_bytes()


def synthetic_png():
    """One drawn stand-in X-ray as PNG bytes, for tests without the fixture."""
    import numpy as np

    from src.synth_data import _render

    buffer = io.BytesIO()
    _render("NORMAL", np.random.default_rng(3)).save(buffer, format="PNG")
    return buffer.getvalue()


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
    """Catches the model being visibly torn between its four classes. That is
    a much narrower thing than "this is not a chest X-ray", which is why it is
    no longer the only signal a caller gets."""
    body = client.post(
        "/predict", files={"file": ("xray.png", png_bytes, "image/png")}
    ).json()

    assert body["low_confidence"] == (body["confidence"] < 0.6)


def test_no_ood_stats_reports_null_rather_than_a_pass(client, png_bytes):
    """The distinction the whole field turns on. Null means the check did not
    run; False would mean it ran and the image passed. Collapsing the two is
    how an unchecked image comes to look like a checked one."""
    body = client.post(
        "/predict", files={"file": ("xray.png", png_bytes, "image/png")}
    ).json()

    assert body["out_of_distribution"] is None
    assert body["ood_score"] is None
    assert body["ood_threshold"] is None
    assert client.get("/health").json()["ood_stats_loaded"] is False


def test_predict_reports_an_image_inside_the_distribution(
    client_in_distribution, png_bytes
):
    body = client_in_distribution.post(
        "/predict", files={"file": ("xray.png", png_bytes, "image/png")}
    ).json()

    assert body["out_of_distribution"] is False
    assert body["ood_score"] >= 0
    assert body["ood_threshold"] == pytest.approx(1e12)
    assert client_in_distribution.get("/health").json()["ood_stats_loaded"] is True


def test_predict_reports_an_image_outside_it(client_rejects_everything, png_bytes):
    """The case the endpoint used to get wrong: a confident answer, and a
    caller told nothing about the image being unlike anything trained on."""
    body = client_rejects_everything.post(
        "/predict", files={"file": ("xray.png", png_bytes, "image/png")}
    ).json()

    assert body["out_of_distribution"] is True
    assert body["ood_score"] > body["ood_threshold"]
    # Still a full prediction. The flag qualifies the answer, it does not
    # replace it -- a caller that ignores the flag sees what it always saw.
    assert body["prediction"] in CLASSES


def test_stats_for_another_checkpoint_are_ignored_not_trusted(
    monkeypatch, checkpoint, tmp_path
):
    """A mismatch has to degrade to "not checked", never to "checked, fine"."""
    from src.model import build_model, save_checkpoint

    other_path = tmp_path / "other.pt"
    torch.manual_seed(123)
    save_checkpoint(
        other_path, build_model("resnet18", pretrained=False), "resnet18", 1, {}
    )
    stats_path = write_stats(tmp_path / "mismatched.pt", other_path, [0.0] * 3)

    module = load_app(monkeypatch, checkpoint, stats_path)
    with TestClient(module.app) as test_client:
        assert test_client.get("/health").json()["ood_stats_loaded"] is False

        body = test_client.post(
            "/predict", files={"file": ("x.png", synthetic_png(), "image/png")}
        ).json()

        assert body["out_of_distribution"] is None


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


# ------------------------------------------------- serving the page as well


def test_the_frontend_is_not_served_unless_asked(client):
    """Two servers is the better arrangement locally, so it stays the default.
    A mount appearing on its own would also mean the mount below had been
    registered, and with it the risk it swallows the API."""
    assert client.get("/").status_code == 404


def test_the_api_still_answers_with_the_page_mounted(monkeypatch, checkpoint):
    """The mount is at "/", which matches every path beneath it. Registered
    before the routes rather than after, it would swallow /health and /predict
    and the container would serve a working page attached to nothing."""
    monkeypatch.setenv("CXR_SERVE_FRONTEND", "1")
    module = load_app(monkeypatch, checkpoint)

    with TestClient(module.app) as test_client:
        health = test_client.get("/health")
        assert health.status_code == 200
        assert health.json()["classes"] == CLASSES

        page = test_client.get("/")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]

        # The page reads its class list from /health, so shipping them from one
        # process is only safe while that request still reaches the API.
        assert test_client.get("/app.js").status_code == 200


def test_a_missing_frontend_directory_does_not_stop_the_api(
    monkeypatch, checkpoint, tmp_path
):
    """The deployment copies frontend/ in. If that step is ever missed, the
    API has to keep serving -- an API with no page is recoverable, a container
    that will not start is not."""
    monkeypatch.setenv("CXR_SERVE_FRONTEND", "1")
    monkeypatch.chdir(tmp_path)
    module = load_app(monkeypatch, checkpoint)
    monkeypatch.setattr(module, "FRONTEND_DIR", tmp_path / "no-frontend-here", raising=False)

    with TestClient(module.app) as test_client:
        assert test_client.get("/health").status_code == 200
