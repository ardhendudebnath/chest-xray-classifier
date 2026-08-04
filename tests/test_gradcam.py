"""Grad-CAM: heatmap shape and range, and that the hooks are cleaned up."""

import numpy as np
import pytest
import torch

from src.dataset import CLASSES, IMG_SIZE, build_transforms
from src.gradcam_utils import GradCAM, denormalize, explain_image, overlay_cam
from src.model import build_model, target_layer


@pytest.fixture
def model():
    return build_model("resnet18", pretrained=False).eval()


def test_cam_covers_the_input_and_is_normalised(model):
    tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    with GradCAM(model, target_layer(model, "resnet18")) as extractor:
        cam, class_idx, logits = extractor(tensor)

    assert cam.shape == (IMG_SIZE, IMG_SIZE)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert not np.isnan(cam).any()
    assert class_idx in range(len(CLASSES))
    assert logits.shape == (1, len(CLASSES))


def test_asking_for_a_specific_class_explains_that_class(model):
    """Used to ask why the model did *not* say pneumonia."""
    tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    with GradCAM(model, target_layer(model, "resnet18")) as extractor:
        _, class_idx, _ = extractor(tensor, class_idx=CLASSES.index("COVID19"))

    assert class_idx == CLASSES.index("COVID19")


def test_batches_are_refused(model):
    with GradCAM(model, target_layer(model, "resnet18")) as extractor:
        with pytest.raises(ValueError, match="one image at a time"):
            extractor(torch.randn(4, 3, IMG_SIZE, IMG_SIZE))


def test_hook_is_removed_on_exit(model):
    """A leaked forward hook keeps every activation tensor alive."""
    layer = target_layer(model, "resnet18")
    before = len(layer._forward_hooks)

    with GradCAM(model, layer):
        assert len(layer._forward_hooks) == before + 1

    assert len(layer._forward_hooks) == before


def test_denormalize_returns_a_viewable_image(xray_image):
    _, image = xray_image
    tensor = build_transforms(train=False)(image.convert("L")).unsqueeze(0)

    array = denormalize(tensor)
    assert array.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert array.dtype == np.uint8


def test_overlay_keeps_the_underlying_xray_visible():
    base = np.full((32, 32, 3), 200, dtype=np.uint8)
    cam = np.zeros((32, 32))

    blended = np.asarray(overlay_cam(base, cam, alpha=0.45))
    assert blended.shape == (32, 32, 3)
    # With alpha 0.45 the X-ray still supplies most of the brightness.
    assert blended.mean() > 100


def test_explain_image_returns_an_overlay_and_a_distribution(model, xray_image):
    _, image = xray_image

    overlay, label, probabilities = explain_image(
        model, "resnet18", image, torch.device("cpu")
    )

    assert overlay.size == (IMG_SIZE, IMG_SIZE)
    assert label in CLASSES
    assert probabilities.shape == (len(CLASSES),)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)


def test_a_dead_cam_does_not_produce_nans(model):
    """Dividing a flat-zero map by its peak would give NaNs at render time."""
    layer = target_layer(model, "resnet18")

    with GradCAM(model, layer) as extractor:
        extractor._activations = torch.zeros(1, 512, 7, 7)
        cam, _, _ = extractor(torch.randn(1, 3, IMG_SIZE, IMG_SIZE))

    assert not np.isnan(cam).any()
