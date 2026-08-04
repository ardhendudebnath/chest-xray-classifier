"""Architecture, freezing, and the checkpoint contract the API depends on."""

import pytest
import torch

from src.dataset import CLASSES, IMG_SIZE, NUM_CLASSES
from src.model import (
    build_model,
    load_checkpoint,
    pick_device,
    save_checkpoint,
    target_layer,
)


def test_head_emits_one_logit_per_class():
    model = build_model("resnet18", pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, IMG_SIZE, IMG_SIZE))
    assert logits.shape == (2, NUM_CLASSES)


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_model("resnet9000", pretrained=False)


def test_freezing_leaves_only_the_head_trainable():
    model = build_model("resnet18", pretrained=False, freeze_backbone=True)

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}


def test_nothing_is_frozen_by_default():
    model = build_model("resnet18", pretrained=False)
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("backbone", ["resnet18", "densenet121"])
def test_gradcam_target_layer_exists_and_runs(backbone):
    model = build_model(backbone, pretrained=False).eval()
    layer = target_layer(model, backbone)

    captured = []
    handle = layer.register_forward_hook(lambda m, i, o: captured.append(o))
    with torch.no_grad():
        model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
    handle.remove()

    # A 224px input reduces to a 7x7 grid at the last block in both families.
    assert captured and captured[0].shape[-2:] == (7, 7)


def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    original = build_model("resnet18", pretrained=False).eval()
    batch = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        expected = original(batch)

    path = tmp_path / "model.pt"
    save_checkpoint(path, original, "resnet18", epoch=3, metrics={"macro_f1": 0.9})
    restored, metadata = load_checkpoint(path, device="cpu")

    with torch.no_grad():
        assert torch.allclose(restored(batch), expected, atol=1e-6)

    assert metadata["backbone"] == "resnet18"
    assert metadata["classes"] == CLASSES
    assert metadata["epoch"] == 3
    assert metadata["metrics"]["macro_f1"] == pytest.approx(0.9)


def test_loaded_model_is_in_eval_mode(checkpoint):
    """Left in train mode, BatchNorm would drift with every served request."""
    model, _ = load_checkpoint(checkpoint, device="cpu")
    assert not model.training


def test_checkpoint_with_different_classes_is_refused(tmp_path):
    """Silently loading this would relabel every prediction."""
    path = tmp_path / "stale.pt"
    model = build_model("resnet18", pretrained=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": "resnet18",
            "classes": ["NORMAL", "PNEUMONIA", "COVID19"],  # reordered
            "img_size": IMG_SIZE,
            "epoch": 1,
            "metrics": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="relabel"):
        load_checkpoint(path, device="cpu")


def test_metrics_survive_the_weights_only_load(tmp_path):
    """torch.load(weights_only=True) refuses numpy scalars, so they are cast."""
    import numpy as np

    path = tmp_path / "numpy_metrics.pt"
    model = build_model("resnet18", pretrained=False)
    save_checkpoint(
        path, model, "resnet18", epoch=1, metrics={"macro_f1": np.float32(0.75)}
    )

    _, metadata = load_checkpoint(path, device="cpu")
    assert isinstance(metadata["metrics"]["macro_f1"], float)


def test_pick_device_honours_an_explicit_request():
    assert pick_device("cpu").type == "cpu"
    assert pick_device().type in {"cpu", "cuda"}
