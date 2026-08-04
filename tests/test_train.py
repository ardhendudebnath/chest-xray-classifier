"""The --resume path in the training entry point.

The loop itself is not covered here -- it needs a dataset and real epochs.
start_model is the part with branches worth pinning: which weights a run begins
from, and the ways that can go wrong quietly.

Nothing here touches the no-resume branch, which would force pretrained=True
and pull 45 MB of ImageNet weights. Everything below resumes from the untrained
checkpoint the conftest fixture writes, so the suite stays offline.
"""

import argparse

import pytest
import torch

from src.model import build_model, save_checkpoint
from src.train import start_model


def make_args(**overrides):
    args = argparse.Namespace(
        backbone=None,
        resume=None,
        freeze_backbone=False,
        out="checkpoints/best.pt",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_resume_loads_the_saved_weights_not_fresh_ones(checkpoint):
    model, backbone = start_model(make_args(resume=str(checkpoint)), "cpu")
    assert backbone == "resnet18"

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, saved["state_dict"][name]), name


def test_resume_reports_the_checkpoints_backbone_without_being_told(checkpoint):
    """--backbone is left at its default; the checkpoint is what decides."""
    _, backbone = start_model(make_args(resume=str(checkpoint)), "cpu")
    assert backbone == "resnet18"


def test_resume_can_still_freeze_to_the_head(checkpoint):
    """Freezing lives in build_model, which the resume path does not call."""
    model, _ = start_model(
        make_args(resume=str(checkpoint), freeze_backbone=True), "cpu"
    )

    assert all(p.requires_grad for p in model.fc.parameters())
    assert not any(p.requires_grad for p in model.layer4.parameters())
    assert not any(p.requires_grad for p in model.conv1.parameters())


def test_resume_without_freezing_leaves_everything_trainable(checkpoint):
    model, _ = start_model(make_args(resume=str(checkpoint)), "cpu")
    assert all(p.requires_grad for p in model.parameters())


def test_resuming_into_the_same_path_is_refused(checkpoint):
    """The default --out makes this the easy mistake, not the unlikely one."""
    args = make_args(resume=str(checkpoint), out=str(checkpoint))
    with pytest.raises(SystemExit, match="overwrite the checkpoint"):
        start_model(args, "cpu")


def test_a_backbone_that_contradicts_the_checkpoint_is_refused(checkpoint):
    args = make_args(resume=str(checkpoint), backbone="resnet50")
    with pytest.raises(SystemExit, match="only fit the architecture"):
        start_model(args, "cpu")


def test_a_missing_resume_path_names_itself(tmp_path):
    args = make_args(resume=str(tmp_path / "nope.pt"))
    with pytest.raises(SystemExit, match="does not exist"):
        start_model(args, "cpu")


def test_a_checkpoint_from_other_classes_is_refused(tmp_path, monkeypatch):
    """A relabelled checkpoint would silently redefine every prediction."""
    path = tmp_path / "other.pt"
    model = build_model("resnet18", pretrained=False)
    save_checkpoint(path, model, "resnet18", epoch=1, metrics={"macro_f1": 0.0})

    stored = torch.load(path, map_location="cpu", weights_only=True)
    stored["classes"] = ["COVID19", "PNEUMONIA", "NORMAL"]
    torch.save(stored, path)

    with pytest.raises(ValueError, match="relabel every prediction"):
        start_model(make_args(resume=str(path)), "cpu")
