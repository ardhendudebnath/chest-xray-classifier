"""The training entry point: where a run starts from, and where it writes.

start_model is the part with branches worth pinning: which weights a run begins
from, and the ways that can go wrong quietly.

The end of the loop is covered too, by running main() over the synthetic split.
The metrics that come out are meaningless -- untrained weights, drawn images --
but the history file is written there, and what it was writing over is the
point.

Nothing here touches the no-resume branch, which would force pretrained=True
and pull 45 MB of ImageNet weights. Everything below resumes from the untrained
checkpoint the conftest fixture writes, so the suite stays offline.
"""

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

from src.model import build_model, save_checkpoint
from src.train import history_path_for, main, start_model


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


@pytest.mark.parametrize(
    "out,expected",
    [
        ("checkpoints/best.pt", "checkpoints/best_history.json"),
        ("checkpoints/stage1.pt", "checkpoints/stage1_history.json"),
        ("checkpoints/masked_best.pt", "checkpoints/masked_best_history.json"),
        # No directory component, and a suffix that is not .pt.
        ("best.pt", "best_history.json"),
        ("checkpoints/best.pth", "checkpoints/best_history.json"),
    ],
)
def test_the_history_file_is_named_after_the_checkpoint(out, expected):
    """Two checkpoint names cannot collide on one history file."""
    assert history_path_for(out) == Path(expected)


def train_once(monkeypatch, data_root, checkpoint, out, epochs):
    """One real training run, driven through the CLI the way a user would.

    --resume off the untrained fixture checkpoint keeps the no-resume branch
    and its 45 MB download out of it, and --freeze-backbone makes an epoch a
    head-only backward over ~27 drawn images, which costs about a second.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "src.train",
            "--data-dir", str(data_root),
            "--resume", str(checkpoint),
            "--out", str(out),
            "--epochs", str(epochs),
            "--batch-size", "8",
            "--freeze-backbone",
            "--device", "cpu",
        ],
    )
    main()


def test_two_runs_into_one_directory_keep_both_histories(
    tmp_path, data_root, checkpoint, monkeypatch
):
    """The history file used to be a fixed name, so the second run took the
    first one's epochs with it -- at the end, with nothing printed to say so.

    Different epoch counts, so this reads the surviving contents rather than
    just two mtimes: whichever run wrote last, both curves have to be intact.
    """
    first = tmp_path / "checkpoints" / "stage1.pt"
    second = tmp_path / "checkpoints" / "masked_stage1.pt"

    train_once(monkeypatch, data_root, checkpoint, first, epochs=2)
    train_once(monkeypatch, data_root, checkpoint, second, epochs=1)

    first_history = history_path_for(first)
    second_history = history_path_for(second)
    assert first_history != second_history
    assert first_history.is_file() and second_history.is_file()

    def epochs_in(path):
        return [entry["epoch"] for entry in json.loads(path.read_text("utf-8"))]

    assert epochs_in(first_history) == [1, 2], "the second run overwrote the first"
    assert epochs_in(second_history) == [1]


def test_a_run_writes_no_history_under_the_old_shared_name(
    tmp_path, data_root, checkpoint, monkeypatch
):
    """Nothing in the repo reads history.json, so it is not kept as an alias.
    Leaving one behind would put the collision straight back."""
    out = tmp_path / "checkpoints" / "best.pt"
    train_once(monkeypatch, data_root, checkpoint, out, epochs=1)

    assert not (out.parent / "history.json").exists()
    assert history_path_for(out).is_file()


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
