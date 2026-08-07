"""The evaluation script, which is where every published number comes from.

Its failure mode is the reason this file exists. If `collect_predictions` ever
paired an image's prediction with a different image's label, the metrics would
not error, would not look odd, and would not be obviously wrong -- they would be
plausible numbers for a model that does not exist, written to a JSON file and
quoted in the README. Nothing downstream could catch it.

So the alignment is tested against a stub whose logits are a pure function of
the image it was handed, which makes a prediction traceable back to the image
that produced it. The rest -- report paths, the metrics file, the plot fallback
-- is plumbing that has already gone wrong once in this repo under a different
name, and is pinned here for the same reason.
"""

import json
import sys

import numpy as np
import pytest
import torch

from src.dataset import CLASSES, build_loader
from src.evaluate import (
    collect_predictions,
    main,
    print_confusion,
    save_confusion_plot,
)
from src.model import build_model


class TraceableModel(torch.nn.Module):
    """Logits determined entirely by the pixels, never by call order.

    An untrained resnet would work for the shape assertions but not for the
    alignment ones: to show that image i's prediction lands beside image i's
    label, the prediction has to be something this test can compute for a
    single image on its own and then look for in the batched output.
    """

    def forward(self, images):
        buckets = (images.flatten(1).mean(dim=1).abs() * 1000).long() % len(CLASSES)
        return torch.nn.functional.one_hot(buckets, len(CLASSES)).float() * 10.0


@pytest.fixture
def test_split(data_root):
    return data_root / "test"


def loader_for(split_dir, batch_size):
    return build_loader(split_dir, train=False, batch_size=batch_size, num_workers=0)


# --------------------------------------------------------------- alignment


def test_targets_come_back_in_dataset_order(test_split):
    """build_loader leaves shuffle off when train=False, so the labels have to
    arrive in exactly the order ImageFolder assigned them. Everything else in
    this file depends on that being true."""
    loader, dataset = loader_for(test_split, batch_size=4)
    targets, _, _ = collect_predictions(TraceableModel(), loader, torch.device("cpu"))

    assert targets.tolist() == [label for _, label in dataset.samples]


def test_a_prediction_stays_with_the_image_that_produced_it(test_split):
    """The silent failure this module exists to rule out.

    Each image is put through the model alone, and the answer has to appear at
    that image's index in the batched run. A pairing that slipped by even one
    would fail here while leaving every metric looking perfectly reasonable.
    """
    loader, dataset = loader_for(test_split, batch_size=4)
    model = TraceableModel()
    _, predictions, _ = collect_predictions(model, loader, torch.device("cpu"))

    one_at_a_time = []
    for index in range(len(dataset)):
        image, _ = dataset[index]
        with torch.no_grad():
            one_at_a_time.append(int(model(image.unsqueeze(0)).argmax(dim=1)))

    assert predictions.tolist() == one_at_a_time


def test_batching_does_not_reorder_anything(test_split):
    """A batch size that divides the split evenly and two that do not. The
    ragged final batch is where an off-by-one would show up."""
    results = []
    for batch_size in (1, 3, 4, 32):
        loader, _ = loader_for(test_split, batch_size=batch_size)
        targets, predictions, _ = collect_predictions(
            TraceableModel(), loader, torch.device("cpu")
        )
        results.append((targets.tolist(), predictions.tolist()))

    assert all(result == results[0] for result in results)


def test_predictions_and_probabilities_come_from_one_forward_pass(test_split):
    """predictions read the logits, probabilities read a softmax of them. They
    are built in separate statements, so nothing but this stops them drifting
    apart -- and a drift would put one image's confidence beside another's
    call in the served API too."""
    loader, _ = loader_for(test_split, batch_size=4)
    _, predictions, probabilities = collect_predictions(
        build_model("resnet18", pretrained=False).eval(), loader, torch.device("cpu")
    )

    assert probabilities.argmax(axis=1).tolist() == predictions.tolist()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_every_image_is_scored_exactly_once(test_split):
    loader, dataset = loader_for(test_split, batch_size=3)
    targets, predictions, probabilities = collect_predictions(
        TraceableModel(), loader, torch.device("cpu")
    )

    assert len(targets) == len(predictions) == len(probabilities) == len(dataset)
    assert probabilities.shape == (len(dataset), len(CLASSES))


# ------------------------------------------------------------------ output


def test_the_confusion_matrix_prints_rows_as_truth(capsys):
    """Rows are actual and columns are predicted. Transposing it would invert
    every recall claim in the README while still printing a tidy square."""
    matrix = np.arange(len(CLASSES) ** 2).reshape(len(CLASSES), len(CLASSES))
    print_confusion(matrix)

    lines = capsys.readouterr().out.strip().splitlines()
    assert "rows = actual, cols = predicted" in lines[0]
    assert lines[1].split() == CLASSES

    for offset, name in enumerate(CLASSES):
        row = lines[2 + offset].split()
        assert row[0] == name
        assert [int(value) for value in row[1:]] == matrix[offset].tolist()


def test_the_plot_creates_its_own_directory(tmp_path):
    pytest.importorskip("matplotlib")
    path = tmp_path / "nested" / "somewhere" / "confusion.png"
    save_confusion_plot(np.eye(len(CLASSES), dtype=int) * 5, path)

    assert path.is_file() and path.stat().st_size > 0


# -------------------------------------------------------------------- main


def evaluate_once(monkeypatch, data_root, checkpoint, report_dir, split="test"):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "src.evaluate",
            "--checkpoint", str(checkpoint),
            "--data-dir", str(data_root),
            "--split", split,
            "--batch-size", "4",
            "--report-dir", str(report_dir),
            "--device", "cpu",
        ],
    )
    main()
    return report_dir / f"metrics_{split}.json"


def test_the_metrics_file_has_the_shape_the_readme_quotes(
    tmp_path, data_root, checkpoint, monkeypatch
):
    """Every number in the README was read out of one of these files, so the
    keys are load-bearing rather than incidental."""
    path = evaluate_once(monkeypatch, data_root, checkpoint, tmp_path / "reports")
    summary = json.loads(path.read_text("utf-8"))

    assert summary["classes"] == CLASSES
    assert summary["split"] == "test"
    assert summary["images"] == len(CLASSES) * 2
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert 0.0 <= summary["macro_f1"] <= 1.0

    matrix = np.array(summary["confusion_matrix"])
    assert matrix.shape == (len(CLASSES), len(CLASSES))
    # Every image lands in exactly one cell, so the matrix has to total the split.
    assert matrix.sum() == summary["images"]


def test_two_report_dirs_do_not_overwrite_each_other(
    tmp_path, data_root, checkpoint, monkeypatch
):
    """--report-dir is what keeps a masked run's metrics off an unmasked run's.
    The same mistake under a fixed filename already cost this repo a training
    history, so the isolation is pinned rather than assumed."""
    first = evaluate_once(monkeypatch, data_root, checkpoint, tmp_path / "unmasked")
    second = evaluate_once(monkeypatch, data_root, checkpoint, tmp_path / "masked")

    assert first != second
    assert first.is_file() and second.is_file()


def test_a_broken_plot_does_not_lose_the_metrics(
    tmp_path, data_root, checkpoint, monkeypatch, capsys
):
    """Matplotlib fails for reasons unrelated to the model -- no backend, an
    unwritable font cache -- and the numbers are the actual output. Losing a
    finished evaluation over the picture is the trade this guards against."""
    def explode(matrix, path):
        raise RuntimeError("no display name and no $DISPLAY environment variable")

    monkeypatch.setattr("src.evaluate.save_confusion_plot", explode)
    path = evaluate_once(monkeypatch, data_root, checkpoint, tmp_path / "reports")

    assert path.is_file()
    assert json.loads(path.read_text("utf-8"))["classes"] == CLASSES
    assert "could not write" in capsys.readouterr().out
