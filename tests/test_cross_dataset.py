"""Scoring a second dataset with its shared images removed.

The partition is what these check. Whether the model is any good is not
testable here -- the fixtures use untrained weights -- but whether the right
images end up in the right population is, and that is the part that decides
whether a published cross-dataset number means anything.
"""

import io

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from src.cross_dataset import label_of, labelled, predict, score_subset
from src.dataset import CLASSES
from src.model import load_checkpoint

CPU = torch.device("cpu")


def distinct_image(rng, size=224):
    image = Image.new("L", (size, size), int(rng.integers(10, 60)))
    draw = ImageDraw.Draw(image)
    for _ in range(int(rng.integers(3, 7))):
        x, y = rng.integers(0, size, 2)
        rx, ry = rng.integers(20, 90, 2)
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=int(rng.integers(70, 250)))
    return image


def reencoded(image, size=180, quality=55):
    buffer = io.BytesIO()
    image.convert("L").resize((size, size), Image.BILINEAR).save(
        buffer, format="JPEG", quality=quality
    )
    return Image.open(io.BytesIO(buffer.getvalue()))


@pytest.fixture
def model_on_cpu(checkpoint):
    model, meta = load_checkpoint(checkpoint, CPU)
    return model, meta["backbone"]


# ------------------------------------------------------------------ labelling


def test_the_class_is_found_at_any_depth(tmp_path):
    assert label_of(tmp_path / "Data" / "train" / "NORMAL" / "a.png") == 1
    assert label_of(tmp_path / "COVID19" / "images" / "a.png") == 0
    assert label_of(tmp_path / "x" / "PNEUMONIA" / "y" / "a.png") == 2


def test_the_deepest_class_directory_wins(tmp_path):
    """A path with two class names in it is ambiguous; the one nearest the
    image is the one that labelled it."""
    assert label_of(tmp_path / "NORMAL" / "COVID19" / "a.png") == 0


def test_an_unlabelled_path_is_dropped_not_guessed(tmp_path):
    """Lung_Opacity has no class here. Silently folding it into one of the
    three would put a finding the model has no output for into the score."""
    assert label_of(tmp_path / "Lung_Opacity" / "images" / "a.png") is None

    paths = [
        tmp_path / "NORMAL" / "a.png",
        tmp_path / "Lung_Opacity" / "b.png",
        tmp_path / "COVID19" / "c.png",
    ]
    kept, labels, dropped = labelled(paths)

    assert len(kept) == 2
    assert labels.tolist() == [1, 0]
    assert len(dropped) == 1


def test_labels_line_up_with_the_paths_they_came_from(tmp_path):
    paths = [tmp_path / name / f"{index}.png"
             for index, name in enumerate(["PNEUMONIA", "COVID19", "NORMAL"])]
    kept, labels, _ = labelled(paths)

    assert [CLASSES[i] for i in labels] == [p.parent.name for p in kept]


# ----------------------------------------------------------------- the scoring


def test_predictions_and_flags_come_back_per_image(model_on_cpu, tmp_path):
    model, backbone = model_on_cpu
    rng = np.random.default_rng(0)
    paths = []
    for index in range(5):
        path = tmp_path / f"{index}.png"
        distinct_image(rng).save(path)
        paths.append(path)

    predictions, flags = predict(model, backbone, paths, CPU, stats=None, batch_size=2)

    assert predictions.shape == (5,)
    assert flags.shape == (5,)
    assert set(predictions.tolist()) <= set(range(len(CLASSES)))


def test_batching_does_not_change_the_predictions(model_on_cpu, tmp_path):
    model, backbone = model_on_cpu
    rng = np.random.default_rng(1)
    paths = []
    for index in range(7):
        path = tmp_path / f"{index}.png"
        distinct_image(rng).save(path)
        paths.append(path)

    whole, _ = predict(model, backbone, paths, CPU, batch_size=64)
    split, _ = predict(model, backbone, paths, CPU, batch_size=2)

    assert whole.tolist() == split.tolist()


def test_no_stats_reports_no_ood_section(model_on_cpu, tmp_path):
    """Absent statistics must not read as a detector that never fired."""
    model, backbone = model_on_cpu
    rng = np.random.default_rng(2)
    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.png"
        distinct_image(rng).save(path)
        paths.append(path)

    result = score_subset(
        model, backbone, paths, np.array([0, 1, 2, 1]), CPU, stats=None
    )

    assert "ood_refused" not in result
    assert result["images"] == 4
    assert 0.0 <= result["macro_f1"] <= 1.0


def test_an_empty_population_scores_as_none(model_on_cpu):
    model, backbone = model_on_cpu
    assert score_subset(model, backbone, [], np.array([]), CPU) is None


def test_the_confusion_matrix_covers_every_class(model_on_cpu, tmp_path):
    """Even when a split contains only one class, so the matrix stays 3x3 and
    the rows keep meaning the same thing between runs."""
    model, backbone = model_on_cpu
    path = tmp_path / "only.png"
    distinct_image(np.random.default_rng(3)).save(path)

    result = score_subset(model, backbone, [path], np.array([1]), CPU)

    assert np.array(result["confusion_matrix"]).shape == (len(CLASSES), len(CLASSES))
    assert set(result["per_class_f1"]) == set(CLASSES)


# ------------------------------------------------------- the partition itself


def test_reencoded_training_images_land_in_the_overlapping_population(tmp_path):
    """End to end: the images that came from the training split have to be the
    ones excluded, or the clean score is measured on training data."""
    from src.dataset_overlap import DUPLICATE, compare

    rng = np.random.default_rng(4)
    originals = [distinct_image(rng) for _ in range(6)]
    fresh = [distinct_image(np.random.default_rng(99)) for _ in range(4)]

    train = tmp_path / "train" / "NORMAL"
    train.mkdir(parents=True)
    for index, image in enumerate(originals):
        image.save(train / f"{index}.png")

    second = tmp_path / "second" / "NORMAL"
    second.mkdir(parents=True)
    for index, image in enumerate(originals):
        reencoded(image).convert("L").save(second / f"copy{index}.jpg")
    for index, image in enumerate(fresh):
        image.save(second / f"new{index}.png")

    result = compare(tmp_path / "second", tmp_path / "train", device=CPU, quiet=True)
    overlapping = result["correlation"] >= DUPLICATE

    assert overlapping.sum() == 6
    # And the four genuinely new images survive into the clean population.
    assert (~overlapping).sum() == 4
    assert all(
        "new" in path.name
        for path, flag in zip(result["left_paths"], overlapping)
        if not flag
    )
