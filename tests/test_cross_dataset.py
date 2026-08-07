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

from src.cross_dataset import absent_classes, label_of, labelled, predict, score_subset
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
    assert label_of(tmp_path / "Data" / "train" / "NORMAL" / "a.png") == CLASSES.index("NORMAL")
    assert label_of(tmp_path / "COVID19" / "images" / "a.png") == CLASSES.index("COVID19")
    assert label_of(tmp_path / "x" / "LUNG_OPACITY" / "y" / "a.png") == CLASSES.index("LUNG_OPACITY")


def test_the_deepest_class_directory_wins(tmp_path):
    """A path with two class names in it is ambiguous; the one nearest the
    image is the one that labelled it."""
    assert label_of(tmp_path / "NORMAL" / "COVID19" / "a.png") == CLASSES.index("COVID19")


def test_an_unlabelled_path_is_dropped_not_guessed(tmp_path):
    """Tuberculosis has no class here. Silently folding it into one of the four
    would put a finding the model has no output for into the score -- which is
    exactly the mistake that adding LUNG_OPACITY was meant to stop making."""
    assert label_of(tmp_path / "Tuberculosis" / "images" / "a.png") is None

    paths = [
        tmp_path / "NORMAL" / "a.png",
        tmp_path / "Tuberculosis" / "b.png",
        tmp_path / "COVID19" / "c.png",
    ]
    kept, labels, dropped = labelled(paths)

    assert len(kept) == 2
    assert labels.tolist() == [CLASSES.index("NORMAL"), CLASSES.index("COVID19")]
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


# ------------------------------------ scoring against a shorter label set


def images_at(tmp_path, count, seed):
    rng = np.random.default_rng(seed)
    paths = []
    for index in range(count):
        path = tmp_path / f"{index}.png"
        distinct_image(rng).save(path)
        paths.append(path)
    return paths


def test_absent_classes_reads_the_labels_not_the_configuration():
    """The second dataset here has no lung opacity directory at all, and which
    classes are missing is a property of the download rather than a setting."""
    labels = np.array([CLASSES.index("NORMAL"), CLASSES.index("PNEUMONIA")])

    missing = [CLASSES[index] for index in absent_classes(labels)]

    assert missing == ["COVID19", "LUNG_OPACITY"]


def test_matching_label_sets_leave_nothing_to_restrict():
    """When the two agree there is no second pass to run, and main skips it."""
    assert absent_classes(np.arange(len(CLASSES))) == []


def test_restricting_keeps_predictions_inside_the_allowed_classes(
    model_on_cpu, tmp_path
):
    model, backbone = model_on_cpu
    paths = images_at(tmp_path, 8, seed=10)
    allowed = [CLASSES.index("NORMAL"), CLASSES.index("PNEUMONIA")]

    predictions, _ = predict(model, backbone, paths, CPU, restrict_to=allowed)

    assert set(predictions.tolist()) <= set(allowed)


def test_restricting_scores_the_same_images_not_fewer(model_on_cpu, tmp_path):
    """Masked in the logits rather than by dropping images, so both passes
    describe the same population and their scores are comparable."""
    model, backbone = model_on_cpu
    paths = images_at(tmp_path, 8, seed=11)

    open_predictions, _ = predict(model, backbone, paths, CPU)
    restricted, _ = predict(
        model, backbone, paths, CPU, restrict_to=[CLASSES.index("NORMAL")]
    )

    assert len(open_predictions) == len(restricted) == len(paths)


def test_restricting_only_moves_the_predictions_it_has_to(model_on_cpu, tmp_path):
    """An image the model already placed inside the allowed set must come back
    unchanged. Anything else would mean the mask is altering the ranking among
    the classes it left alone, and the restricted pass would stop being a fair
    reading of the same model."""
    model, backbone = model_on_cpu
    paths = images_at(tmp_path, 12, seed=12)
    allowed = [CLASSES.index("COVID19"), CLASSES.index("NORMAL")]

    open_predictions, _ = predict(model, backbone, paths, CPU)
    restricted, _ = predict(model, backbone, paths, CPU, restrict_to=allowed)

    for before, after in zip(open_predictions.tolist(), restricted.tolist()):
        if before in allowed:
            assert after == before


def test_the_open_pass_counts_predictions_the_labels_cannot_express(
    model_on_cpu, tmp_path
):
    """The number that makes the open and restricted scores readable together:
    how often the model reached for a class this dataset never labels."""
    model, backbone = model_on_cpu
    paths = images_at(tmp_path, 10, seed=13)
    labels = np.full(len(paths), CLASSES.index("NORMAL"))

    result = score_subset(model, backbone, paths, labels, CPU)

    predictions, _ = predict(model, backbone, paths, CPU)
    expected = {
        CLASSES[index]: int((predictions == index).sum())
        for index in absent_classes(labels)
        if (predictions == index).any()
    }
    assert result["predicted_outside_labels"] == expected
    assert set(result["predicted_outside_labels"]) <= set(CLASSES) - {"NORMAL"}


def test_both_passes_average_over_the_same_classes(model_on_cpu, tmp_path):
    """The two numbers exist to be compared, so they have to be means over the
    same set. sklearn infers that set from labels-union-predictions when it is
    not told, which gave the open pass four classes -- it predicts the absent
    one, scoring 0.000 against zero support -- and the restricted pass three.
    On the real second dataset that inflated a 0.02 difference into 0.25, all
    of it denominator."""
    model, backbone = model_on_cpu
    paths = images_at(tmp_path, 12, seed=15)
    labels = np.array([CLASSES.index("NORMAL"), CLASSES.index("PNEUMONIA")] * 6)
    allowed = [CLASSES.index("NORMAL"), CLASSES.index("PNEUMONIA")]

    opened = score_subset(model, backbone, paths, labels, CPU)
    restricted = score_subset(model, backbone, paths, labels, CPU, restrict_to=allowed)

    assert opened["macro_f1_over"] == restricted["macro_f1_over"] == [
        CLASSES[index] for index in allowed
    ]


def test_a_class_the_dataset_never_labels_is_kept_out_of_the_mean(
    model_on_cpu, tmp_path
):
    """Its F1 is 0.000 against zero support, which is an artefact of the label
    set rather than a fact about the model. It stays visible in per_class_f1
    and out of the headline average."""
    model, backbone = model_on_cpu
    paths = images_at(tmp_path, 8, seed=16)
    labels = np.full(len(paths), CLASSES.index("NORMAL"))

    result = score_subset(model, backbone, paths, labels, CPU)

    assert result["macro_f1_over"] == ["NORMAL"]
    assert "LUNG_OPACITY" in result["per_class_f1"]
    assert result["macro_f1"] == pytest.approx(result["per_class_f1"]["NORMAL"])


def test_the_restricted_pass_leaves_nothing_outside_the_labels(
    model_on_cpu, tmp_path
):
    """Zero by construction. Asserted anyway, because it is the property that
    makes the restricted number a like-for-like against a shorter class list."""
    model, backbone = model_on_cpu
    paths = images_at(tmp_path, 10, seed=14)
    labels = np.full(len(paths), CLASSES.index("NORMAL"))

    result = score_subset(
        model, backbone, paths, labels, CPU,
        restrict_to=[CLASSES.index("NORMAL")],
    )

    assert result["predicted_outside_labels"] == {}
    assert result["accuracy"] == 1.0
