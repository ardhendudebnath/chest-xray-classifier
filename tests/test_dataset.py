"""Loaders, label ordering and the two imbalance corrections."""

from collections import Counter

import pytest
import torch

from src.dataset import (
    CLASSES,
    IMG_SIZE,
    NUM_CLASSES,
    build_dataset,
    build_loader,
    build_transforms,
    class_counts,
    class_weights,
    make_sampler,
)


def test_transforms_produce_three_channel_square_tensors(xray_image):
    _, image = xray_image
    tensor = build_transforms(train=False)(image.convert("L"))
    assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)


def test_eval_transform_is_deterministic_and_train_is_not(xray_image):
    _, image = xray_image
    grayscale = image.convert("L")

    evaluation = build_transforms(train=False)
    assert torch.equal(evaluation(grayscale), evaluation(grayscale))

    torch.manual_seed(0)
    first = build_transforms(train=True)(grayscale)
    torch.manual_seed(1)
    second = build_transforms(train=True)(grayscale)
    assert not torch.equal(first, second)


def test_label_indices_match_the_declared_class_order(data_root):
    dataset = build_dataset(data_root / "train", train=False)
    assert dataset.class_to_idx == {name: i for i, name in enumerate(CLASSES)}


def test_missing_class_directory_is_rejected(tmp_path):
    """The failure this guards against renumbers labels without any error."""
    split = tmp_path / "train"
    for class_name in ("COVID19", "NORMAL"):
        (split / class_name).mkdir(parents=True)
        (split / class_name / "a.png").write_bytes(b"")

    with pytest.raises(ValueError, match="expected exactly"):
        build_dataset(split, train=False)


def test_absent_split_directory_names_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        build_dataset(tmp_path / "nope", train=False)


def test_class_counts_are_reported_for_every_class(data_root):
    counts = class_counts(build_dataset(data_root / "train", train=False))
    assert set(counts) == set(CLASSES)
    assert counts["PNEUMONIA"] > counts["COVID19"]


def test_class_weights_favour_the_rare_class(data_root):
    dataset = build_dataset(data_root / "train", train=False)
    weights = dict(zip(CLASSES, class_weights(dataset).tolist()))

    assert weights["COVID19"] > weights["NORMAL"] > weights["PNEUMONIA"]

    # Averaged over the dataset, not over the four classes -- the per-class
    # mean is only 1.0 when the classes are equal-sized. This is the form that
    # keeps a weighted run on the same loss scale as an unweighted one, and it
    # pins the constant: halving every weight, or dropping NUM_CLASSES from the
    # formula, still passes the ordering assertion above but not this one.
    counts = class_counts(dataset)
    weighted = sum(counts[name] * weights[name] for name in CLASSES)
    assert weighted / sum(counts.values()) == pytest.approx(1.0, rel=1e-5)


def test_sampler_draws_the_classes_about_evenly(data_root):
    """Without this the model can score well by ignoring COVID-19 entirely."""
    dataset = build_dataset(data_root / "train", train=False)
    sampler = make_sampler(dataset)

    torch.manual_seed(0)
    drawn = Counter(
        CLASSES[dataset.samples[i][1]] for _ in range(200) for i in iter(sampler)
    )
    share = {name: drawn[name] / sum(drawn.values()) for name in CLASSES}

    for name in CLASSES:
        assert share[name] == pytest.approx(1 / NUM_CLASSES, abs=0.08), share


def test_loader_batches_have_the_expected_shapes(data_root):
    loader, dataset = build_loader(data_root / "train", train=True, batch_size=4)
    images, targets = next(iter(loader))

    assert images.shape == (4, 3, IMG_SIZE, IMG_SIZE)
    assert targets.shape == (4,)
    assert set(targets.tolist()) <= set(range(len(CLASSES)))
    assert len(dataset) == sum(class_counts(dataset).values())


def test_validation_loader_does_not_resample(data_root):
    loader, dataset = build_loader(data_root / "val", train=False, batch_size=4)
    assert loader.sampler is not None  # SequentialSampler
    assert sum(len(batch) for batch, _ in loader) == len(dataset)
