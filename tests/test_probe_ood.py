"""The six not-a-chest-X-ray probes.

Nothing here asserts that the model refuses them -- that needs trained weights,
and the suite runs on untrained ones. What is worth pinning is everything the
README's table depends on being true regardless of the checkpoint: that the
inputs are reproducible, that they go through the same preprocessing the API
uses, and that a row comes back with the fields the table has columns for.
"""

import numpy as np
import pytest
import torch

from src.dataset import CLASSES, NUM_CLASSES
from src.model import DEFAULT_BACKBONE, build_model
from src.ood import calibrate, fit_gaussians, score
from src.probe_ood import PROBES, SIZE, probe


@pytest.fixture(scope="module")
def model():
    return build_model(DEFAULT_BACKBONE, pretrained=False).eval()


@pytest.fixture(scope="module")
def stats():
    """Gaussians over random features -- meaningless distances, valid shapes.

    The cutoffs are calibrated on a second draw so `probe` gets a stats dict
    with every key it reads, which is what these tests are checking it reads.
    """
    rng = np.random.default_rng(0)
    features = rng.normal(0.0, 1.0, (40 * NUM_CLASSES, 512))
    labels = np.repeat(np.arange(NUM_CLASSES), 40)
    means, precision, _ = fit_gaussians(features, labels)

    fitted = {"means": means, "precision": precision}
    held_out = rng.normal(0.0, 1.0, (40 * NUM_CLASSES, 512))
    fitted["thresholds"] = calibrate(*score(held_out, fitted), percentile=95)
    return fitted


def test_there_are_six_of_them():
    """The README table has six rows and quotes a range across them."""
    assert len(PROBES) == 6
    assert len(dict(PROBES)) == 6, "probe names have to be unique to label rows"


@pytest.mark.parametrize("name,make", PROBES, ids=[name for name, _ in PROBES])
def test_each_probe_is_reproducible(name, make):
    """Two calls, identical bytes. A table regenerated against the same
    checkpoint has to produce the same numbers, or it is not a record of
    anything -- which is the failure that lost the original inputs."""
    assert make().tobytes() == make().tobytes()


@pytest.mark.parametrize("name,make", PROBES, ids=[name for name, _ in PROBES])
def test_each_probe_is_square_and_larger_than_the_input_size(name, make):
    """Bigger than 224, so the eval transform downsamples exactly as it does
    for a real upload rather than stretching a small image up."""
    image = make()
    assert image.size == (SIZE, SIZE)
    assert SIZE > 224


def test_the_colour_probe_is_actually_colour():
    """It is the only one testing the convert("L") step in the serving path,
    so a greyscale image here would leave that step unexercised."""
    by_name = dict(PROBES)
    assert by_name["smooth colour image"]().mode == "RGB"


def test_the_flat_probes_differ_from_each_other():
    """Three exposures, not three copies."""
    by_name = dict(PROBES)
    greys = {
        by_name[f"flat {shade} square"]().getpixel((0, 0))
        for shade in ("grey", "black", "white")
    }
    assert len(greys) == 3


def test_a_probe_row_has_every_field_the_table_reports(model, stats):
    row = probe(model, DEFAULT_BACKBONE, stats, torch.device("cpu"), PROBES[0][1]())

    assert row["prediction"] in CLASSES
    assert row["nearest"] in CLASSES
    assert 0.0 <= row["confidence"] <= 1.0
    assert row["low_confidence"] == (row["confidence"] < 0.6)
    assert row["ood_score"] >= 0.0
    assert row["out_of_distribution"] == (row["ood_score"] > row["ood_threshold"])


def test_the_cutoff_reported_is_the_nearest_class_one(model, stats):
    """The per-class cutoff is the whole design; reporting a different class's
    would make the refused column unreadable."""
    row = probe(model, DEFAULT_BACKBONE, stats, torch.device("cpu"), PROBES[3][1]())

    expected = stats["thresholds"][CLASSES.index(row["nearest"])]
    assert row["ood_threshold"] == pytest.approx(expected)


def test_every_probe_runs_through_the_model(model, stats):
    """Including the RGB one and the text one, which take different paths
    through convert("L")."""
    rows = [
        probe(model, DEFAULT_BACKBONE, stats, torch.device("cpu"), make())
        for _, make in PROBES
    ]

    assert len(rows) == len(PROBES)
    assert all(row["prediction"] in CLASSES for row in rows)
