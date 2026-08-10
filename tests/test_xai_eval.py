"""Metrics that score the explanations, where a sign error is invisible.

Deletion AUC is better when low and insertion AUC is better when high, and the
two are computed from the same mask used in opposite directions. Swap that one
`torch.where` and both numbers invert: the module would report that Grad-CAM is
performing well precisely when it is performing badly, and every value would
still sit in [0, 1] and still respond sensibly to changes in the model. No
downstream check could catch it, so the endpoints of both curves are pinned
here against what they have to be by definition.

The localisation numbers get the same treatment for a different reason. A
"fraction of heat inside the lungs" of 0.25 sounds poor and is exactly what a
uniform map scores, because lungs are about a quarter of a radiograph. The
tests fix that null expectation so the enrichment ratio cannot quietly start
reporting the raw fraction.
"""

import json
import sys

import numpy as np
import pytest
import torch

from src.dataset import CLASSES, build_loader
from src.model import load_checkpoint, target_layer
from src.xai_eval import (
    _stepped_masks,
    blurred,
    curve_auc,
    curves,
    enrichment,
    evaluate_image,
    lung_mass_fraction,
    main,
    pairwise_cosine,
    perturbation_order,
    stability,
    summarise,
)

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def loaded(checkpoint):
    model, meta = load_checkpoint(checkpoint, DEVICE)
    return model, meta["backbone"]


@pytest.fixture
def one_image(data_root):
    loader, _ = build_loader(
        data_root / "test", train=False, batch_size=1, num_workers=0
    )
    images, _ = next(iter(loader))
    return images


# ----------------------------------------------------------------- the curves


def test_the_curves_start_and_end_where_they_must(loaded, one_image):
    """The assertion the whole module rests on.

    Deletion begins with the untouched image and ends fully blurred; insertion
    is the exact reverse. A swapped `where` inverts both metrics while leaving
    every number in range and every trend plausible.
    """
    model, _ = loaded
    order = perturbation_order(np.random.default_rng(1).random((224, 224)))

    deletion, insertion = curves(model, one_image, order, class_idx=0, steps=8)

    with torch.no_grad():
        original = torch.softmax(model(one_image), dim=1)[0, 0].item()
        blurred_probability = torch.softmax(model(blurred(one_image)), dim=1)[0, 0].item()

    assert deletion[0] == pytest.approx(original, abs=1e-5)
    assert deletion[-1] == pytest.approx(blurred_probability, abs=1e-5)
    assert insertion[0] == pytest.approx(blurred_probability, abs=1e-5)
    assert insertion[-1] == pytest.approx(original, abs=1e-5)


def test_both_curves_have_one_point_per_step_plus_the_start(loaded, one_image):
    model, _ = loaded
    order = perturbation_order(np.random.default_rng(1).random((224, 224)))

    deletion, insertion = curves(model, one_image, order, class_idx=0, steps=12)

    assert deletion.shape == insertion.shape == (13,)


def test_the_curves_are_computed_in_batches_without_reordering(loaded, one_image):
    """The step masks are chunked to keep memory flat, which is where an
    off-by-one would land the wrong probability at the wrong fraction."""
    model, _ = loaded
    order = perturbation_order(np.random.default_rng(1).random((224, 224)))

    small = curves(model, one_image, order, 0, steps=10, batch_size=2)
    large = curves(model, one_image, order, 0, steps=10, batch_size=64)

    assert np.allclose(small[0], large[0], atol=1e-6)
    assert np.allclose(small[1], large[1], atol=1e-6)


# ------------------------------------------------------------------- the masks


def test_the_masks_reveal_progressively():
    order = np.arange(100)
    masks = _stepped_masks(order, steps=4)

    assert masks.shape == (5, 100)
    assert not masks[0].any()
    assert masks[-1].all()
    assert [int(row.sum()) for row in masks] == [0, 25, 50, 75, 100]


def test_each_mask_contains_the_one_before_it():
    """Nested by construction. A step that un-removed a pixel would make the
    curve non-monotone in what has been perturbed, which is not what either
    metric is defined over."""
    order = np.random.default_rng(0).permutation(64)
    masks = _stepped_masks(order, steps=8)

    for earlier, later in zip(masks, masks[1:]):
        assert np.all(later[earlier])


# ------------------------------------------------------------------- ordering


def test_the_highest_attributed_pixel_is_removed_first():
    attribution = np.array([[0.1, 0.9], [0.5, 0.2]])
    order = perturbation_order(attribution)

    assert order[0] == 1
    assert order[-1] == 0


def test_ties_are_not_broken_in_raster_order():
    """Grad-CAM upsamples a 7x7 grid, so large blocks share a value exactly.
    Breaking those ties by index would delete top-left first and put a spatial
    bias into a metric that is supposed to be about ranking alone."""
    flat = perturbation_order(np.ones((16, 16)), np.random.default_rng(3))

    assert flat.tolist() != list(range(256))
    assert sorted(flat.tolist()) == list(range(256))


# ---------------------------------------------------------------- the area


def test_curve_auc_of_a_flat_curve_is_its_value():
    assert curve_auc(np.full(11, 0.7)) == pytest.approx(0.7)


def test_curve_auc_of_a_straight_ramp_is_a_half():
    """Trapezoid, not a mean: a mean over 11 evenly spaced points of a ramp
    gives 0.5 too, but only because the endpoints happen to be symmetric. A
    curve that starts high and ends high would separate them."""
    assert curve_auc(np.linspace(0.0, 1.0, 11)) == pytest.approx(0.5)


def test_curve_auc_weights_the_endpoints_at_half():
    assert curve_auc(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.25)


def test_a_one_point_curve_has_no_area():
    with pytest.raises(ValueError, match="at least two points"):
        curve_auc(np.array([1.0]))


# -------------------------------------------------------------- localisation


def test_a_uniform_map_scores_the_area_fraction_not_a_good_number():
    """The null result this metric exists to be read against. A map that has
    localised nothing already puts a quarter of its mass in the lungs, because
    the lungs are a quarter of the picture."""
    cam = np.ones((10, 10))
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, :3] = True

    mass, area = lung_mass_fraction(cam, mask)

    assert mass == pytest.approx(0.3)
    assert area == pytest.approx(0.3)
    assert enrichment(mass, area) == pytest.approx(1.0)


def test_a_map_entirely_inside_the_lungs_scores_one():
    cam = np.zeros((10, 10))
    cam[0, :3] = 1.0
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, :3] = True

    mass, area = lung_mass_fraction(cam, mask)

    assert mass == pytest.approx(1.0)
    assert enrichment(mass, area) == pytest.approx(1 / 0.3)


def test_a_map_outside_the_lungs_scores_zero():
    cam = np.zeros((10, 10))
    cam[0, 9] = 1.0
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, :3] = True

    mass, _ = lung_mass_fraction(cam, mask)
    assert mass == pytest.approx(0.0)


def test_an_all_zero_map_is_not_a_division_by_zero():
    mass, area = lung_mass_fraction(np.zeros((8, 8)), np.ones((8, 8), dtype=bool))
    assert mass == 0.0 and area == 1.0


def test_a_mask_of_the_wrong_size_is_refused():
    """The masks ship at 256x256 and the CAM comes back at 224x224. Numpy would
    broadcast some mismatches rather than raising."""
    with pytest.raises(ValueError, match="but the mask is"):
        lung_mass_fraction(np.ones((10, 10)), np.ones((8, 8), dtype=bool))


# ---------------------------------------------------------------- stability


def test_identical_vectors_are_perfectly_similar():
    assert pairwise_cosine(np.ones((4, 6))) == pytest.approx(1.0)


def test_orthogonal_vectors_are_not_similar():
    assert pairwise_cosine(np.eye(2)) == pytest.approx(0.0)


def test_a_single_vector_has_no_pairwise_similarity():
    assert pairwise_cosine(np.ones((1, 5))) is None


def test_a_zero_vector_does_not_produce_nan():
    assert np.isfinite(pairwise_cosine(np.zeros((3, 5))))


def test_stability_reports_the_cross_class_control():
    """Every SHAP vector here is w * (x - E[x]) for one shared w, so the
    vectors agree in direction before any similarity between the images is
    considered. Without the control the within-class number reads as evidence
    of consistency when it is partly arithmetic."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(12, 8))
    labels = np.array([0, 1, 2, 3] * 3)

    result = stability(vectors, labels, top_k=3)

    assert result["all_pairs_cosine"] is not None
    assert set(result["per_class"]) == set(CLASSES)
    assert result["per_class"]["COVID19"]["images"] == 3
    assert 0.0 <= result["per_class"]["COVID19"]["mean_top_k_coverage"] <= 1.0


def test_summarise_skips_columns_that_were_not_computed():
    """deletion_ood_rate is only present when OOD statistics were loaded, and
    averaging over rows that lack it would silently treat absent as zero."""
    rows = [
        {"deletion_auc": 0.2, "deletion_ood_rate": 0.5, "predicted": "NORMAL"},
        {"deletion_auc": 0.4, "predicted": "COVID19"},
    ]
    summary = summarise(rows)

    assert summary["deletion_auc"] == pytest.approx(0.3)
    assert summary["deletion_ood_rate"] == pytest.approx(0.5)
    assert "predicted" not in summary


# ------------------------------------------------------------- the whole path


def test_evaluate_image_produces_four_areas_in_range(loaded, one_image):
    model, backbone = loaded
    row, cam, class_idx = evaluate_image(
        model, backbone, one_image, steps=8, generator=np.random.default_rng(0)
    )

    assert row["predicted"] in CLASSES
    assert cam.shape == one_image.shape[-2:]
    assert 0 <= class_idx < len(CLASSES)
    for key in (
        "deletion_auc",
        "insertion_auc",
        "random_deletion_auc",
        "random_insertion_auc",
    ):
        assert 0.0 <= row[key] <= 1.0


def test_the_gradcam_hook_does_not_leak(loaded, one_image):
    """evaluate_image runs Grad-CAM inside a context manager and then does
    dozens more forward passes. A hook left attached would keep every one of
    those activation tensors alive."""
    model, backbone = loaded
    before = len(target_layer(model, backbone)._forward_hooks)

    evaluate_image(model, backbone, one_image, steps=4)

    assert len(target_layer(model, backbone)._forward_hooks) == before


# -------------------------------------------------------------------- main


def test_main_writes_a_report(monkeypatch, tmp_path, data_root, checkpoint, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "src.xai_eval",
            "--checkpoint", str(checkpoint),
            "--data-dir", str(data_root),
            "--split", "test",
            "--limit", "2",
            "--steps", "4",
            "--background", str(tmp_path / "absent.pt"),
            "--report-dir", str(tmp_path / "reports"),
            "--device", "cpu",
        ],
    )
    main()

    path = tmp_path / "reports" / "xai_fidelity_test.json"
    summary = json.loads(path.read_text("utf-8"))

    assert summary["images"] == 2
    assert summary["steps"] == 4
    assert 0.0 <= summary["fidelity"]["deletion_auc"] <= 1.0
    # No background file exists, so stability is skipped rather than guessed.
    assert "stability" not in summary
    assert "no SHAP background" in capsys.readouterr().out
