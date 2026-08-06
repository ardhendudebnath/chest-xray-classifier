"""The out-of-distribution detector: the estimator, and the plumbing round it.

The estimator is tested against Gaussians drawn to order rather than against
the model, because "does Mahalanobis distance do what it should" is a question
about arithmetic and the untrained weights the rest of the suite uses would
only add noise to it. What does need the model is the feature extraction and
the checkpoint pairing, and those are exercised separately below.
"""

import numpy as np
import pytest
import torch

from src.dataset import CLASSES, NUM_CLASSES
from src.model import (
    build_model,
    fingerprint_state_dict,
    forward_with_features,
    load_checkpoint,
)
from src.ood import (
    calibrate,
    class_distances,
    collect_features,
    fit,
    fit_gaussians,
    is_out_of_distribution,
    load_stats,
    save_stats,
    score,
)

TRUE_MEANS = np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]])


def gaussian_features(rng, per_class=400, dim=2, means=None):
    """Points around three well-separated means, sharing one covariance."""
    means = TRUE_MEANS if means is None else means
    padded = np.zeros((NUM_CLASSES, dim))
    padded[:, : means.shape[1]] = means[:, :dim] if dim < means.shape[1] else means

    features = np.concatenate(
        [rng.normal(padded[index], 1.0, (per_class, dim)) for index in range(NUM_CLASSES)]
    )
    labels = np.repeat(np.arange(NUM_CLASSES), per_class)
    return features, labels


@pytest.fixture
def fitted():
    """Stats over known Gaussians, with cutoffs calibrated on fresh draws."""
    rng = np.random.default_rng(0)
    features, labels = gaussian_features(rng)
    means, precision, shrinkage = fit_gaussians(features, labels)

    stats = {"means": means, "precision": precision, "shrinkage": shrinkage}
    held_out, _ = gaussian_features(np.random.default_rng(1), per_class=200)
    stats["thresholds"] = calibrate(*score(held_out, stats), percentile=95)
    return stats


# ------------------------------------------------------------------ estimator


def test_fit_recovers_the_means_it_was_given():
    features, labels = gaussian_features(np.random.default_rng(0))
    means, _, _ = fit_gaussians(features, labels)

    assert means.shape == (NUM_CLASSES, 2)
    assert np.allclose(means, TRUE_MEANS, atol=0.2)


def test_in_distribution_distance_matches_the_dimension():
    """Squared Mahalanobis distance to one's own mean is chi-square with d
    degrees of freedom, so it averages d. Getting this scale right is what
    makes a percentile cutoff mean anything."""
    features, labels = gaussian_features(np.random.default_rng(0), dim=2)
    means, precision, _ = fit_gaussians(features, labels)

    scores, _ = score(features, {"means": means, "precision": precision})
    assert scores.mean() == pytest.approx(2.0, rel=0.15)


def test_a_distant_point_scores_far_higher_than_the_training_data(fitted):
    scores, _ = score(np.array([[500.0, 500.0]]), fitted)
    in_distribution, _ = score(gaussian_features(np.random.default_rng(2))[0], fitted)

    assert scores[0] > 1000 * in_distribution.mean()
    assert bool(is_out_of_distribution(*score(np.array([[500.0, 500.0]]), fitted), fitted))


def test_the_nearest_class_is_reported(fitted):
    """The cutoff applied depends on this index, so it has to be the right one."""
    _, nearest = score(TRUE_MEANS, fitted)
    assert nearest.tolist() == [0, 1, 2]


def test_shrinkage_survives_more_dimensions_than_samples():
    """The empirical covariance is singular here -- 30 points in 64 dimensions.
    Inverting it unshrunk is what this guards against; the distances have to
    come out finite and positive regardless."""
    features, labels = gaussian_features(
        np.random.default_rng(0), per_class=10, dim=64
    )
    assert features.shape[0] < features.shape[1]

    means, precision, shrinkage = fit_gaussians(features, labels)
    assert shrinkage > 0
    assert np.isfinite(precision).all()

    distances = class_distances(features, means, precision)
    assert np.isfinite(distances).all()
    assert (distances >= 0).all()


def test_a_class_with_no_features_is_refused():
    features, labels = gaussian_features(np.random.default_rng(0), per_class=20)
    keep = labels != 1

    with pytest.raises(ValueError, match=CLASSES[1]):
        fit_gaussians(features[keep], labels[keep])


def test_distance_to_a_point_on_the_mean_is_zero(fitted):
    # The fitted mean, not the one the samples were drawn around -- those
    # differ by sampling error, which is a fact about the draw and not about
    # the distance.
    distances = class_distances(
        fitted["means"][1], fitted["means"], fitted["precision"]
    )
    assert distances[0, 1] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- calibration


def test_cutoffs_are_per_class_not_pooled(fitted):
    """Three cutoffs, and they are applied by nearest class. A pooled cutoff
    set by the largest class is the bug this design exists to avoid."""
    assert len(fitted["thresholds"]) == NUM_CLASSES

    stats = {**fitted, "thresholds": [1e12, 0.0, 1e12]}
    scores, nearest = score(TRUE_MEANS, stats)

    # Every point sits on its own mean, so distance is ~0 for all three. Only
    # the one judged against the zero cutoff comes back flagged.
    assert is_out_of_distribution(scores, nearest, stats).tolist() == [False, True, False]


def test_the_percentile_is_the_flagged_rate():
    rng = np.random.default_rng(0)
    features, labels = gaussian_features(rng)
    means, precision, _ = fit_gaussians(features, labels)
    stats = {"means": means, "precision": precision}

    held_out, _ = gaussian_features(np.random.default_rng(1), per_class=500)
    scores, nearest = score(held_out, stats)
    stats["thresholds"] = calibrate(scores, nearest, percentile=90)

    flagged = is_out_of_distribution(scores, nearest, stats)
    assert flagged.mean() == pytest.approx(0.10, abs=0.02)


def test_too_few_calibration_images_is_an_error_not_a_guess(fitted):
    scores, nearest = score(gaussian_features(np.random.default_rng(3))[0], fitted)

    with pytest.raises(ValueError, match="noise"):
        calibrate(scores[:5], nearest[:5], min_per_class=20)


@pytest.mark.parametrize("percentile", [0, 100, -5, 101])
def test_a_percentile_outside_the_open_interval_is_refused(fitted, percentile):
    scores, nearest = score(gaussian_features(np.random.default_rng(3))[0], fitted)
    with pytest.raises(ValueError, match="percentile"):
        calibrate(scores, nearest, percentile=percentile, min_per_class=1)


# --------------------------------------------------------------- the features


def test_features_come_back_with_the_logits(checkpoint):
    model, meta = load_checkpoint(checkpoint, "cpu")
    batch = torch.randn(4, 3, 224, 224)

    logits, features = forward_with_features(model, meta["backbone"], batch)

    assert logits.shape == (4, NUM_CLASSES)
    # resnet18's head takes 512 numbers, and that is the space distances live in.
    assert features.shape == (4, 512)


def test_the_hook_does_not_outlive_the_call(checkpoint):
    """A forward hook left attached keeps every activation tensor alive."""
    model, meta = load_checkpoint(checkpoint, "cpu")
    before = len(model.fc._forward_hooks)

    forward_with_features(model, meta["backbone"], torch.randn(1, 3, 224, 224))

    assert len(model.fc._forward_hooks) == before


def test_collect_features_lines_up_with_its_labels(checkpoint, data_root):
    from src.dataset import build_loader

    model, meta = load_checkpoint(checkpoint, "cpu")
    loader, dataset = build_loader(data_root / "train", train=False, batch_size=8)

    features, labels = collect_features(model, meta["backbone"], loader, "cpu")

    assert features.shape == (len(dataset), 512)
    assert labels.shape == (len(dataset),)
    assert sorted(set(labels.tolist())) == list(range(NUM_CLASSES))


# ------------------------------------------------------ pairing and round-trip


def test_stats_round_trip_through_disk(checkpoint, data_root, tmp_path):
    from src.dataset import build_loader

    model, meta = load_checkpoint(checkpoint, "cpu")
    fit_loader, _ = build_loader(data_root / "train", train=False, batch_size=8)
    calibration_loader, _ = build_loader(data_root / "test", train=False, batch_size=8)

    stats = fit(
        model,
        meta["backbone"],
        fit_loader,
        calibration_loader,
        "cpu",
        # The fixture splits hold a handful of images. The real default refuses
        # to read a percentile off that few, which is the point of it.
        min_per_class=1,
    )
    path = save_stats(tmp_path / "ood.pt", stats)

    # weights_only=True is what load_stats uses, and it refuses numpy scalars.
    # Anything that leaked one into the dict fails here rather than in serving.
    reloaded = load_stats(path, model)

    assert reloaded["classes"] == CLASSES
    assert reloaded["feature_dim"] == 512
    assert np.allclose(reloaded["means"].numpy(), stats["means"].numpy())
    assert reloaded["thresholds"] == stats["thresholds"]


def test_stats_fitted_against_another_checkpoint_are_refused(
    checkpoint, data_root, tmp_path
):
    """Loading these anyway would measure distances from the wrong means and
    report a number with nothing wrong on its face."""
    from src.dataset import build_loader

    model, meta = load_checkpoint(checkpoint, "cpu")
    fit_loader, _ = build_loader(data_root / "train", train=False, batch_size=8)
    calibration_loader, _ = build_loader(data_root / "test", train=False, batch_size=8)

    stats = fit(
        model, meta["backbone"], fit_loader, calibration_loader, "cpu", min_per_class=1
    )
    path = save_stats(tmp_path / "ood.pt", stats)

    torch.manual_seed(99)
    other = build_model("resnet18", pretrained=False)
    assert fingerprint_state_dict(other.state_dict()) != stats["fingerprint"]

    with pytest.raises(ValueError, match="different"):
        load_stats(path, other)


def test_the_fingerprint_tracks_the_weights():
    torch.manual_seed(0)
    one = build_model("resnet18", pretrained=False)
    torch.manual_seed(0)
    same = build_model("resnet18", pretrained=False)
    torch.manual_seed(1)
    different = build_model("resnet18", pretrained=False)

    assert fingerprint_state_dict(one.state_dict()) == fingerprint_state_dict(
        same.state_dict()
    )
    assert fingerprint_state_dict(one.state_dict()) != fingerprint_state_dict(
        different.state_dict()
    )
