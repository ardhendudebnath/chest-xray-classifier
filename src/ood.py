"""Is this even a chest X-ray? A distance check that softmax cannot do.

The four-class head has no way to say "none of these". Softmax normalises
whatever it is given, so any image at all gets a probability distribution, and
off-distribution inputs do not come out uncertain -- they come out wrong and
confident. Against `checkpoints/best.pt` a flat grey square scored COVID19 at
100.00%, a flat black one likewise, and uniform noise at 99.96%. Thresholding
the confidence catches none of that, because there is no confidence to threshold.

The signal is in the layer before. A logit says how much the image looks like
COVID19 relative to the other three; the penultimate feature vector says
what the image looks like at all, and off-distribution inputs land somewhere no
training image ever did. So: fit a Gaussian per class over the training
features, measure Mahalanobis distance to the nearest one, and reject anything
far enough out. This is Lee et al. 2018 ("A Simple Unified Framework for
Detecting Out-of-Distribution Samples and Adversarial Attacks"), which is the
standard version of this idea.

Three details that matter:

*Shrinkage.* The covariance is DxD -- 512x512 for resnet18 -- and is estimated
from at most a few thousand images. The empirical estimate is rank-deficient
whenever there are fewer images than feature dimensions, and merely
ill-conditioned when there are a few times more, so inverting it directly gives
a precision matrix built mostly out of noise. Ledoit-Wolf shrinks it toward a
scaled identity by an analytically chosen amount, which is what makes the
inverse stable enough to threshold.

*The threshold is calibrated, not chosen.* It is a percentile of the scores on
a held-out in-distribution split, so `--percentile 95` means about 5% of real
X-rays get flagged. That number is a knob with a cost on both sides and the
default is not a recommendation for any particular deployment.

*The threshold is per class, for the same reason selection is on macro F1
rather than accuracy.* One pooled cutoff is set by whichever class dominates
the calibration split. Measured here that was NORMAL, at two thirds of it, and
a pooled threshold that refused 0.4% of normal films refused 30.2% of real
pneumonia ones while reporting a reassuring 4.5% overall. Each class gets its
own cutoff, applied by whichever class the image is nearest.

    python -m src.ood --checkpoint checkpoints/best.pt --out checkpoints/ood.pt

Read what this does and does not establish in the README before quoting it. It
detects "unlike the training images", which is not the same as "not a chest
X-ray" and is much further still from "the model cannot handle this".

Lung opacity used to be the demonstration of that gap: real chest X-rays, of a
finding the model had no class for, mostly accepted here and then confidently
called NORMAL. It is a class now, which fixes those films and not the general
case. Any finding outside the four -- effusion, pneumothorax, a nodule -- is a
real chest X-ray that looks like the training data to this check, and will be
accepted and then labelled with whichever class is nearest. Widening the class
list moves that boundary; it does not remove it.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.dataset import CLASSES, NUM_CLASSES, build_loader, build_transforms
from src.model import (
    fingerprint_state_dict,
    forward_with_features,
    load_checkpoint,
    pick_device,
)

# 95 means roughly one real X-ray in twenty gets flagged, per class. Lower it
# and the check starts refusing ordinary images; raise it and it stops catching
# anything. There is no setting that is right for every caller, which is why
# the fitted stats record the percentile they were calibrated at.
DEFAULT_PERCENTILE = 95.0

# Below this many calibration images for a class, a 95th percentile is being
# read off a handful of points and the resulting cutoff is noise.
MIN_CALIBRATION_IMAGES = 20

DEFAULT_STATS_PATH = "checkpoints/ood.pt"


def collect_features(model, backbone, loader, device):
    """Returns (features (N, D), labels (N)) as float64/int64 numpy arrays.

    float64 because everything downstream is a covariance: accumulating outer
    products of thousands of 512-vectors in float32 loses enough precision to
    matter for the smallest eigenvalues, which are precisely the ones the
    inverse is most sensitive to.
    """
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            _, batch_features = forward_with_features(
                model, backbone, images.to(device)
            )
            features.append(batch_features.cpu().numpy())
            labels.append(targets.numpy())

    return (
        np.concatenate(features).astype(np.float64),
        np.concatenate(labels).astype(np.int64),
    )


def fit_gaussians(features, labels):
    """Class means and one shared precision matrix. Returns (means, precision, shrinkage).

    One covariance for all four classes rather than one each. Per-class
    covariances need per-class sample sizes to estimate them, and COVID19 and
    PNEUMONIA are both a fraction of NORMAL here -- the classes whose
    covariance would be worst estimated are the ones the model is least
    reliable on already. Tying them spends every image on a single
    better-conditioned estimate.
    """
    if features.ndim != 2:
        raise ValueError(f"Expected (N, D) features, got shape {features.shape}.")

    present = sorted(set(labels.tolist()))
    if present != list(range(NUM_CLASSES)):
        missing = [CLASSES[i] for i in range(NUM_CLASSES) if i not in present]
        raise ValueError(
            f"No features for {', '.join(missing)}. Every class needs a mean, "
            f"so fitting on a split that is missing one is not meaningful."
        )

    means = np.stack(
        [features[labels == index].mean(axis=0) for index in range(NUM_CLASSES)]
    )

    # Centre each image on its own class mean, then pool. This is the tied
    # (within-class) covariance: what is left after class identity is removed.
    centered = features - means[labels]

    from sklearn.covariance import LedoitWolf

    # assume_centered because the centring above is by class mean, not by the
    # grand mean -- letting the estimator subtract its own mean would put the
    # between-class spread back in and inflate every distance.
    estimator = LedoitWolf(assume_centered=True).fit(centered)

    return means, estimator.precision_, float(estimator.shrinkage_)


def class_distances(features, means, precision):
    """Squared Mahalanobis distance from each row to each class mean: (N, C).

    Everything is coerced to float64 numpy here, so callers can pass whatever
    they have -- the fitted stats hold torch tensors because that is what
    torch.save round-trips under weights_only, while the arithmetic wants numpy.
    """
    features = np.asarray(features, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)

    if features.ndim == 1:
        features = features[None, :]

    deltas = features[:, None, :] - means[None, :, :]
    return np.einsum("ncd,de,nce->nc", deltas, precision, deltas)


def score(features, stats):
    """Returns (distance to the nearest class mean, which class that was).

    Nearest rather than the predicted class's: the question is whether this
    image resembles anything the model was trained on, and an image sitting far
    from every class mean is off-distribution regardless of which one it is
    marginally closest to. The index comes back because the cutoff is per class
    and the caller needs to know which one to apply.
    """
    distances = class_distances(features, stats["means"], stats["precision"])
    return distances.min(axis=1), distances.argmin(axis=1)


def is_out_of_distribution(scores, nearest, stats):
    """Compare each score against the cutoff for the class it is nearest to."""
    thresholds = np.asarray(stats["thresholds"], dtype=np.float64)
    return np.asarray(scores) > thresholds[np.asarray(nearest)]


def calibrate(
    scores,
    nearest,
    percentile=DEFAULT_PERCENTILE,
    min_per_class=MIN_CALIBRATION_IMAGES,
):
    """One cutoff per class: the percentile of scores among images nearest to it.

    Grouped by nearest class rather than by true label, because that is the
    only one of the two available at serving time. Calibrating on the labels
    and then applying by nearest class would mean the cutoff and the images it
    is applied to were never the same population.
    """
    if not 0 < percentile < 100:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}.")

    scores = np.asarray(scores)
    nearest = np.asarray(nearest)
    thresholds = []

    for index in range(NUM_CLASSES):
        group = scores[nearest == index]
        if len(group) < min_per_class:
            raise ValueError(
                f"Only {len(group)} calibration images are nearest to "
                f"{CLASSES[index]}, and a p{percentile:g} cutoff read off that "
                f"few points is noise. Calibrate on a larger split, or lower "
                f"min_per_class deliberately."
            )
        thresholds.append(float(np.percentile(group, percentile)))

    return thresholds


def fit(
    model,
    backbone,
    fit_loader,
    calibration_loader,
    device,
    percentile=DEFAULT_PERCENTILE,
    min_per_class=MIN_CALIBRATION_IMAGES,
):
    """Fit on one split, calibrate the cutoffs on another. Returns a stats dict.

    Two different splits on purpose. The training features are what the
    Gaussians were fitted to, so they sit closer to their own means than unseen
    images do; calibrating on them would set the cutoffs too tight and flag
    real X-rays at several times the advertised rate.
    """
    features, labels = collect_features(model, backbone, fit_loader, device)
    means, precision, shrinkage = fit_gaussians(features, labels)

    stats = {
        "means": torch.from_numpy(means),
        "precision": torch.from_numpy(precision),
        "classes": list(CLASSES),
        "backbone": backbone,
        "feature_dim": int(features.shape[1]),
        "shrinkage": shrinkage,
        "fingerprint": fingerprint_state_dict(model.state_dict()),
        "fit_images": int(features.shape[0]),
    }

    calibration_features, _ = collect_features(
        model, backbone, calibration_loader, device
    )
    scores, nearest = score(calibration_features, stats)

    stats["thresholds"] = calibrate(scores, nearest, percentile, min_per_class)
    stats["percentile"] = float(percentile)
    stats["calibration_images"] = int(calibration_features.shape[0])
    # Plain floats: torch.load(weights_only=True) refuses to unpickle numpy
    # scalars, and that flag is worth keeping on.
    stats["calibration_quantiles"] = {
        str(int(q)): float(np.percentile(scores, q)) for q in (5, 25, 50, 75, 95, 99)
    }
    return stats


def save_stats(path, stats):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)
    return path


def load_stats(path, model=None):
    """Load fitted statistics, refusing them if they do not match this model.

    Statistics describe distances in one specific feature space. Loading a set
    fitted against a different checkpoint produces distances measured from the
    wrong means -- numbers with no meaning, and no symptom other than a
    rejection rate that looks a bit off. The fingerprint check makes that loud.
    """
    stats = torch.load(path, map_location="cpu", weights_only=True)

    if stats.get("classes") != list(CLASSES):
        raise ValueError(
            f"OOD stats were fitted for {stats.get('classes')}, but this code "
            f"expects {list(CLASSES)}."
        )

    if model is not None:
        expected = fingerprint_state_dict(model.state_dict())
        if stats.get("fingerprint") != expected:
            raise ValueError(
                f"OOD stats at {path} were fitted against a different "
                f"checkpoint ({stats.get('fingerprint')} vs {expected}). "
                f"Refit them with python -m src.ood."
            )

    return stats


def score_batch(model, backbone, batch, stats):
    """(scores, nearest) for a preprocessed batch."""
    with torch.no_grad():
        _, features = forward_with_features(model, backbone, batch)
    return score(features.cpu().numpy(), stats)


def score_image(model, backbone, pil_image, device, stats):
    """The same, straight from a PIL image. Returns (score, nearest index)."""
    tensor = build_transforms(train=False)(pil_image.convert("L"))
    scores, nearest = score_batch(
        model, backbone, tensor.unsqueeze(0).to(device), stats
    )
    return float(scores[0]), int(nearest[0])


def main():
    parser = argparse.ArgumentParser(
        description="Fit the out-of-distribution detector for a checkpoint."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--fit-split", default="train", choices=("train", "val", "test"))
    parser.add_argument(
        "--calibration-split", default="val", choices=("train", "val", "test")
    )
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument(
        "--report-split",
        default="test",
        choices=("train", "val", "test", "none"),
        help="Split to report the false-reject rate on. Defaults to test, "
        "which is used for neither fitting nor calibration -- the rate on the "
        "calibration split is the chosen percentile by construction.",
    )
    parser.add_argument("--out", default=DEFAULT_STATS_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.fit_split == args.calibration_split:
        parser.error(
            "--fit-split and --calibration-split must differ. Calibrating on "
            "the split the Gaussians were fitted to sets the cutoffs too "
            "tight, because training features sit closer to their own means."
        )

    device = pick_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    backbone = checkpoint["backbone"]
    print(f"checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']}, {backbone})")

    # train=False for both: augmentation exists to vary what the training loop
    # sees, and varying the features here would widen the fitted Gaussians to
    # cover crops and rotations that serving never produces.
    fit_loader, fit_dataset = build_loader(
        Path(args.data_dir) / args.fit_split,
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    calibration_loader, calibration_dataset = build_loader(
        Path(args.data_dir) / args.calibration_split,
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"fit on {args.fit_split}: {len(fit_dataset)} images")
    print(f"calibrate on {args.calibration_split}: {len(calibration_dataset)} images")

    stats = fit(
        model,
        backbone,
        fit_loader,
        calibration_loader,
        device,
        percentile=args.percentile,
    )

    print(f"\nfeature dim: {stats['feature_dim']}")
    print(f"Ledoit-Wolf shrinkage: {stats['shrinkage']:.4f}")
    print(f"\ncutoffs at p{stats['percentile']:g}, by nearest class")
    for name, threshold in zip(CLASSES, stats["thresholds"]):
        print(f"  {name:<10} {threshold:10.1f}")

    # The number with a cost attached: the fraction of real X-rays this refuses.
    # Measured on a split used for neither fitting nor calibration, and broken
    # out per class, because pooled it is dominated by whichever class is
    # largest -- the same reason src.evaluate leads with per-class recall.
    if args.report_split != "none":
        report_loader, report_dataset = build_loader(
            Path(args.data_dir) / args.report_split,
            train=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        features, labels = collect_features(model, backbone, report_loader, device)
        flagged = is_out_of_distribution(*score(features, stats), stats)

        print(f"\nfalse rejects on {args.report_split} -- real X-rays this refuses")
        for index, name in enumerate(CLASSES):
            group = flagged[labels == index]
            if len(group):
                print(
                    f"  {name:<10} {group.sum():>5}/{len(group):<5} {group.mean():>7.1%}"
                )
        print(
            f"  {'pooled':<10} {flagged.sum():>5}/{len(flagged):<5} "
            f"{flagged.mean():>7.1%}  (dominated by the largest class -- "
            f"read the rows above)"
        )
        print(f"  ({len(report_dataset)} images)")

    path = save_stats(args.out, stats)
    print(f"\nsaved {path}")
    print("the API picks this up from CXR_OOD_STATS, default checkpoints/ood.pt")


if __name__ == "__main__":
    main()
