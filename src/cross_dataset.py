"""Score a checkpoint on a second dataset, with its own training images removed.

The whole project rests on numbers from one dataset whose classes are separable
by provenance, so the obvious next move is to score against a different
download. Done naively that measures nothing: the public chest X-ray sets
re-package each other, and `src.dataset_overlap` found that 24.0% of
`prashant268/chest-xray-covid19-pneumonia` is already in this model's training
split -- with zero matching checksums, because every copy had been resized or
re-encoded on the way in.

So the images are partitioned first and three numbers come out, not one:

    all          every labelled image. Contaminated. Not quotable.
    overlapping  the ones that are training images. A control -- if the filter
                 picked the right images these score near the training data,
                 and if they do not then the matching is wrong.
    clean        what is left. The only one worth reporting.

Reporting all three is the point. A single filtered number gives a reader no
way to check that the filter did anything, and the gap between the first and
the last is what the contamination was worth.

**When the second dataset has fewer classes than the model.** This one does:
`prashant268` carries NORMAL, PNEUMONIA and COVID19 and no lung opacity at all,
while the model has four outputs. So what is a LUNG_OPACITY prediction here?
There is no answer that is simply correct, and the two defensible ones measure
different things, so both are reported for the same reason all three
populations are:

    open        all four outputs live. A LUNG_OPACITY prediction counts as
                wrong, because these labels cannot confirm it. A lower bound:
                some of those films may really show an opacity that this
                dataset had no category for and filed under something else.
    restricted  the absent classes are masked out of the logits before argmax,
                so the model must choose among the classes the dataset knows.
                This is the like-for-like against a three-class score.

The gap between them is how often the model reaches for a class this dataset
cannot express. Neither number is the honest one on its own: quote `open` and
you may be charging the model for being right in a vocabulary the labels lack;
quote `restricted` and you are hiding how often it wanted to say something else.

    python -m src.cross_dataset --dataset ~/Downloads/chest-xray-cp/Data \\
        --exclude-against ~/cxr-data-4class/train

**A filtered score is still not a different-provenance score.** Removing shared
images removes image-level contamination and nothing else. Both datasets are
compiled from the same handful of public archives, so a non-duplicate image can
easily come from the same collection as a training one. This answers "was the
model scored on its own training data", not "does the model read pathology
rather than which repository produced the film".
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.dataset import CLASSES, build_transforms
from src.dataset_overlap import DUPLICATE, compare
from src.model import forward_with_features, load_checkpoint, pick_device
from src.ood import is_out_of_distribution, load_stats, score


def label_of(path):
    """The class index for an image, from whichever path part names a class.

    Read from the path rather than a manifest because these downloads arrive
    as one directory per class, but at different depths -- Data/train/NORMAL/
    here, NORMAL/images/ elsewhere. Returns None when no part names a class,
    which is how directories outside the four classes get dropped.
    """
    for part in reversed(path.parts[:-1]):
        if part in CLASSES:
            return CLASSES.index(part)
    return None


def labelled(paths):
    """Split paths into (kept, labels, dropped) by whether a class was found."""
    kept, labels, dropped = [], [], []
    for path in paths:
        index = label_of(path)
        if index is None:
            dropped.append(path)
        else:
            kept.append(path)
            labels.append(index)
    return kept, np.array(labels, dtype=np.int64), dropped


def absent_classes(labels):
    """Class indices the model can predict but this dataset never labels.

    Read off the labels actually found rather than configured, because it is a
    property of the download in front of us. An empty result means the two
    label sets line up and the restricted pass has nothing to do.
    """
    present = set(int(label) for label in labels)
    return [index for index in range(len(CLASSES)) if index not in present]


def predict(model, backbone, paths, device, stats=None, batch_size=64, restrict_to=None):
    """Returns (predictions, out-of-distribution flags) for a list of images.

    The flags are all False when no statistics are given. That is reported as
    "not checked" by the caller rather than folded into the numbers, since a
    detector that never fires and one that was never run look identical here.

    restrict_to limits argmax to those class indices by driving the rest to
    -inf. Done to the logits rather than by dropping images, so every image is
    still scored and the two passes cover the same population -- the point is
    what the model says when it cannot reach for a class this dataset has no
    label for, not what it says about a smaller set of films.
    """
    transform = build_transforms(train=False)
    predictions, flags = [], []

    for start in range(0, len(paths), batch_size):
        batch = torch.stack(
            [
                transform(Image.open(path).convert("L"))
                for path in paths[start : start + batch_size]
            ]
        ).to(device)

        with torch.no_grad():
            logits, features = forward_with_features(model, backbone, batch)

        if restrict_to is not None:
            allowed = torch.full((len(CLASSES),), float("-inf"), device=logits.device)
            allowed[list(restrict_to)] = 0.0
            logits = logits + allowed

        predictions.append(logits.argmax(dim=1).cpu().numpy())
        if stats is not None:
            flags.append(is_out_of_distribution(*score(features.cpu().numpy(), stats), stats))

    if not predictions:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)

    return (
        np.concatenate(predictions),
        np.concatenate(flags) if flags else np.zeros(len(paths), dtype=bool),
    )


def score_subset(
    model, backbone, paths, labels, device, stats=None, batch_size=64, restrict_to=None
):
    """Metrics for one population. Returns a dict, or None when it is empty."""
    if len(paths) == 0:
        return None

    predictions, flags = predict(
        model, backbone, paths, device, stats, batch_size, restrict_to
    )
    matrix = confusion_matrix(labels, predictions, labels=range(len(CLASSES)))

    # Average over the classes this dataset labels, and say so, rather than
    # letting sklearn infer the set from labels-union-predictions. Inferred, the
    # open pass averages over four classes -- it predicts LUNG_OPACITY, which
    # scores 0.000 against zero support -- while the restricted pass averages
    # over three. That put a 0.25 gap between two numbers whose real difference
    # was 0.02, entirely from the denominator, and the two passes exist to be
    # compared. Predicting outside the label set is still charged for: it costs
    # the true class its recall, which these classes' scores do see.
    present = [index for index in range(len(CLASSES)) if index not in absent_classes(labels)]

    result = {
        "images": len(paths),
        # How many landed in a class this dataset never labels. Zero by
        # construction under restrict_to; the number worth reading is the one
        # from the open pass.
        "predicted_outside_labels": {
            CLASSES[index]: int((predictions == index).sum())
            for index in absent_classes(labels)
            if (predictions == index).any()
        },
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", labels=present, zero_division=0)
        ),
        # Named so nobody has to work out what the average ran over.
        "macro_f1_over": [CLASSES[index] for index in present],
        "confusion_matrix": matrix.tolist(),
        # Per class over all four regardless, because a zero against a class
        # with no support is information: it says the dataset never labels it.
        "per_class_f1": {
            name: float(value)
            for name, value in zip(
                CLASSES,
                f1_score(labels, predictions, average=None, labels=range(len(CLASSES)), zero_division=0),
            )
        },
        "report": classification_report(
            labels, predictions, labels=present,
            target_names=[CLASSES[index] for index in present],
            digits=3, zero_division=0,
        ),
    }

    if stats is not None:
        result["ood_refused"] = {
            name: float(flags[labels == index].mean())
            for index, name in enumerate(CLASSES)
            if (labels == index).any()
        }
        result["ood_refused_pooled"] = float(flags.mean())

    return result


def print_subset(title, result):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    if result is None:
        print("no images")
        return

    print(result["report"])
    print(f"macro F1 {result['macro_f1']:.4f}   accuracy {result['accuracy']:.4f}")

    # Width derived from the class names rather than fixed, for the reason
    # src.evaluate carries the same note: a name as long as the column fills it
    # edge to edge and fuses with its neighbour in the header.
    width = max(len(name) for name in CLASSES) + 2
    print("\nconfusion (rows actual, cols predicted)")
    print(" " * width + "".join(f"{name:>{width}}" for name in CLASSES))
    for name, row in zip(CLASSES, result["confusion_matrix"]):
        print(f"{name:<{width}}" + "".join(f"{value:>{width}}" for value in row))

    outside = result.get("predicted_outside_labels")
    if outside:
        total = sum(outside.values())
        detail = ", ".join(f"{name} {count}" for name, count in outside.items())
        print(
            f"\n{total} of {result['images']} images ({total / result['images']:.1%}) "
            f"were predicted into a class this dataset never labels: {detail}."
        )
        print("Counted as errors above. The restricted pass below removes them.")

    if "ood_refused" in result:
        print("\nout-of-distribution check -- real X-rays refused")
        for name, fraction in result["ood_refused"].items():
            print(f"  {name:<10} {fraction:>7.1%}")
        print(f"  {'pooled':<10} {result['ood_refused_pooled']:>7.1%}")
    else:
        print("\nout-of-distribution check: not run (no statistics loaded)")


def main():
    parser = argparse.ArgumentParser(
        description="Score a checkpoint on a second dataset, minus shared images."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--dataset", required=True, help="The second dataset's root.")
    parser.add_argument(
        "--exclude-against",
        required=True,
        help="The training split whose images must be removed from the score.",
    )
    parser.add_argument("--ood-stats", default="checkpoints/ood.pt")
    parser.add_argument("--threshold", type=float, default=DUPLICATE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default=None, help="Write the metrics here as JSON.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    model, meta = load_checkpoint(args.checkpoint, device)
    backbone = meta["backbone"]
    print(f"checkpoint: {args.checkpoint} (epoch {meta['epoch']}, {backbone})")

    try:
        stats = load_stats(args.ood_stats, model)
    except (FileNotFoundError, ValueError) as error:
        print(f"no usable OOD stats ({error}); that section will be skipped")
        stats = None

    print(f"\nmatching {args.dataset} against {args.exclude_against}...")
    result = compare(args.dataset, args.exclude_against, device=device, quiet=True)

    paths, labels, dropped = labelled(result["left_paths"])
    if dropped:
        print(f"dropped {len(dropped)} images in no class directory")
    if not paths:
        raise SystemExit(
            f"No images under {args.dataset} sit in a directory named after one "
            f"of {CLASSES}. Nothing to score."
        )

    # Realign the correlations onto the images that survived labelling.
    keep = np.array([label_of(path) is not None for path in result["left_paths"]])
    correlation = result["correlation"][keep]
    overlapping = correlation >= args.threshold

    print(f"{len(paths)} labelled images, {overlapping.sum()} "
          f"({overlapping.mean():.1%}) already in the training split")

    populations = {
        "all": (np.ones(len(paths), dtype=bool),
                "A. ALL IMAGES -- contaminated, not quotable"),
        "overlapping": (overlapping,
                        "B. OVERLAPPING ONLY -- training images, a control"),
        "clean": (~overlapping,
                  "C. CLEAN -- training images removed, the honest number"),
    }

    missing = absent_classes(labels)
    if missing:
        print(
            f"this dataset never labels {[CLASSES[i] for i in missing]}, which the "
            f"model can still predict -- scoring both open and restricted"
        )
    allowed = [index for index in range(len(CLASSES)) if index not in missing]

    metrics = {}
    for key, (mask, title) in populations.items():
        chosen = [paths[i] for i in np.flatnonzero(mask)]
        metrics[key] = score_subset(
            model, backbone, chosen, labels[mask], device, stats, args.batch_size
        )
        print_subset(title, metrics[key])

        # Second pass over the same images with the absent classes masked out
        # of the logits. Skipped entirely when the label sets already agree,
        # because then it is the same computation twice.
        if missing:
            metrics[f"{key}_restricted"] = score_subset(
                model, backbone, chosen, labels[mask], device, stats,
                args.batch_size, restrict_to=allowed,
            )
            print_subset(f"{title}  [RESTRICTED to the dataset's classes]",
                         metrics[f"{key}_restricted"])

    if metrics["clean"] and metrics["all"]:
        gap = metrics["all"]["macro_f1"] - metrics["clean"]["macro_f1"]
        print(f"\ncontamination was worth {gap:+.4f} macro F1")

    if missing and metrics.get("clean") and metrics.get("clean_restricted"):
        open_f1 = metrics["clean"]["macro_f1"]
        restricted_f1 = metrics["clean_restricted"]["macro_f1"]
        print(
            f"clean macro F1: {open_f1:.4f} open, {restricted_f1:.4f} restricted "
            f"({restricted_f1 - open_f1:+.4f})"
        )
        print(
            "The gap is what the model wanted to say and these labels could not "
            "express. Quote both or neither."
        )

    print(
        "\nA filtered score is not a different-provenance score. These datasets "
        "are compiled from the same public archives, so removing shared images "
        "does not remove shared sources."
    )

    if args.out:
        # The report strings are for reading, not for a machine; drop them.
        serialisable = {
            key: {k: v for k, v in value.items() if k != "report"}
            for key, value in metrics.items()
            if value is not None
        }
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
