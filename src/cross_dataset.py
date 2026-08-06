"""Score a checkpoint on a second dataset, with its own training images removed.

The whole project rests on numbers from one dataset whose classes are separable
by provenance, so the obvious next move is to score against a different
download. Done naively that measures nothing: the public chest X-ray sets
re-package each other, and `src.dataset_overlap` found that 23.9% of
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

    python -m src.cross_dataset --dataset ~/Downloads/chest-xray-cp/Data \\
        --exclude-against ~/cxr-data-real/train

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
    which is how directories outside the three classes get dropped.
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


def predict(model, backbone, paths, device, stats=None, batch_size=64):
    """Returns (predictions, out-of-distribution flags) for a list of images.

    The flags are all False when no statistics are given. That is reported as
    "not checked" by the caller rather than folded into the numbers, since a
    detector that never fires and one that was never run look identical here.
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

        predictions.append(logits.argmax(dim=1).cpu().numpy())
        if stats is not None:
            flags.append(is_out_of_distribution(*score(features.cpu().numpy(), stats), stats))

    if not predictions:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)

    return (
        np.concatenate(predictions),
        np.concatenate(flags) if flags else np.zeros(len(paths), dtype=bool),
    )


def score_subset(model, backbone, paths, labels, device, stats=None, batch_size=64):
    """Metrics for one population. Returns a dict, or None when it is empty."""
    if len(paths) == 0:
        return None

    predictions, flags = predict(model, backbone, paths, device, stats, batch_size)
    matrix = confusion_matrix(labels, predictions, labels=range(len(CLASSES)))

    result = {
        "images": len(paths),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "per_class_f1": {
            name: float(value)
            for name, value in zip(
                CLASSES,
                f1_score(labels, predictions, average=None, labels=range(len(CLASSES)), zero_division=0),
            )
        },
        "report": classification_report(
            labels, predictions, labels=range(len(CLASSES)),
            target_names=CLASSES, digits=3, zero_division=0,
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

    print("\nconfusion (rows actual, cols predicted)")
    print(" " * 12 + "".join(f"{name:>12}" for name in CLASSES))
    for name, row in zip(CLASSES, result["confusion_matrix"]):
        print(f"{name:<12}" + "".join(f"{value:>12}" for value in row))

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

    metrics = {}
    for key, (mask, title) in populations.items():
        chosen = [paths[i] for i in np.flatnonzero(mask)]
        metrics[key] = score_subset(
            model, backbone, chosen, labels[mask], device, stats, args.batch_size
        )
        print_subset(title, metrics[key])

    if metrics["clean"] and metrics["all"]:
        gap = metrics["all"]["macro_f1"] - metrics["clean"]["macro_f1"]
        print(f"\ncontamination was worth {gap:+.4f} macro F1")

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
