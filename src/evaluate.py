"""Score a trained checkpoint on a held-out split.

Overall accuracy is the least useful number here and it is printed last on
purpose. What matters for a screening model is per-class recall: of the images
that really were pneumonia, how many did it find? A model at 94% accuracy that
recalls only 60% of COVID-19 cases is worse than useless for the thing you
would actually want it for, and only the confusion matrix shows that.

    python -m src.evaluate --checkpoint checkpoints/best.pt --split test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.dataset import CLASSES, build_loader, class_counts
from src.model import load_checkpoint, pick_device


def collect_predictions(model, loader, device):
    """Returns (targets, predictions, probabilities) as numpy arrays."""
    all_targets = []
    all_predictions = []
    all_probabilities = []

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)

            all_targets.extend(targets.tolist())
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    return (
        np.array(all_targets),
        np.array(all_predictions),
        np.array(all_probabilities),
    )


def per_class_auc(targets, probabilities):
    """One-vs-rest ROC AUC per class, with None wherever it is undefined.

    Computed per class from binarised labels rather than through
    roc_auc_score(multi_class="ovr"), which requires every class to be present
    and raises for the whole split when one is not. A split missing a class is
    a real case here -- src.cross_dataset scores a dataset that has no lung
    opacity directory -- and losing the other three columns to it is not a
    trade worth making. The missing class gets None and the rest are reported.

    AUC is threshold-free: it asks whether the model *ranks* the positives
    above the negatives, not whether argmax lands on the right one. That makes
    it the metric least affected by the class imbalance, and also the one least
    connected to what the served model actually does, since serving takes an
    argmax. Read it beside the per-class recall, never instead of it.
    """
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    scores = []

    for index in range(len(CLASSES)):
        positive = targets == index
        # A column that is all positives or all negatives has no ROC curve --
        # one of the two rates is 0/0 at every threshold.
        if positive.all() or not positive.any():
            scores.append(None)
            continue
        scores.append(float(roc_auc_score(positive, probabilities[:, index])))

    return scores


def macro_average(values):
    """Mean over the classes that have a value, or None if none of them do.

    Macro rather than weighted, for the same reason selection is on macro F1:
    a weighted average is set by whichever class is largest, which here is
    NORMAL at roughly half the split.
    """
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def per_class_table(targets, predictions, probabilities):
    """The paper's Table 1 row 1, per class: precision, recall, F1, AUC, support."""
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=range(len(CLASSES)),
        zero_division=0,
    )
    auc = per_class_auc(targets, probabilities)

    return {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "auc": auc[index],
            "support": int(support[index]),
        }
        for index, name in enumerate(CLASSES)
    }


def print_auc(per_class):
    """AUC does not appear in classification_report, so it gets its own block."""
    print("\none-vs-rest ROC AUC")
    for name, row in per_class.items():
        value = "     n/a" if row["auc"] is None else f"{row['auc']:8.4f}"
        absent = "  (class absent from this split)" if row["auc"] is None else ""
        print(f"  {name:<14}{value}{absent}")


def print_confusion(matrix):
    """Rows are the truth, columns are what the model said.

    One width, derived from the longest class name, for the row labels and the
    columns alike. The column width used to be a hardcoded 12, which was fine
    until LUNG_OPACITY arrived at exactly 12 characters and filled its field
    edge to edge -- the header printed COVID19LUNG_OPACITY with the two names
    fused, and stayed that way for a whole release. Deriving it means the next
    class to be added cannot do the same thing.
    """
    width = max(len(name) for name in CLASSES) + 2
    header = " " * width + "".join(f"{name:>{width}}" for name in CLASSES)
    print("\nconfusion matrix (rows = actual, cols = predicted)")
    print(header)
    for name, row in zip(CLASSES, matrix):
        print(f"{name:<{width}}" + "".join(f"{value:>{width}}" for value in row))


def save_confusion_plot(matrix, path):
    import matplotlib

    # Agg so this works over SSH and in CI, where there is no display.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Row-normalised: raw counts make the largest class dominate the colouring
    # and hide the failure mode on the smallest one.
    normalised = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    figure, axes = plt.subplots(figsize=(5.5, 5))
    axes.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    axes.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    axes.set_yticks(range(len(CLASSES)), CLASSES)
    axes.set_xlabel("predicted")
    axes.set_ylabel("actual")

    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            axes.text(
                column,
                row,
                f"{matrix[row, column]}\n{normalised[row, column]:.0%}",
                ha="center",
                va="center",
                color="white" if normalised[row, column] > 0.5 else "black",
                fontsize=9,
            )

    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)

    loader, dataset = build_loader(
        Path(args.data_dir) / args.split,
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']}, {checkpoint['backbone']})")
    print(f"{args.split}: {len(dataset)} images {class_counts(dataset)}")

    targets, predictions, probabilities = collect_predictions(model, loader, device)

    print("\nper-class scores")
    print(
        classification_report(
            targets,
            predictions,
            labels=range(len(CLASSES)),
            target_names=CLASSES,
            digits=3,
            zero_division=0,
        )
    )

    per_class = per_class_table(targets, predictions, probabilities)
    print_auc(per_class)

    matrix = confusion_matrix(targets, predictions, labels=range(len(CLASSES)))
    print_confusion(matrix)

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    macro_auc = macro_average([row["auc"] for row in per_class.values()])
    print(f"\nmacro F1:  {macro_f1:.4f}")
    if macro_auc is not None:
        print(f"macro AUC: {macro_auc:.4f}")
    print(f"accuracy:  {accuracy:.4f}")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Keys are only ever added here, never renamed. Every number in the README
    # was read out of one of these files, and the older runs under reports/ are
    # not going to be regenerated.
    summary = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "images": len(dataset),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_auc": macro_auc,
        "macro_precision": macro_average(
            [row["precision"] for row in per_class.values()]
        ),
        "macro_recall": macro_average([row["recall"] for row in per_class.values()]),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "classes": CLASSES,
    }
    summary_path = report_dir / f"metrics_{args.split}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {summary_path}")

    # The metrics are the output of this script; the PNG is a convenience, and
    # the same matrix has already gone to stdout as text. Matplotlib is the one
    # import here that fails for reasons that have nothing to do with the model
    # -- no backend, a blocked DLL, an unwritable font cache -- and throwing
    # away a finished evaluation over the picture is not a trade worth making.
    # Hence: numbers first, plot second, and a warning rather than a traceback.
    plot_path = report_dir / f"confusion_{args.split}.png"
    try:
        save_confusion_plot(matrix, plot_path)
        print(f"saved {plot_path}")
    except Exception as error:
        print(f"\ncould not write {plot_path}: {error}")
        print("the confusion matrix above is the same data.")


if __name__ == "__main__":
    main()
