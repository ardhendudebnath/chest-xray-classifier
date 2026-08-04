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


def print_confusion(matrix):
    """Rows are the truth, columns are what the model said."""
    width = max(len(name) for name in CLASSES) + 2
    header = " " * width + "".join(f"{name:>12}" for name in CLASSES)
    print("\nconfusion matrix (rows = actual, cols = predicted)")
    print(header)
    for name, row in zip(CLASSES, matrix):
        print(f"{name:<{width}}" + "".join(f"{value:>12}" for value in row))


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

    targets, predictions, _ = collect_predictions(model, loader, device)

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

    matrix = confusion_matrix(targets, predictions, labels=range(len(CLASSES)))
    print_confusion(matrix)

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    print(f"\nmacro F1: {macro_f1:.4f}")
    print(f"accuracy: {accuracy:.4f}")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "images": len(dataset),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
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
