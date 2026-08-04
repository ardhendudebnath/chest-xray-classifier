"""Fine-tune the classifier and keep the checkpoint that generalises best.

Model selection is on macro F1, not accuracy. With the imbalance these datasets
carry, a model that never predicts the rarest class can still post a high
accuracy, and selecting on accuracy would faithfully preserve that model. Macro
F1 averages the per-class scores equally, so ignoring a class is expensive.

Typical first run:

    python -m src.train --epochs 15 --freeze-backbone
    python -m src.train --epochs 25 --lr 1e-4   # then unfreeze and go again
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.dataset import CLASSES, build_loader, class_counts, class_weights
from src.model import (
    DEFAULT_BACKBONE,
    SUPPORTED_BACKBONES,
    build_model,
    pick_device,
    save_checkpoint,
)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over a loader. Passing an optimizer makes it a training pass."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    count = len(all_targets)
    accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / count
    macro_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)

    return {
        "loss": total_loss / count,
        "accuracy": accuracy,
        "macro_f1": float(macro_f1),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the chest X-ray classifier.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=SUPPORTED_BACKBONES)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", default="checkpoints/best.pt")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the new head. Start here when data is scarce.",
    )
    parser.add_argument(
        "--imbalance",
        choices=("sampler", "loss", "none"),
        default="sampler",
        help="How to handle unequal class sizes. Pick one, not both.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop after this many epochs with no val macro-F1 improvement.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    data_dir = Path(args.data_dir)

    train_loader, train_dataset = build_loader(
        data_dir / "train",
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance=(args.imbalance == "sampler"),
    )
    val_loader, val_dataset = build_loader(
        data_dir / "val",
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"device: {device}  backbone: {args.backbone}")
    print(f"train: {len(train_dataset)} images {class_counts(train_dataset)}")
    print(f"val:   {len(val_dataset)} images {class_counts(val_dataset)}")

    model = build_model(
        args.backbone, pretrained=True, freeze_backbone=args.freeze_backbone
    ).to(device)

    weights = None
    if args.imbalance == "loss":
        weights = class_weights(train_dataset).to(device)
        print(f"loss weights: {dict(zip(CLASSES, weights.tolist()))}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_f1 = -1.0
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"epoch {epoch:>3}/{args.epochs}  "
            f"train loss {train_metrics['loss']:.4f} f1 {train_metrics['macro_f1']:.4f}  |  "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
            f"f1 {val_metrics['macro_f1']:.4f}  ({time.time() - started:.0f}s)"
        )

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            epochs_since_best = 0
            save_checkpoint(args.out, model, args.backbone, epoch, val_metrics)
            print(f"  saved {args.out} (val macro F1 {best_f1:.4f})")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"no improvement in {args.patience} epochs, stopping early")
                break

    history_path = Path(args.out).with_name("history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nbest val macro F1: {best_f1:.4f}")
    print(f"history -> {history_path}")
    print(f"next: python -m src.evaluate --checkpoint {args.out}")


if __name__ == "__main__":
    main()
