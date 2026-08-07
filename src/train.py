"""Fine-tune the classifier and keep the checkpoint that generalises best.

Model selection is on macro F1, not accuracy. With the imbalance these datasets
carry, a model that never predicts the rarest class can still post a high
accuracy, and selecting on accuracy would faithfully preserve that model. Macro
F1 averages the per-class scores equally, so ignoring a class is expensive.

Typical first run -- converge the head, then unfreeze and continue from it:

    python -m src.train --epochs 15 --freeze-backbone --out checkpoints/stage1.pt
    python -m src.train --epochs 25 --lr 1e-4 --resume checkpoints/stage1.pt

The --resume is what makes that two stages rather than two unrelated runs.
Without it the second command starts over from ImageNet weights and the first
command's epochs are discarded.
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
    freeze_to_head,
    load_checkpoint,
    pick_device,
    save_checkpoint,
)


def start_model(args, device):
    """The weights this run begins from, and the backbone they belong to.

    Without --resume that is an ImageNet-pretrained backbone under a fresh
    head. With it, a previous run's weights.

    Weights only, deliberately. The checkpoint carries no optimizer state, and
    restoring it would be wrong here even if it did: AdamW moments accumulated
    while the backbone was frozen do not describe the parameters the second
    stage unfreezes, and the cosine schedule belongs to the epoch budget of the
    run that declared it. So the optimizer and the schedule start clean, which
    is what you want when the whole point of the second stage is a different
    --lr over a different set of trainable parameters.
    """
    if not args.resume:
        backbone = args.backbone or DEFAULT_BACKBONE
        model = build_model(
            backbone, pretrained=True, freeze_backbone=args.freeze_backbone
        )
        return model, backbone

    resume_path = Path(args.resume)
    if not resume_path.is_file():
        raise SystemExit(f"--resume {resume_path} does not exist.")

    # Loading a checkpoint and writing over it in the same run costs you the
    # only copy of the stage you were building on, and the default --out makes
    # that the easy mistake rather than the unlikely one.
    if resume_path.resolve() == Path(args.out).resolve():
        raise SystemExit(
            f"--resume and --out are both {resume_path}. This run would "
            f"overwrite the checkpoint it started from. Pass a different --out."
        )

    # load_checkpoint rebuilds from the stored backbone name and rejects a
    # checkpoint whose class list has drifted, which would otherwise relabel
    # every prediction without raising anything.
    model, checkpoint = load_checkpoint(resume_path, device)
    backbone = checkpoint["backbone"]

    if args.backbone and args.backbone != backbone:
        raise SystemExit(
            f"--backbone {args.backbone}, but {resume_path} holds a {backbone}. "
            f"Those weights only fit the architecture they were trained on."
        )

    if args.freeze_backbone:
        freeze_to_head(model, backbone)

    saved_f1 = checkpoint.get("metrics", {}).get("macro_f1")
    scored = f", val macro F1 {saved_f1:.4f}" if saved_f1 is not None else ""
    print(f"resuming from {resume_path} (epoch {checkpoint['epoch']}{scored})")

    return model, backbone


def history_path_for(out):
    """Where this run's per-epoch history goes, named after its checkpoint.

        checkpoints/best.pt         -> checkpoints/best_history.json
        checkpoints/masked_best.pt  -> checkpoints/masked_best_history.json

    Derived from --out rather than fixed, because a fixed name means every run
    into one directory overwrites the last one's curve. That fails at the very
    end, silently, once the epochs it recorded are already spent -- a masked
    run destroyed the unmasked four-class model's 25-epoch stage 2 exactly this
    way, and checkpoints/ is gitignored, so there was nothing to recover from.

    The two-stage recipe is two runs into one directory by design, so this is
    the ordinary case rather than the careless one.
    """
    out = Path(out)
    return out.with_name(f"{out.stem}_history.json")


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
    parser.add_argument(
        "--backbone",
        default=None,
        choices=SUPPORTED_BACKBONES,
        help=f"Default {DEFAULT_BACKBONE}, or the checkpoint's under --resume.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Start from this checkpoint's weights instead of ImageNet. Weights "
            "only: the optimizer and the LR schedule start clean, which is what "
            "the unfreezing second stage wants."
        ),
    )
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

    model, backbone = start_model(args, device)
    model.to(device)

    print(f"device: {device}  backbone: {backbone}")
    print(f"train: {len(train_dataset)} images {class_counts(train_dataset)}")
    print(f"val:   {len(val_dataset)} images {class_counts(val_dataset)}")

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
    # Not seeded from a resumed checkpoint's score even when there is one. Doing
    # that would let a second stage finish having written no checkpoint at all
    # if it never beat the first, which reads as a silent failure. --out is the
    # best of this run; compare it against the one you resumed from yourself.
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
            save_checkpoint(args.out, model, backbone, epoch, val_metrics)
            print(f"  saved {args.out} (val macro F1 {best_f1:.4f})")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"no improvement in {args.patience} epochs, stopping early")
                break

    history_path = history_path_for(args.out)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nbest val macro F1: {best_f1:.4f}")
    print(f"history -> {history_path}")
    print(f"next: python -m src.evaluate --checkpoint {args.out}")


if __name__ == "__main__":
    main()
