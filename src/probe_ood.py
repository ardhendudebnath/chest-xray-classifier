"""Hand the model six things that are not chest X-rays and see what comes back.

This exists because the README makes two claims that are easy to state and easy
to let rot: that `low_confidence` does not catch an off-distribution input, and
that the Mahalanobis check does. Both are claims about specific numbers against
a specific checkpoint, and both were originally measured with throwaway images
that were not kept -- so when the class list changed from three to four, there
was no way to re-run them and no way to tell whether the table had gone stale.

The inputs are generated here rather than stored, so the table can be regenerated
against any checkpoint:

    flat grey / black / white   no structure at all, three different exposures
    uniform noise               maximum structure, none of it anatomical
    smooth colour image         low-frequency colour content, no edges
    a page of text              high-frequency structure in the wrong places

Read the two columns as answering different questions. `low_confidence` is the
softmax being visibly torn between classes, and it fires on none of these --
that is the point, not a bug in the threshold. A softmax over a fixed class list
normalises whatever it is handed, so "none of the above" is not in its range at
all. `ood_score` is distance from the training features in the layer below, and
it is the column that separates them.

**These six are the easy case.** Every one of them is obviously not a
radiograph, and a check that failed here would be worthless. Passing here does
not establish that the check catches a real chest X-ray showing a finding the
model has no class for -- it does not, and `src/ood.py` says so at more length.

    python -m src.probe_ood

"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.dataset import CLASSES, build_transforms
from src.model import forward_with_features, load_checkpoint, pick_device
from src.ood import DEFAULT_STATS_PATH, is_out_of_distribution, load_stats, score

# Larger than the 224 the transforms resize to, so the resize is doing the same
# work it does for a real upload rather than an upscale.
SIZE = 512

# Fixed, because a table in the README should not move between runs.
SEED = 0

# The API's cutoff, from app/main.py. Duplicated rather than imported because
# importing the app pulls in FastAPI and a checkpoint load at module scope.
LOW_CONFIDENCE_BELOW = 0.6


def _flat(value):
    return Image.new("L", (SIZE, SIZE), value)


def _uniform_noise():
    rng = np.random.default_rng(SEED)
    return Image.fromarray(rng.integers(0, 256, (SIZE, SIZE), dtype=np.uint8), "L")


def _colour_image():
    """Smooth low-frequency colour content -- no edges, no texture.

    Stands in for an ordinary photograph. The greyscale conversion in the
    serving path flattens it to a soft gradient, which is roughly what a
    snapshot of a wall or a sky becomes by the time the model sees it.
    """
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] / SIZE
    red = 0.5 + 0.4 * np.sin(2.1 * np.pi * xx + 0.4)
    green = 0.5 + 0.4 * np.sin(1.7 * np.pi * yy + 1.1)
    blue = 0.5 + 0.35 * np.exp(-((xx - 0.6) ** 2 + (yy - 0.35) ** 2) / 0.09)
    stack = np.clip(np.dstack([red, green, blue]), 0, 1)
    return Image.fromarray((stack * 255).astype(np.uint8), "RGB")


def _page_of_text():
    """Dark text on a light page: the tonal inverse of a radiograph.

    PIL's default bitmap font, so this needs no font file on the machine and
    renders identically everywhere.
    """
    image = Image.new("L", (SIZE, SIZE), 245)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    rng = np.random.default_rng(SEED)
    words = [
        "the", "model", "returns", "a", "number", "which", "is", "not",
        "a", "finding", "and", "should", "not", "be", "read", "as", "one",
    ]
    for row in range(28):
        line = " ".join(rng.choice(words) for _ in range(12))
        draw.text((24, 20 + row * 17), line, fill=30, font=font)
    return image


PROBES = [
    ("flat grey square", lambda: _flat(128)),
    ("flat black square", lambda: _flat(0)),
    ("flat white square", lambda: _flat(255)),
    ("uniform noise", _uniform_noise),
    ("smooth colour image", _colour_image),
    ("a page of text", _page_of_text),
]


def probe(model, backbone, stats, device, image):
    """One image through the serving path. Returns the row, not a print."""
    # Exactly what app.main does to an upload: greyscale, then the eval
    # transforms. Any divergence here would make the table describe a
    # preprocessing pipeline that nothing actually serves.
    tensor = build_transforms(train=False)(image.convert("L"))
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, features = forward_with_features(model, backbone, tensor)

    probabilities = torch.softmax(logits, dim=1)[0]
    index = int(probabilities.argmax())
    confidence = float(probabilities[index])

    scores, nearest = score(features.cpu().numpy(), stats)
    return {
        "prediction": CLASSES[index],
        "confidence": confidence,
        "low_confidence": confidence < LOW_CONFIDENCE_BELOW,
        "ood_score": float(scores[0]),
        "nearest": CLASSES[int(nearest[0])],
        "ood_threshold": float(stats["thresholds"][int(nearest[0])]),
        "out_of_distribution": bool(is_out_of_distribution(scores, nearest, stats)[0]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Score six not-a-chest-X-ray inputs against a checkpoint."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--stats", default=DEFAULT_STATS_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Also write the generated inputs here, to look at them.",
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    backbone = checkpoint["backbone"]
    # load_stats refuses statistics fitted against a different checkpoint, which
    # is the failure this table would otherwise report as a result.
    stats = load_stats(args.stats, model)

    print(f"checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']}, {backbone})")
    print(f"stats: {args.stats}, cutoffs {[round(t, 1) for t in stats['thresholds']]}")
    print(
        f"\n{'input':<22} {'prediction':<20} {'low_conf':<9} "
        f"{'ood_score':>10} {'cutoff':>9}  refused"
    )

    refused_all = True
    for name, make in PROBES:
        image = make()
        if args.save_dir:
            directory = Path(args.save_dir)
            directory.mkdir(parents=True, exist_ok=True)
            image.save(directory / f"{name.replace(' ', '_')}.png")

        row = probe(model, backbone, stats, device, image)
        refused_all &= row["out_of_distribution"]
        prediction = f"{row['prediction']} {row['confidence']:.2%}"
        print(
            f"{name:<22} {prediction:<20} "
            f"{('yes' if row['low_confidence'] else 'no'):<9} "
            f"{row['ood_score']:>10.1f} {row['ood_threshold']:>9.1f}  "
            f"{'yes' if row['out_of_distribution'] else 'NO'}"
        )

    if not refused_all:
        print(
            "\nAt least one of these was accepted as in-distribution. That is a "
            "regression in the check, not a curiosity -- refit with src.ood and "
            "look at --percentile before trusting any out_of_distribution field."
        )


if __name__ == "__main__":
    main()
