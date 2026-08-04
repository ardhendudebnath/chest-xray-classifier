"""Procedurally drawn fake X-rays, for testing the pipeline without a download.

The real datasets are several gigabytes behind a Kaggle login. That is a slow
first step for something whose only question is "does the code run end to end",
so this generates a small stand-in set with the same directory layout, the same
class imbalance, and a per-class signal a CNN can actually learn:

    NORMAL      clear lung fields
    PNEUMONIA   one bright consolidation in a single lung, lobar-looking
    COVID19     faint bilateral patches out at the lung periphery

Read this next part before quoting any number produced from these images.

These are ellipses and Gaussian blobs. They are not radiographs, they are not
derived from radiographs, and the pattern separating the three classes is one I
drew on purpose to be learnable. A model will reach high 90s on them within a
couple of epochs, and that result says only that gradients flow and the loaders
are wired up correctly -- it is a unit test with pictures. It is not evidence
about pneumonia, about COVID-19, or about anything a real classifier would face.
Any accuracy that goes in a report or a README has to come from real data.

    python -m src.synth_data --out data --per-class 120
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.dataset import CLASSES

IMG_SIZE = 256

# Roughly the imbalance the public sets have: pneumonia most common, COVID-19
# rarest. Kept because the sampler and the class weights are the parts of the
# training code most likely to be wrong, and a balanced set would not exercise
# them at all.
CLASS_SHARE = {"COVID19": 0.5, "NORMAL": 1.0, "PNEUMONIA": 1.6}


def _blob(yy, xx, cy, cx, ry, rx):
    """A soft Gaussian bump centred at (cy, cx), values in [0, 1]."""
    return np.exp(-(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2))


def _chest(yy, xx, rng):
    """The parts every image gets: thorax, two lung fields, spine, ribs."""
    size = yy.shape[0]
    canvas = np.full((size, size), 0.10, dtype=np.float32)

    # Soft tissue envelope.
    body = _blob(yy, xx, 0.52, 0.5, 0.40, 0.34)
    canvas += 0.55 * body

    # Lung fields are radiolucent, so they read darker than the tissue around
    # them. Two of them, offset either side of the midline.
    lung_y = 0.48 + rng.uniform(-0.02, 0.02)
    lung_ry = 0.24 + rng.uniform(-0.02, 0.02)
    lung_rx = 0.11 + rng.uniform(-0.01, 0.01)
    left = _blob(yy, xx, lung_y, 0.36, lung_ry, lung_rx)
    right = _blob(yy, xx, lung_y, 0.64, lung_ry, lung_rx)
    canvas -= 0.42 * (left + right)

    # Spine down the middle, and ribs as a few arcs across both fields.
    canvas += 0.22 * _blob(yy, xx, 0.5, 0.5, 0.44, 0.022)
    for offset in np.linspace(0.30, 0.68, 7):
        arc = np.exp(-(((yy - offset - 0.05 * (xx - 0.5) ** 2 * 4) / 0.012) ** 2))
        canvas += 0.09 * arc * body

    return canvas, (lung_y, lung_ry, lung_rx), left + right


def _render(class_name, rng):
    size = IMG_SIZE
    axis = np.linspace(0, 1, size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")

    canvas, (lung_y, lung_ry, lung_rx), lungs = _chest(yy, xx, rng)

    if class_name == "PNEUMONIA":
        # One dense opacity filling part of a single lung.
        side = 0.36 if rng.random() < 0.5 else 0.64
        canvas += 0.45 * _blob(
            yy,
            xx,
            lung_y + rng.uniform(-0.08, 0.08),
            side + rng.uniform(-0.02, 0.02),
            rng.uniform(0.07, 0.11),
            rng.uniform(0.05, 0.08),
        )

    elif class_name == "COVID19":
        # Several faint patches, both lungs, biased outward towards the
        # periphery rather than centred on the hilum.
        for side in (0.36, 0.64):
            outward = -1 if side < 0.5 else 1
            for _ in range(rng.integers(2, 5)):
                canvas += 0.20 * _blob(
                    yy,
                    xx,
                    lung_y + rng.uniform(-0.16, 0.16),
                    side + outward * rng.uniform(0.02, 0.09),
                    rng.uniform(0.03, 0.055),
                    rng.uniform(0.025, 0.045),
                )

    # Vignette, exposure drift and grain, so the model cannot separate classes
    # on global brightness alone.
    canvas *= 1.0 - 0.35 * _blob(yy, xx, 0.5, 0.5, 1.6, 1.6).max() + 0.35 * _blob(
        yy, xx, 0.5, 0.5, 0.75, 0.75
    )
    canvas *= rng.uniform(0.88, 1.12)
    canvas += rng.normal(0, 0.02, size=(size, size))

    array = np.clip(canvas, 0, 1)
    image = Image.fromarray((array * 255).astype(np.uint8), mode="L")

    # Small rotations, the way patient positioning varies between films.
    return image.rotate(rng.uniform(-6, 6), resample=Image.BILINEAR, fillcolor=25)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic X-rays for smoke-testing the pipeline."
    )
    parser.add_argument("--out", default="data")
    parser.add_argument(
        "--per-class",
        type=int,
        default=120,
        help="Training images for the middle class; the others scale off it.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)

    print("SYNTHETIC DATA -- drawn shapes, not radiographs.")
    print("Use it to check the pipeline runs. Never to report accuracy.\n")

    for class_name in CLASSES:
        train_count = max(1, round(args.per_class * CLASS_SHARE[class_name]))
        counts = {
            "train": train_count,
            "val": max(1, round(train_count * args.val_ratio)),
            "test": max(1, round(train_count * args.test_ratio)),
        }

        for split, count in counts.items():
            directory = out / split / class_name
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                _render(class_name, rng).save(directory / f"{split}_{index:04d}.png")

        print(f"{class_name:<10} {counts}")

    print(f"\nwrote {out.resolve()}")
    print("next: python -m src.train --epochs 5 --freeze-backbone")


if __name__ == "__main__":
    main()
