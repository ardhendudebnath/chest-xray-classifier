"""Mirror a prepared split with everything outside the lung fields blacked out.

This exists to answer one question: how much of a test score is the model
reading the lungs, and how much is it reading the picture around them?

The question is not academic for this dataset. Every COVID-19 image in the
COVID-19 Radiography Database comes from BIMCV, Eurorad, SIRM or a GitHub
collection, and every Normal and Viral Pneumonia image comes from Kaggle --
zero overlap. The classes are therefore perfectly separable by provenance
before any lung is examined, and scanner, exposure, collimation, burned-in
markers and background all carry that provenance.

Masking removes the easy half of it. Train on the masked mirror, compare
against the unmasked run, and the difference is the part of the score that
lived outside the lungs.

    python -m src.mask_lungs --split-root ../cxr-data \\
        --source ~/Downloads/COVID-19_Radiography_Dataset --out ../cxr-data-masked

Mirrors an existing split rather than re-splitting, so the two runs differ in
exactly one variable.

**Read this before treating a surviving score as reassurance.** Two confounds
outlive masking:

- The lung *silhouette*. A child's lungs are a different shape and size from an
  adult's, and Viral Pneumonia here is the paediatric Kermany collection while
  COVID-19 is adult European patients. Outline alone separates them.
- In-lung acquisition signature. A scanner's noise and processing character is
  present in lung pixels too, not only in the margins.

So a score that collapses under masking is strong evidence of a shortcut. A
score that holds is not evidence against one. Only validation on a dataset
where provenance does not correlate with label can settle that.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.dataset import CLASSES
from src.prepare_data import IMAGE_SUFFIXES, resolve_class

SPLITS = ("train", "val", "test")


def find_mask_dirs(source):
    """Our class name -> the masks/ directory belonging to it.

    Resolved through prepare_data's aliases rather than a second hardcoded
    mapping, so "COVID" and "Viral Pneumonia" are understood here for the same
    reason they are understood there.
    """
    found = {}
    for directory in source.rglob("masks"):
        if not directory.is_dir():
            continue
        canonical = resolve_class(directory.parent.name)
        if canonical is not None:
            found[canonical] = directory
    return found


def original_name(prepared_name):
    """prepare_data writes '00042_COVID-1447.png'; its mask is 'COVID-1447.png'.

    The index prefix exists to stop identical filenames from two source
    directories colliding. It has to come back off to find the mask.
    """
    prefix, _, rest = prepared_name.partition("_")
    return rest if prefix.isdigit() and rest else prepared_name


def apply_mask(image_path, mask_path):
    """The image with every non-lung pixel set to 0.

    Nearest-neighbour on the mask because it is binary -- the masks ship at
    256x256 against 299x299 images, and interpolating would fringe the border
    with values that are neither lung nor background.
    """
    image = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L").resize(image.size, Image.NEAREST)

    array = np.array(image)
    array[np.array(mask) <= 127] = 0
    return Image.fromarray(array, mode="L")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Black out everything outside the lungs in a prepared split."
    )
    parser.add_argument(
        "--split-root", required=True, help="The prepared data/ directory to mirror."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="The original download, which is where the masks live.",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    split_root = Path(args.split_root).expanduser()
    source = Path(args.source).expanduser()
    out = Path(args.out).expanduser()

    if not split_root.is_dir():
        raise SystemExit(f"--split-root {split_root} is not a directory")
    if not source.is_dir():
        raise SystemExit(f"--source {source} is not a directory")

    mask_dirs = find_mask_dirs(source)
    absent = [name for name in CLASSES if name not in mask_dirs]
    if absent:
        raise SystemExit(
            f"found no masks/ directory for {absent} under {source}. "
            "Only some downloads ship segmentation masks; the COVID-19 "
            "Radiography Database does."
        )

    missing = []
    tally = {split: {} for split in SPLITS}

    for split in SPLITS:
        for class_name in CLASSES:
            in_dir = split_root / split / class_name
            if not in_dir.is_dir():
                raise SystemExit(f"{in_dir} does not exist. Run src.prepare_data first.")

            out_dir = out / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            written = 0
            for path in sorted(in_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue

                mask_path = mask_dirs[class_name] / original_name(path.name)
                if not mask_path.is_file():
                    missing.append(path.name)
                    continue

                apply_mask(path, mask_path).save(out_dir / path.name)
                written += 1

            tally[split][class_name] = written
        print(f"{split:<6} {sum(tally[split].values()):>6}  {tally[split]}", flush=True)

    # Dropping images would silently change the split, which is the one thing
    # this script exists to hold constant.
    if missing:
        raise SystemExit(
            f"\n{len(missing)} images had no matching mask, e.g. {missing[:3]}. "
            "The mirror would not match the split it is being compared against."
        )

    print(f"\nwrote {out.resolve()}")
    print(f"next: python -m src.train --epochs 15 --freeze-backbone --data-dir {out}")


if __name__ == "__main__":
    main()
