"""Turn a downloaded public dataset into the train/val/test layout.

The public chest X-ray collections all arrive in different shapes -- some
already split, some one flat directory per class, some with the images buried
another level down -- and none of them use the class names this project does.
This script normalises whichever one you downloaded into:

    data/train/{COVID19,NORMAL,PNEUMONIA}/
    data/val/...
    data/test/...

Two things it does that a naive split does not:

**Patient grouping.** The Kermany pneumonia images are named person1_virus_6,
person1_bacteria_1, and so on -- several X-rays of the same patient. Split those
at random and the same chest lands in train and in test, so the test score
measures memorisation and comes out 5-10 points too high. Groups are kept whole
in one split. Where a filename carries no patient id, each image is its own
group, which is the correct assumption for the COVID-19 Radiography Database.

**Class aliasing.** "COVID", "COVID-19" and "COVID19" are the same class;
"Viral Pneumonia" and "PNEUMONIA" are the same class. Lung_Opacity is mapped to
LUNG_OPACITY and is *not* folded into pneumonia -- it is a broader finding, and
merging them would change what the model claims to detect.

Not every download has all four. The Radiography Database does; the smaller
Kermany-derived sets have no lung opacity directory at all, so they cannot
train this model and the run stops rather than quietly producing a split with
one class missing -- which ImageFolder would renumber the rest around.

    python -m src.prepare_data --source ~/Downloads/COVID-19_Radiography_Dataset
"""

import argparse
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from src.dataset import CLASSES

# Lowercased directory names that map onto each of our four classes. Matching
# is done on the whole directory name, not a substring, so "Non-COVID" cannot
# fall through into COVID19.
CLASS_ALIASES = {
    "COVID19": {"covid", "covid19", "covid-19", "covid_19", "covid19_cases"},
    "LUNG_OPACITY": {"lung_opacity", "lung opacity", "lung-opacity", "opacity"},
    "NORMAL": {"normal", "normal_cases", "healthy"},
    "PNEUMONIA": {
        "pneumonia",
        "viral pneumonia",
        "viral_pneumonia",
        "pneumonia_cases",
        "bacterial pneumonia",
        "bacterial_pneumonia",
    },
}

# Present in some downloads and deliberately not mapped. Named here so the
# script can say "skipping tuberculosis" instead of "found nothing".
KNOWN_IGNORED = {"masks", "tuberculosis", "lung_masks"}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# person1_virus_6.jpeg and person1_bacteria_1.jpeg are the same patient.
PATIENT_PATTERN = re.compile(r"(person\d+)", re.IGNORECASE)


def resolve_class(directory_name):
    """Our class name for a source directory, or None if it is not one of ours."""
    key = directory_name.strip().lower()
    for canonical, aliases in CLASS_ALIASES.items():
        if key in aliases:
            return canonical
    return None


def find_class_dirs(source):
    """Map canonical class -> list of source directories holding its images.

    Walks the whole tree because the layouts differ: the Radiography Database
    puts images in COVID/images/, and already-split downloads repeat each class
    under train/ and test/. Both are collected and re-split from scratch --
    the published splits are not patient-grouped either.
    """
    found = defaultdict(list)
    ignored = set()

    for directory in [source, *source.rglob("*")]:
        if not directory.is_dir():
            continue

        name = directory.name
        # "images" is a container, not a class; the class name is its parent.
        if name.lower() == "images" and directory.parent != source:
            name = directory.parent.name

        canonical = resolve_class(name)
        if canonical is None:
            if directory.name.lower() in KNOWN_IGNORED:
                ignored.add(directory.name)
            continue

        if any(child.suffix.lower() in IMAGE_SUFFIXES for child in directory.iterdir()):
            found[canonical].append(directory)

    return found, ignored


def patient_group(path):
    """A key that keeps one patient's images together.

    Falls back to the filename, which makes the image its own group -- correct
    when the dataset has one image per patient, and no worse than a random
    split when it does not.
    """
    match = PATIENT_PATTERN.search(path.name)
    return match.group(1).lower() if match else path.name.lower()


def collect_images(directories):
    """Group -> list of image paths, across every source directory for a class."""
    groups = defaultdict(list)
    for directory in directories:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                groups[patient_group(path)].append(path)
    return groups


def split_groups(groups, ratios, rng):
    """Assign whole groups to train/val/test, aiming at the image-count ratios.

    Greedy rather than exact: groups are shuffled, then each goes to whichever
    split is furthest below its target share. With mostly single-image groups
    this lands within a few images of the requested ratios; with large groups
    it stays balanced where an index-slice split would not.
    """
    names = list(groups.keys())
    rng.shuffle(names)

    total = sum(len(paths) for paths in groups.values())
    targets = {split: ratio * total for split, ratio in ratios.items()}
    assigned = {split: [] for split in ratios}
    counts = {split: 0 for split in ratios}

    # Largest groups first, so the big ones are placed while there is still
    # room to compensate for them.
    for name in sorted(names, key=lambda n: -len(groups[n])):
        deficit = {
            split: targets[split] - counts[split]
            for split in ratios
            if targets[split] > 0
        }
        pick = max(deficit, key=deficit.get)
        assigned[pick].extend(groups[name])
        counts[pick] += len(groups[name])

    return assigned


def place(source_path, destination_path, link):
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(source_path, destination_path)
            return
        except OSError:
            # Different volume, or a filesystem without hardlinks. Copying is
            # always correct, just slower.
            pass
    shutil.copy2(source_path, destination_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalise a downloaded chest X-ray dataset into data/."
    )
    parser.add_argument(
        "--source", required=True, help="Directory you unzipped the download into."
    )
    parser.add_argument("--out", default="data")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Use at most this many images per class. Good for a fast first run.",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Hardlink instead of copying. Saves several GB; same volume only.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing --out directory."
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    source = Path(args.source).expanduser()
    if not source.is_dir():
        raise SystemExit(f"--source {source} is not a directory")

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {total_ratio}")

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(
                f"{out} already has files in it. Pass --force to replace it."
            )
        # Only the split directories, so a stray README or .gitkeep survives.
        for split in ratios:
            shutil.rmtree(out / split, ignore_errors=True)

    class_dirs, ignored = find_class_dirs(source)
    if ignored:
        print(f"skipping non-target directories: {', '.join(sorted(ignored))}")

    missing = [name for name in CLASSES if name not in class_dirs]
    if missing:
        raise SystemExit(
            f"found no images for {missing} under {source}.\n"
            f"Directories matched: "
            f"{ {k: [str(d) for d in v] for k, v in class_dirs.items()} }\n"
            "Check you unzipped the whole download, and see CLASS_ALIASES in "
            "this file if your copy uses different directory names.\n"
            "Not every public set has all four classes -- the Kermany-derived "
            "ones carry no lung opacity images, and cannot train this model."
        )

    rng = random.Random(args.seed)
    tally = {split: {} for split in ratios}

    for class_name in CLASSES:
        directories = class_dirs[class_name]
        groups = collect_images(directories)

        if args.limit_per_class:
            # Trim whole groups, never half a patient.
            kept = {}
            running = 0
            for name in sorted(groups, key=lambda n: rng.random()):
                if running >= args.limit_per_class:
                    break
                kept[name] = groups[name]
                running += len(groups[name])
            groups = kept

        image_count = sum(len(paths) for paths in groups.values())
        print(
            f"{class_name:<10} {image_count:>6} images in {len(groups)} groups "
            f"from {len(directories)} director{'y' if len(directories) == 1 else 'ies'}"
        )

        for split, paths in split_groups(groups, ratios, rng).items():
            destination_dir = out / split / class_name
            for index, path in enumerate(paths):
                # Prefix with the index to survive identical filenames coming
                # from two different source directories.
                place(path, destination_dir / f"{index:05d}_{path.name}", args.link)
            tally[split][class_name] = len(paths)

    print()
    for split in ("train", "val", "test"):
        counts = tally[split]
        print(f"{split:<6} {sum(counts.values()):>6}  {counts}")

    print(f"\nwrote {out.resolve()}")
    print("next: python -m src.train --epochs 15 --freeze-backbone")


if __name__ == "__main__":
    main()
