"""Do two chest X-ray collections share images?

Every number in this project comes from one dataset, whose classes are
separable by which repository produced them. The stated way out is to score
against a set from different hospitals -- but that only means anything if the
second set is actually different data. Compiled Kaggle collections routinely
re-package the same sources, and an image resized or re-encoded on the way in
has different bytes while being the same radiograph. Running a "cross-dataset"
evaluation against a repackage measures nothing and looks like it measured
something.

Two passes, because one is not enough:

    exact   sha256 over the file bytes. Catches straight copies only.
    visual  Pearson correlation between contrast-normalised 64x64 thumbnails.

**The visual pass is deliberately not a perceptual hash.** A 64-bit difference
hash is the standard tool here and it does not work on this material. Every
chest radiograph has the same silhouette and the same bright-spine-dark-lungs
layout, so at 8x8 they collapse onto each other. Measured against the COVID-19
Radiography Database: 89% of *independent* normal films came within hamming
distance 5 of some COVID film, and 5% collided exactly. A hash that cannot tell
two different chest X-rays apart cannot detect a shared one, and it fails in
the direction that kills a valid experiment -- reporting a fresh dataset as a
repackage.

Correlation over 4096 contrast-normalised pixels does separate them. The
thresholds below were read off two controls on the real dataset: 120 JPEG
re-encodes of COVID films (all scored >= 0.99, 119 of them above 0.999) against
120 untouched Normal films (none above 0.95). Nothing landed in between.

One warning for anyone recalibrating. Do **not** do it against `src.synth_data`
output. Those images are drawn from one template per class, so different draws
reach r = 0.9987 while true copies sit at 0.9994 -- no usable gap. They are
more self-similar than real radiographs, not less. The tests use a procedural
generator with randomised structure for exactly this reason.

    python -m src.dataset_overlap --left <dir> --right <dir>
"""

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

THUMBNAIL = 64

# Above DUPLICATE is the same radiograph through a re-encode. Below SUSPICIOUS
# is a different one. The band between them is where a human should look, and
# on the controls it was empty.
DUPLICATE = 0.99
SUSPICIOUS = 0.95

# Directory names that describe a split or a payload rather than a class, so
# class_of walks past them looking for something meaningful.
STRUCTURAL_DIRS = {"images", "masks", "train", "test", "val"}


def find_images(root, skip=("masks",)):
    """Every image under root, minus any path containing a skipped directory.

    masks/ is skipped by default because the Radiography Database ships a lung
    mask beside every image. They are not radiographs, they would dominate any
    correlation (all masks look alike), and counting them as overlap would be
    meaningless.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} does not exist.")

    skip = {term.lower() for term in skip}
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not skip & {part.lower() for part in path.parts}
    ]


def thumbnail_vector(image):
    """A contrast-normalised 64x64 thumbnail, flattened to a unit vector.

    Centring and scaling per image is what makes this compare structure rather
    than exposure: the same radiograph brightened, darkened or re-windowed
    still correlates at ~1.0, and that is precisely the case that has to be
    caught. The dot product of two of these is Pearson correlation directly.
    """
    small = image.convert("L").resize((THUMBNAIL, THUMBNAIL), Image.BILINEAR)
    pixels = np.asarray(small, dtype=np.float32).flatten()
    pixels -= pixels.mean()

    norm = float(np.linalg.norm(pixels))
    # A flat image has no structure to correlate. Left as zeros it matches
    # nothing, where dividing by ~0 would make it match everything.
    if norm < 1e-6:
        return pixels
    return pixels / norm


def fingerprint(paths, label=""):
    """Returns (byte digests, thumbnail matrix, the paths that could be read)."""
    digests, vectors, kept = [], [], []

    for index, path in enumerate(paths):
        if label and index and index % 4000 == 0:
            print(f"  {label}: {index}/{len(paths)}")
        try:
            raw = path.read_bytes()
            with Image.open(path) as image:
                vector = thumbnail_vector(image)
        except Exception as error:
            # One unreadable file in a 15,000-image download should not end the
            # comparison; it should be named and skipped.
            print(f"  skipped {path.name}: {error}")
            continue

        digests.append(hashlib.sha256(raw).hexdigest())
        vectors.append(vector)
        kept.append(path)

    matrix = (
        np.array(vectors, dtype=np.float32)
        if vectors
        else np.zeros((0, THUMBNAIL * THUMBNAIL), dtype=np.float32)
    )
    return digests, matrix, kept


def best_correlation(left, right, device=None, chunk=2048):
    """For each left thumbnail, (highest correlation, index of that match).

    Chunked over the left side because the full product is len(left) by
    len(right) floats -- 15,000 against 15,000 is nearly a gigabyte, and both
    real datasets are that size.
    """
    if len(left) == 0 or len(right) == 0:
        return (
            np.zeros(len(left), dtype=np.float32),
            np.zeros(len(left), dtype=np.int64),
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    right_tensor = torch.from_numpy(right).to(device)

    scores, indices = [], []
    for start in range(0, len(left), chunk):
        block = torch.from_numpy(left[start : start + chunk]).to(device)
        best, where = (block @ right_tensor.T).max(dim=1)
        scores.append(best.cpu().numpy())
        indices.append(where.cpu().numpy())

    return np.concatenate(scores), np.concatenate(indices)


def class_of(path, root):
    """The directory under root that most likely names a class."""
    relative = path.relative_to(root).parts
    for part in relative[:-1]:
        if part.lower() not in STRUCTURAL_DIRS:
            return part
    return relative[0] if len(relative) > 1 else "?"


def compare(left_root, right_root, skip=("masks",), device=None, quiet=False):
    """Fingerprint both trees and match every left image against the right."""
    left_root, right_root = Path(left_root), Path(right_root)
    label_left, label_right = ("", "") if quiet else ("left", "right")

    left_digests, left_vectors, left_paths = fingerprint(
        find_images(left_root, skip), label_left
    )
    right_digests, right_vectors, right_paths = fingerprint(
        find_images(right_root, skip), label_right
    )

    right_digest_set = set(right_digests)
    best, where = best_correlation(left_vectors, right_vectors, device)

    return {
        "left_root": left_root,
        "right_root": right_root,
        "left_paths": left_paths,
        "right_paths": right_paths,
        "exact": [digest in right_digest_set for digest in left_digests],
        "correlation": best,
        "match_index": where,
    }


def summarise(result):
    """Counts worth printing or asserting on."""
    correlation = result["correlation"]
    total = max(len(result["left_paths"]), 1)
    return {
        "images": len(result["left_paths"]),
        "exact": sum(result["exact"]),
        "duplicates": int((correlation >= DUPLICATE).sum()),
        "suspicious": int((correlation >= SUSPICIOUS).sum()),
        "duplicate_fraction": float((correlation >= DUPLICATE).sum() / total),
    }


def print_report(result):
    counts = summarise(result)
    total = max(counts["images"], 1)
    correlation = result["correlation"]

    print(f"\nleft  {result['left_root']}: {counts['images']} images")
    print(f"right {result['right_root']}: {len(result['right_paths'])} images\n")
    print(f"exact byte matches:  {counts['exact']:>7}/{counts['images']} "
          f"({counts['exact'] / total:.1%})")
    print(f"correlation >={DUPLICATE}: {counts['duplicates']:>7}/{counts['images']} "
          f"({counts['duplicate_fraction']:.1%})   the same radiograph")
    print(f"correlation >={SUSPICIOUS}: {counts['suspicious']:>7}/{counts['images']} "
          f"({counts['suspicious'] / total:.1%})   worth eyeballing")

    print("\ncorrelation histogram (each left image to its best match)")
    edges = [0.0, 0.80, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999, 1.01]
    binned = Counter(np.digitize(correlation, edges).tolist())
    for index in range(1, len(edges)):
        count = binned.get(index, 0)
        print(
            f"  {edges[index - 1]:.3f}-{edges[index]:.3f} {count:>7}  "
            + "#" * min(50, count * 50 // total)
        )

    print(f"\nby class on the left (correlation >= {DUPLICATE})")
    per_class = defaultdict(lambda: [0, 0])
    for path, value in zip(result["left_paths"], correlation):
        entry = per_class[class_of(path, result["left_root"])]
        entry[1] += 1
        if value >= DUPLICATE:
            entry[0] += 1
    for name, (matched, count) in sorted(per_class.items()):
        print(f"  {name:<22} {matched:>6}/{count:<6} {matched / max(count, 1):>7.1%}")

    if counts["duplicate_fraction"] > 0.05:
        print(
            f"\n{counts['duplicate_fraction']:.1%} of the left set is already in "
            f"the right one. A score measured across these two is not a "
            f"cross-dataset result."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Image overlap between two chest X-ray datasets."
    )
    parser.add_argument("--left", required=True, help="The dataset being checked.")
    parser.add_argument("--right", required=True, help="The one to check it against.")
    parser.add_argument("--skip", nargs="*", default=["masks"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    print_report(compare(args.left, args.right, args.skip, device))


if __name__ == "__main__":
    main()
