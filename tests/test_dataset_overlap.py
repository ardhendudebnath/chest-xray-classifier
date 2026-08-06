"""Overlap detection, and the two controls the thresholds were read off.

The controls matter more than usual here. The first implementation of this used
a 64-bit difference hash, passed every "does it find a copy" test, and was
still wrong -- it called 89% of independent chest films near-duplicates,
because at 8x8 every radiograph looks like every other one. Only a negative
control catches that, so there is one below and it is not optional.

The images are procedural rather than from src.synth_data. Those are drawn from
one template per class and are more self-similar than real radiographs, which
makes them useless as a negative control; test_synthetic_renders_are_not_a_
valid_control pins that down so nobody rebuilds the calibration on them.
"""

import io

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from src.dataset_overlap import (
    DUPLICATE,
    SUSPICIOUS,
    best_correlation,
    class_of,
    compare,
    find_images,
    summarise,
    thumbnail_vector,
)

CPU = torch.device("cpu")


def distinct_image(rng, size=224):
    """An X-ray-ish image whose structure is randomised, not templated.

    Randomised position, size, count and intensity, so two draws are genuinely
    different pictures rather than one picture with noise on it.
    """
    image = Image.new("L", (size, size), int(rng.integers(10, 60)))
    draw = ImageDraw.Draw(image)
    for _ in range(int(rng.integers(3, 7))):
        x, y = rng.integers(0, size, 2)
        rx, ry = rng.integers(20, 90, 2)
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=int(rng.integers(70, 250)))
    return image


def reencoded(image, size=180, quality=55):
    """The same picture after a resize and a lossy round trip, as a repackaged
    dataset would carry it."""
    buffer = io.BytesIO()
    image.convert("L").resize((size, size), Image.BILINEAR).save(
        buffer, format="JPEG", quality=quality
    )
    return Image.open(io.BytesIO(buffer.getvalue()))


def write_images(directory, images, suffix="png"):
    directory.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        image.convert("L").save(directory / f"{index:03d}.{suffix}")
    return directory


def vectors(images):
    return np.array([thumbnail_vector(i) for i in images], dtype=np.float32)


# -------------------------------------------------------------- the controls


def test_positive_control_reencoded_copies_are_found(tmp_path):
    """The case byte hashing misses entirely: same radiograph, different file."""
    rng = np.random.default_rng(0)
    originals = [distinct_image(rng) for _ in range(30)]

    write_images(tmp_path / "right" / "COVID19", originals)
    write_images(tmp_path / "left" / "COVID19", [reencoded(i) for i in originals], "jpg")

    counts = summarise(compare(tmp_path / "left", tmp_path / "right", device=CPU, quiet=True))

    assert counts["images"] == 30
    assert counts["duplicates"] == 30
    # And the bytes differ, so a checksum comparison would have found nothing.
    assert counts["exact"] == 0


def test_negative_control_independent_images_are_not_flagged(tmp_path):
    """The test that caught the difference-hash implementation being wrong."""
    write_images(
        tmp_path / "left" / "NORMAL",
        [distinct_image(np.random.default_rng(1)) for _ in range(30)],
    )
    write_images(
        tmp_path / "right" / "COVID19",
        [distinct_image(np.random.default_rng(2)) for _ in range(30)],
    )

    counts = summarise(compare(tmp_path / "left", tmp_path / "right", device=CPU, quiet=True))

    assert counts["duplicates"] == 0
    assert counts["suspicious"] == 0


def test_synthetic_renders_are_not_a_valid_control():
    """src.synth_data draws one template per class, so different draws land
    above the duplicate threshold. Recalibrating against them would set the
    thresholds somewhere meaningless -- this records why not to."""
    from src.synth_data import _render

    rng = np.random.default_rng(0)
    drawn = vectors([_render("NORMAL", rng) for _ in range(20)])

    similarity = drawn @ drawn.T
    np.fill_diagonal(similarity, -1.0)

    assert similarity.max() > DUPLICATE


# ------------------------------------------------------------- the mechanics


def test_correlation_ignores_brightness_and_contrast():
    """Two exposures of one radiograph are the same radiograph. Per-image
    centring and scaling is what makes that true of the score."""
    image = distinct_image(np.random.default_rng(3))
    darker = Image.eval(image, lambda value: int(value * 0.45) + 10)

    assert float(thumbnail_vector(image) @ thumbnail_vector(darker)) > 0.999


def test_a_blank_image_matches_nothing(tmp_path):
    """It has no structure, so the vector is zeros. Normalising it would divide
    by ~0 and make it correlate with everything instead."""
    blank = thumbnail_vector(Image.new("L", (224, 224), 128))
    assert not blank.any()

    other = thumbnail_vector(distinct_image(np.random.default_rng(4)))
    assert float(blank @ other) == pytest.approx(0.0)


def test_chunking_does_not_change_the_answer():
    """The real datasets need chunking for memory; it must be arithmetic-free."""
    left = vectors([distinct_image(np.random.default_rng(i)) for i in range(9)])
    right = vectors([distinct_image(np.random.default_rng(i)) for i in range(5, 14)])

    whole, whole_index = best_correlation(left, right, CPU, chunk=1000)
    split, split_index = best_correlation(left, right, CPU, chunk=2)

    assert np.allclose(whole, split, atol=1e-6)
    assert whole_index.tolist() == split_index.tolist()


def test_the_match_index_points_at_the_right_image():
    images = [distinct_image(np.random.default_rng(i)) for i in range(6)]
    right = vectors(images)
    # One query, which is a re-encode of the image at index 3.
    left = vectors([reencoded(images[3])])

    best, where = best_correlation(left, right, CPU)

    assert int(where[0]) == 3
    assert float(best[0]) > DUPLICATE


def test_identical_files_are_caught_by_both_passes(tmp_path):
    rng = np.random.default_rng(5)
    images = [distinct_image(rng) for _ in range(8)]
    write_images(tmp_path / "left" / "COVID19", images)
    write_images(tmp_path / "right" / "COVID19", images)

    counts = summarise(compare(tmp_path / "left", tmp_path / "right", device=CPU, quiet=True))

    assert counts["exact"] == 8
    assert counts["duplicates"] == 8


# ------------------------------------------------------------- file handling


def test_masks_are_skipped(tmp_path):
    """The Radiography Database ships a lung mask beside every image. They all
    look alike, so counting them would manufacture overlap out of nothing."""
    rng = np.random.default_rng(6)
    write_images(tmp_path / "COVID" / "images", [distinct_image(rng) for _ in range(3)])
    write_images(tmp_path / "COVID" / "masks", [distinct_image(rng) for _ in range(3)])

    assert len(find_images(tmp_path)) == 3
    assert len(find_images(tmp_path, skip=())) == 6


def test_non_images_are_ignored(tmp_path):
    write_images(tmp_path / "COVID19", [distinct_image(np.random.default_rng(7))])
    (tmp_path / "COVID19" / "notes.txt").write_text("not an image", encoding="utf-8")
    (tmp_path / "COVID19" / "archive.zip").write_bytes(b"PK\x03\x04")

    assert len(find_images(tmp_path)) == 1


def test_a_missing_directory_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_images(tmp_path / "nope")


def test_the_class_name_survives_a_structural_directory(tmp_path):
    assert class_of(tmp_path / "COVID" / "images" / "a.png", tmp_path) == "COVID"
    assert class_of(tmp_path / "test" / "NORMAL" / "a.png", tmp_path) == "NORMAL"


def test_comparing_against_an_empty_set_is_not_an_error(tmp_path):
    write_images(tmp_path / "left" / "COVID19", [distinct_image(np.random.default_rng(8))])
    (tmp_path / "right").mkdir()

    counts = summarise(compare(tmp_path / "left", tmp_path / "right", device=CPU, quiet=True))

    assert counts["images"] == 1
    assert counts["duplicates"] == 0


def test_the_thresholds_leave_a_gap():
    assert SUSPICIOUS < DUPLICATE < 1.0
