"""Lung masking: filename mapping, the mask itself, and the guards.

No real dataset here -- a few drawn images and hand-made binary masks are
enough to pin the behaviour that matters, which is that the mirror matches the
split it will be compared against.
"""

import numpy as np
import pytest
from PIL import Image

from src.mask_lungs import apply_mask, find_mask_dirs, original_name


def write_image(path, value, size=(8, 8)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(size[::-1], value, dtype=np.uint8), mode="L").save(path)


def write_mask(path, size=(8, 8), lung_rows=slice(2, 6)):
    """A mask that is white across lung_rows and black elsewhere."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros(size[::-1], dtype=np.uint8)
    array[lung_rows, :] = 255
    Image.fromarray(array, mode="L").save(path)


def test_the_index_prefix_comes_back_off():
    assert original_name("00042_COVID-1447.png") == "COVID-1447.png"
    assert original_name("00000_Viral Pneumonia-937.png") == "Viral Pneumonia-937.png"


def test_a_name_without_an_index_prefix_is_left_alone():
    """Underscores occur in real filenames; only a leading number is a prefix."""
    assert original_name("person1_virus_6.jpeg") == "person1_virus_6.jpeg"
    assert original_name("COVID-1.png") == "COVID-1.png"


def test_masking_zeroes_outside_the_lungs_and_preserves_inside(tmp_path):
    image_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    write_image(image_path, 200)
    write_mask(mask_path)

    result = np.array(apply_mask(image_path, mask_path))

    assert (result[2:6, :] == 200).all()
    assert (result[:2, :] == 0).all()
    assert (result[6:, :] == 0).all()


def test_a_mask_of_a_different_size_is_resized_not_rejected(tmp_path):
    """Masks ship at 256x256 against 299x299 images."""
    image_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    write_image(image_path, 180, size=(16, 16))
    write_mask(mask_path, size=(8, 8))

    result = np.array(apply_mask(image_path, mask_path))

    assert result.shape == (16, 16)
    assert result.max() == 180
    assert result.min() == 0


def test_mask_directories_resolve_through_the_shared_aliases(tmp_path):
    """"COVID" and "Viral Pneumonia" are understood, without a second mapping."""
    for source_name in ("COVID", "Normal", "Viral Pneumonia"):
        (tmp_path / source_name / "masks").mkdir(parents=True)
        (tmp_path / source_name / "images").mkdir(parents=True)

    found = find_mask_dirs(tmp_path)

    assert set(found) == {"COVID19", "NORMAL", "PNEUMONIA"}
    assert found["PNEUMONIA"].parent.name == "Viral Pneumonia"


def test_a_masks_directory_outside_our_classes_is_ignored(tmp_path):
    """Tuberculosis ships masks in some downloads and is not one of ours.
    Lung opacity is, so its masks are collected like any other class's."""
    (tmp_path / "Tuberculosis" / "masks").mkdir(parents=True)
    (tmp_path / "Lung_Opacity" / "masks").mkdir(parents=True)
    (tmp_path / "COVID" / "masks").mkdir(parents=True)

    found = find_mask_dirs(tmp_path)

    assert set(found) == {"COVID19", "LUNG_OPACITY"}


def test_a_download_without_masks_finds_nothing(tmp_path):
    (tmp_path / "COVID" / "images").mkdir(parents=True)
    assert find_mask_dirs(tmp_path) == {}
