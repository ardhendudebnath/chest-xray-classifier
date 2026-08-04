"""Dataset normalisation: aliases, patient grouping, and the split itself."""

import random

import numpy as np
import pytest

from src.dataset import CLASSES
from src.prepare_data import (
    collect_images,
    find_class_dirs,
    patient_group,
    resolve_class,
    split_groups,
)
from src.synth_data import _render


@pytest.mark.parametrize(
    "directory_name,expected",
    [
        ("COVID", "COVID19"),
        ("covid-19", "COVID19"),
        ("Normal", "NORMAL"),
        ("Viral Pneumonia", "PNEUMONIA"),
        ("PNEUMONIA", "PNEUMONIA"),
        ("Lung_Opacity", None),
        ("Tuberculosis", None),
        ("masks", None),
    ],
)
def test_source_directory_names_map_to_our_classes(directory_name, expected):
    assert resolve_class(directory_name) == expected


def test_alias_matching_is_whole_name_not_substring():
    """'Non-COVID' contains 'covid' and must not be read as the COVID class."""
    assert resolve_class("Non-COVID") is None


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("person1_virus_6.jpeg", "person1"),
        ("person1_bacteria_1.jpeg", "person1"),
        ("PERSON23_virus_2.jpeg", "person23"),
        ("COVID-1.png", "covid-1.png"),
    ],
)
def test_patient_id_is_pulled_out_of_the_filename(filename, expected):
    from pathlib import Path

    assert patient_group(Path(filename)) == expected


def _make_source(root, layout):
    """layout: {source_dir_relative_path: [filenames]}"""
    rng = np.random.default_rng(0)
    for relative, filenames in layout.items():
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            _render("NORMAL", rng).save(directory / filename)
    return root


def test_images_nested_under_an_images_subdir_are_found(tmp_path):
    """The COVID-19 Radiography Database puts them in COVID/images/."""
    source = _make_source(
        tmp_path,
        {
            "COVID/images": ["COVID-1.png", "COVID-2.png"],
            "Normal/images": ["Normal-1.png"],
            "Viral Pneumonia/images": ["Viral Pneumonia-1.png"],
        },
    )

    found, _ = find_class_dirs(source)
    assert set(found) == set(CLASSES)


def test_an_already_split_download_is_pooled_before_resplitting(tmp_path):
    """Published splits are not patient-grouped, so they are not trusted."""
    source = _make_source(
        tmp_path,
        {
            "train/COVID19": ["a.png"],
            "train/NORMAL": ["b.png"],
            "train/PNEUMONIA": ["c.png"],
            "test/COVID19": ["d.png"],
            "test/NORMAL": ["e.png"],
            "test/PNEUMONIA": ["f.png"],
        },
    )

    found, _ = find_class_dirs(source)
    assert len(found["COVID19"]) == 2
    assert len(collect_images(found["COVID19"])) == 2


def test_unmapped_directories_are_reported_not_silently_dropped(tmp_path):
    source = _make_source(
        tmp_path,
        {
            "COVID": ["a.png"],
            "Normal": ["b.png"],
            "PNEUMONIA": ["c.png"],
            "Lung_Opacity": ["d.png"],
        },
    )

    found, ignored = find_class_dirs(source)
    assert "Lung_Opacity" not in found
    assert "Lung_Opacity" in ignored


def test_one_patient_never_lands_in_two_splits(tmp_path):
    """The whole reason grouping exists: this inflates the test score."""
    source = _make_source(
        tmp_path,
        {
            "PNEUMONIA": [
                f"person{patient}_virus_{image}.png"
                for patient in range(1, 21)
                for image in range(3)
            ]
        },
    )

    groups = collect_images([source / "PNEUMONIA"])
    assert len(groups) == 20
    assert all(len(paths) == 3 for paths in groups.values())

    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    assignment = split_groups(groups, ratios, random.Random(0))

    seen = {
        split: {patient_group(path) for path in paths}
        for split, paths in assignment.items()
    }
    assert seen["train"].isdisjoint(seen["val"])
    assert seen["train"].isdisjoint(seen["test"])
    assert seen["val"].isdisjoint(seen["test"])


def test_split_lands_close_to_the_requested_ratios(tmp_path):
    groups = {f"img{i}.png": [tmp_path / f"img{i}.png"] for i in range(200)}
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    assignment = split_groups(groups, ratios, random.Random(0))
    total = sum(len(paths) for paths in assignment.values())

    assert total == 200
    for split, ratio in ratios.items():
        assert len(assignment[split]) / total == pytest.approx(ratio, abs=0.02)


def test_splitting_is_reproducible_for_a_fixed_seed(tmp_path):
    groups = {f"img{i}.png": [tmp_path / f"img{i}.png"] for i in range(50)}
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    first = split_groups(groups, ratios, random.Random(42))
    second = split_groups(groups, ratios, random.Random(42))
    assert first == second
