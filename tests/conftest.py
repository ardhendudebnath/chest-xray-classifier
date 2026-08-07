"""Shared fixtures.

Everything here builds on synthetic images and untrained weights, so the whole
suite runs in well under a minute on a CPU and needs no dataset and no network.
`pretrained=False` throughout for the same reason -- the first pretrained
build_model call downloads 45 MB of ImageNet weights.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# The modules import as `src.thing`, which only resolves with the project root
# on the path. Adding it here means pytest works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CLASSES  # noqa: E402
from src.model import build_model, save_checkpoint  # noqa: E402
from src.synth_data import _render  # noqa: E402

# Uneven on purpose: the sampler and the class weights are what these counts
# exist to exercise.
COUNTS = {"COVID19": 4, "LUNG_OPACITY": 7, "NORMAL": 6, "PNEUMONIA": 10}


def write_split(directory, counts, rng):
    for class_name, count in counts.items():
        class_dir = directory / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            _render(class_name, rng).save(class_dir / f"{index:03d}.png")


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    """A dataset directory with all three splits and all four classes."""
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)

    write_split(root / "train", COUNTS, rng)
    write_split(root / "val", {name: 2 for name in CLASSES}, rng)
    write_split(root / "test", {name: 2 for name in CLASSES}, rng)

    return root


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory):
    """An untrained resnet18 saved the way train.py saves one.

    Untrained is fine for every test here: they check shapes, plumbing and
    error handling, none of which care whether the predictions are any good.
    """
    path = tmp_path_factory.mktemp("checkpoints") / "test.pt"
    model = build_model("resnet18", pretrained=False)
    save_checkpoint(path, model, "resnet18", epoch=1, metrics={"macro_f1": 0.0})
    return path


@pytest.fixture
def xray_image(tmp_path):
    """One synthetic PNG on disk, and the PIL image beside it."""
    from PIL import Image

    path = tmp_path / "xray.png"
    _render("PNEUMONIA", np.random.default_rng(7)).save(path)
    return path, Image.open(path)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
