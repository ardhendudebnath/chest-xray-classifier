"""Loading chest X-rays off disk and turning them into batched tensors.

The layout is one directory per class inside each split, which is what
torchvision's ImageFolder expects:

    data/train/COVID19/*.png
    data/train/NORMAL/*.png
    data/train/PNEUMONIA/*.png

CLASSES is fixed here rather than discovered. ImageFolder builds its label
mapping from whatever directories it happens to find, so a split that is
missing one class renumbers the rest -- a model trained on train/ would then be
scored against different labels in test/ with nothing to warn you.
build_dataset checks the mapping instead of trusting it.

Two things in here exist because of class imbalance, which every public chest
X-ray set has: pneumonia images usually outnumber normal ones several times
over, and COVID-19 is usually the smallest group. A model can post 75% accuracy
by never predicting the rarest class at all. make_sampler evens out what the
training loop sees; class_weights makes the loss care more about rare classes.
Use one or the other -- both at once over-corrects and the model starts
crying wolf.
"""

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# Alphabetical, so this matches the order ImageFolder assigns. Index 0 is
# COVID19, 1 is LUNG_OPACITY, 2 is NORMAL, 3 is PNEUMONIA -- in training, in
# evaluation, and in the served API.
#
# LUNG_OPACITY is here because leaving it out was worse than the alternative.
# It ships with the Radiography Database and was skipped for three releases as
# "a broader finding than pneumonia", which is true and was not the point: the
# model still met those films and had to answer, so it called them NORMAL 94.2%
# of the time at a mean confidence of 0.972. A class the model has no output
# for does not go away, it gets absorbed into whichever class is nearest, and
# here that was the one a reader is most likely to act on.
CLASSES = ["COVID19", "LUNG_OPACITY", "NORMAL", "PNEUMONIA"]
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 224

# ImageNet statistics. X-rays are not natural images, but the pretrained
# backbone's early filters were fitted against this scale; feeding them
# anything else throws away most of the benefit of starting pretrained.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(train):
    """Preprocessing for one split. Augmentation only ever applies to training."""
    # Grayscale(3) rather than one channel: X-rays carry no colour, but
    # ImageNet backbones have a 3-channel first conv, and replacing it means
    # throwing away its pretrained weights.
    steps = [transforms.Grayscale(num_output_channels=3)]

    if train:
        steps += [
            # Mild crops and rotations stand in for the framing and patient
            # positioning differences between machines and technicians.
            transforms.RandomResizedCrop(
                IMG_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.RandomRotation(10),
            # Exposure varies a lot between imaging setups. Jittering it stops
            # the model keying on one hospital's particular look.
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
        # Deliberately no horizontal flip. It is the default augmentation for
        # natural images and it appears in most X-ray notebooks, but a
        # mirrored chest puts the heart on the right -- an anatomy that occurs
        # in roughly 1 in 10,000 people. Teaching the model that it is ordinary
        # is not a trade worth making for a little extra data.
    else:
        steps.append(transforms.Resize((IMG_SIZE, IMG_SIZE)))

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def build_dataset(split_dir, train):
    """An ImageFolder over one split, with its label mapping verified."""
    root = Path(split_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} does not exist. Expected one directory per class inside "
            f"it: {', '.join(CLASSES)}."
        )

    dataset = datasets.ImageFolder(str(root), transform=build_transforms(train))

    found = list(dataset.class_to_idx.keys())
    if found != CLASSES:
        raise ValueError(
            f"{root} has classes {found}, expected exactly {CLASSES}. "
            "A missing or extra directory here silently renumbers every label."
        )
    if len(dataset) == 0:
        raise ValueError(f"{root} contains no images.")

    return dataset


def class_counts(dataset):
    """How many images per class name. Print this before every training run."""
    counts = Counter(CLASSES[label] for _, label in dataset.samples)
    return {name: counts.get(name, 0) for name in CLASSES}


def class_weights(dataset):
    """Inverse-frequency weights for CrossEntropyLoss.

    Averaged over the dataset these come to 1.0 -- every class contributes
    total/NUM_CLASSES to sum(count * weight) -- so a weighted run sits on the
    same loss scale as an unweighted one and --lr carries over between them.
    Averaged over the four classes they do not, and climb with the imbalance.
    Only the ratios reach the optimizer regardless: CrossEntropyLoss divides by
    the summed weights of the batch, so a constant factor cancels.
    """
    counts = class_counts(dataset)
    total = sum(counts.values())
    return torch.tensor(
        [total / (NUM_CLASSES * max(counts[name], 1)) for name in CLASSES],
        dtype=torch.float32,
    )


def make_sampler(dataset):
    """Draw each class about equally often, oversampling the rare ones."""
    counts = class_counts(dataset)
    per_class = {name: 1.0 / max(counts[name], 1) for name in CLASSES}
    weights = [per_class[CLASSES[label]] for _, label in dataset.samples]
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )


def build_loader(split_dir, train, batch_size=32, num_workers=0, balance=True):
    """Returns (loader, dataset). The dataset is handy for counts and paths.

    num_workers defaults to 0 because on Windows each worker re-imports the
    calling module, which deadlocks unless the entry point is guarded by
    `if __name__ == "__main__"`. Raise it on Linux, or after adding the guard.
    """
    dataset = build_dataset(split_dir, train=train)
    sampler = make_sampler(dataset) if (train and balance) else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(train and sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset
