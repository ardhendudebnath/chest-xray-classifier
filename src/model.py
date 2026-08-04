"""The classifier: an ImageNet-pretrained backbone with a fresh 3-class head.

Training a CNN from scratch needs far more labelled images than the public
chest X-ray sets provide, and it would spend its first several epochs
relearning the edge and texture filters ImageNet weights already encode. So the
whole network is kept and only the final classification layer is replaced.

Checkpoints store the backbone name and the class list next to the weights. The
API has to rebuild the identical architecture before a state dict will load,
and a class list that drifts between training and serving relabels every
prediction without raising anything.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from src.dataset import CLASSES, IMG_SIZE, NUM_CLASSES

SUPPORTED_BACKBONES = ("resnet18", "resnet50", "densenet121")

# resnet18 is the default because it trains in minutes on a CPU and is large
# enough to hit the mid-90s on this task. densenet121 is what most chest X-ray
# papers use and is worth trying once the pipeline works end to end.
DEFAULT_BACKBONE = "resnet18"


def build_model(backbone=DEFAULT_BACKBONE, pretrained=True, freeze_backbone=False):
    """A backbone with its 1000-class ImageNet head swapped for a 3-class one.

    freeze_backbone trains only the new head. That is the right first move when
    the training set is small -- a few hundred images per class -- because
    fine-tuning everything on that much data mostly memorises it. Unfreeze once
    the head has converged.
    """
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Supported: {', '.join(SUPPORTED_BACKBONES)}"
        )

    weights = "DEFAULT" if pretrained else None

    if backbone.startswith("resnet"):
        model = getattr(models, backbone)(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        head = model.fc
    else:
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        head = model.classifier

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in head.parameters():
            param.requires_grad = True

    return model


def target_layer(model, backbone):
    """The last convolutional block, which is what Grad-CAM reads.

    Later layers carry the semantics but a coarser grid; earlier ones are
    sharper but describe textures rather than findings. The last block is the
    usual compromise -- 7x7 for a 224px input.
    """
    if backbone.startswith("resnet"):
        return model.layer4[-1]
    if backbone == "densenet121":
        return model.features.denseblock4
    raise ValueError(f"No Grad-CAM target layer defined for {backbone!r}")


def save_checkpoint(path, model, backbone, epoch, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": backbone,
            "classes": list(CLASSES),
            "img_size": IMG_SIZE,
            "epoch": epoch,
            # Plain floats only: torch.load(weights_only=True) refuses to
            # unpickle numpy scalars, and that flag is worth keeping on.
            "metrics": {k: float(v) for k, v in metrics.items()},
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    """Returns (model in eval mode on device, checkpoint metadata dict)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    saved_classes = checkpoint.get("classes")
    if saved_classes != list(CLASSES):
        raise ValueError(
            f"Checkpoint was trained on {saved_classes}, but this code expects "
            f"{list(CLASSES)}. Loading it anyway would relabel every prediction."
        )

    model = build_model(checkpoint["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def pick_device(preferred=None):
    """CUDA when it is there, otherwise CPU. resnet18 is fine on CPU."""
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
