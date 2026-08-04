"""Grad-CAM: which pixels moved the prediction.

A number on its own ("87% pneumonia") is not reviewable by anybody. Grad-CAM
weights the last convolutional feature maps by how strongly the predicted class
score responds to each one, giving a coarse heatmap over the input.

For this project it is mainly a debugging tool, and it earns its place the
first time it catches the model cheating. Public chest X-ray sets are built by
merging collections, and each collection has its own tells: burned-in
annotations, a hospital's letter marker in one corner, a different aspect ratio,
a different exposure. A model that reaches 97% by reading those markers looks
perfect on every metric until the heatmap shows it never once looked at a lung.

Run it as a script to check a single image:

    python -m src.gradcam_utils --checkpoint checkpoints/best.pt \\
        --image data/test/PNEUMONIA/person1_virus_6.jpeg --out cam.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD, build_transforms
from src.model import load_checkpoint, pick_device, target_layer


class GradCAM:
    """Hooks one layer and turns a backward pass into a heatmap.

    Use it as a context manager -- the forward hook stays attached to the model
    until it is removed, and a leaked hook keeps every activation tensor alive.
    """

    def __init__(self, model, layer):
        self.model = model
        self._activations = None
        self._gradients = None
        self._handle = layer.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._activations = output
        # The gradient hook goes on the output tensor rather than the module.
        # Module-level backward hooks fire unreliably for blocks whose forward
        # splits and rejoins, which is exactly what a resnet or dense block is.
        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def _capture_gradient(self, grad):
        self._gradients = grad

    def __call__(self, input_tensor, class_idx=None):
        """Returns (cam as HxW float array in [0,1], class_idx, logits)."""
        if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
            raise ValueError("Grad-CAM expects one image at a time: (1, C, H, W)")

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "No activations captured. The target layer never ran, or the "
                "forward pass was under torch.no_grad()."
            )

        # One weight per feature map: how much that map, on average, pushes the
        # chosen class score up.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)

        # ReLU because only evidence *for* this class is wanted. Negative
        # contributions are evidence for the other two.
        cam = torch.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()
        peak = float(cam.max())
        # A flat-zero map means the layer contributed nothing to this class.
        # Dividing by it would produce NaNs that only surface at render time.
        if peak > 0:
            cam = cam / peak

        return cam, class_idx, logits.detach()

    def close(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def denormalize(tensor):
    """Undo Normalize, giving an HxWx3 uint8 array you can look at."""
    array = tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    array = array * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return (np.clip(array, 0, 1) * 255).astype(np.uint8)


def overlay_cam(base_rgb, cam, alpha=0.45, colormap="jet"):
    """Blend a heatmap over the X-ray and return a PIL image.

    jet is the default only because it is what published Grad-CAM figures use,
    which makes side-by-side comparison easier. Pass "inferno" for something
    perceptually uniform.
    """
    from matplotlib import colormaps

    coloured = colormaps[colormap](cam)[:, :, :3]
    coloured = (coloured * 255).astype(np.uint8)
    blended = (1 - alpha) * base_rgb.astype(float) + alpha * coloured.astype(float)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def explain_image(model, backbone, pil_image, device, class_idx=None):
    """Full path from a PIL image to (overlay image, label, probabilities)."""
    tensor = build_transforms(train=False)(pil_image.convert("L"))
    tensor = tensor.unsqueeze(0).to(device)

    with GradCAM(model, target_layer(model, backbone)) as cam_extractor:
        cam, predicted_idx, logits = cam_extractor(tensor, class_idx)

    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    overlay = overlay_cam(denormalize(tensor), cam)

    return overlay, CLASSES[predicted_idx], probabilities


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for one X-ray.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="cam.png")
    parser.add_argument(
        "--class-name",
        choices=CLASSES,
        help="Explain this class instead of the predicted one. Useful for "
        "asking why the model did *not* say pneumonia.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)

    class_idx = CLASSES.index(args.class_name) if args.class_name else None
    image = Image.open(args.image)
    overlay, label, probabilities = explain_image(
        model, checkpoint["backbone"], image, device, class_idx
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.out)

    print(f"predicted: {label}")
    for name, probability in zip(CLASSES, probabilities):
        print(f"  {name:<10} {probability:.3f}")
    print(f"saved overlay -> {args.out}")


if __name__ == "__main__":
    main()
