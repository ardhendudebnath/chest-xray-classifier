"""SHAP: which learned features moved the logit, and by how much.

Grad-CAM answers *where* in the image the evidence was. It cannot say what the
evidence was, and it cannot put a number on it -- a heatmap is a picture, and
two heatmaps that look alike can come from quite different reasons. This module
answers the other half: of the numbers the classification head actually reads,
which ones pushed the prediction up, which pushed it down, and by how many
logits each.

The attribution is over the **penultimate features**, not over pixels. That is a
deliberate choice and it is what makes this complementary to Grad-CAM rather
than a second opinion on the same question. Pixel-level SHAP would produce
another spatial map, and the project would then have two answers to "where" and
none to "how much".

*Why this can be exact.* The head is one nn.Linear over those features, so the
logit is w . x + b and nothing else. For a linear model the Shapley values have
a closed form -- there is no sampling, no approximation and no convergence to
check:

    phi_i = w_i * (x_i - E[x_i])        base = w . E[x] + b

with E[x] the mean feature vector over a reference set. Sum the contributions,
add the base, and you get the logit back exactly. That identity is the Shapley
efficiency axiom, and `tests/test_shap.py` asserts it to floating-point
tolerance rather than trusting it. It also asserts these values agree with
shap.LinearExplainer, which is the reference implementation of Lundberg & Lee
2017; the closed form is used at runtime so that serving a prediction does not
depend on a heavyweight optional package.

    python -m src.shap_utils --fit --checkpoint checkpoints/best.pt
    python -m src.shap_utils --checkpoint checkpoints/best.pt --image chest.png

**Four things this does not tell you.** They are the whole reason the module has
a docstring this long.

- *A feature index is not a clinical concept.* "Feature 314 contributed +0.81
  logits" is a true statement about the network and says nothing to a
  radiologist. The 512 numbers are whatever the backbone learned; none of them
  is "consolidation". `channel_map` exists to close some of that gap, by showing
  where the channel behind a contribution actually fires, but a named finding is
  not recoverable from this and should not be implied.
- *The features are correlated; this treats them as independent.* Interventional
  Shapley values assume a feature can be held out on its own, and neighbouring
  channels of a conv net plainly cannot. This is the same assumption
  shap.LinearExplainer makes by default, so the number is standard rather than
  unusual, but it is an assumption. src.ood already estimates the covariance of
  exactly these features if a correlation-aware variant is ever wanted.
- *It explains logits, not probabilities.* Softmax is not additive, so no
  decomposition of a probability sums to that probability. The logit is the
  thing the linear head produces and the only thing an additive attribution can
  be exact about.
- *The baseline carries the training set's provenance.* E[x] is the mean feature
  vector over training images, so every contribution is measured against "the
  average image in this dataset" -- not against a healthy chest. Given that this
  dataset's classes are 100% separable by source archive (see the README), a
  large contribution is as consistent with "unlike the average scanner here" as
  with "unlike a healthy lung". SHAP inherits the confound; it does not detect
  it.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.dataset import CLASSES, NUM_CLASSES, build_loader, build_transforms
from src.model import (
    fingerprint_state_dict,
    forward_with_features,
    head_layer,
    load_checkpoint,
    pick_device,
    target_layer,
)

DEFAULT_BACKGROUND_PATH = "checkpoints/shap_background.pt"

# How many features to show by default. The vector is 512 long for resnet18 and
# 2048 for resnet50; a bar chart of all of them is a solid block of ink, and the
# tail is individually negligible either way. The printed summary always reports
# how much of the total the shown bars account for, so a k that hides the story
# is visible as a small covered fraction rather than being silently misleading.
DEFAULT_TOP_K = 15


def head_parameters(model, backbone):
    """The head's (weight (C, D), bias (C,)) as float64 numpy.

    float64 because the efficiency check below compares a sum of D terms
    against a single logit, and in float32 the accumulated rounding over 512
    terms is the same size as the smallest contributions being reported.
    """
    head = head_layer(model, backbone)
    weight = head.weight.detach().cpu().numpy().astype(np.float64)
    bias = head.bias.detach().cpu().numpy().astype(np.float64)
    return weight, bias


def shap_values(features, background_mean, weight, bias):
    """Exact Shapley values for a linear head. Returns (values (N, C, D), base (C,)).

    values[n, c, i] is how many logits feature i contributed to class c for
    image n, relative to an image whose features were average. base[c] is what
    that average image would have scored.

    The identity that makes this checkable: values.sum(axis=2) + base is the
    logit vector, exactly.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features[None, :]

    background_mean = np.asarray(background_mean, dtype=np.float64)
    if features.shape[1] != background_mean.shape[0]:
        raise ValueError(
            f"Features are {features.shape[1]}-dimensional but the background "
            f"is {background_mean.shape[0]}. These were fitted for a different "
            f"backbone."
        )

    # (N, 1, D) * (1, C, D) -> (N, C, D). The deviation from the reference is
    # what is being attributed; a feature sitting exactly at the mean
    # contributes nothing, whatever its weight.
    deviation = (features - background_mean)[:, None, :]
    return deviation * weight[None, :, :], weight @ background_mean + bias


def explain_tensor(model, backbone, batch, background, class_idx=None):
    """SHAP for a preprocessed batch. Returns (values, base, logits, features).

    One forward pass. The features feeding the head are exactly what src.ood
    measures distances in, so a caller that wants both pays for one pass.
    """
    with torch.no_grad():
        logits, features = forward_with_features(model, backbone, batch)

    weight, bias = head_parameters(model, backbone)
    values, base = shap_values(
        features.cpu().numpy(), background["mean"], weight, bias
    )

    if class_idx is not None:
        values = values[:, class_idx, :]
        base = base[class_idx]

    return values, base, logits.cpu(), features.cpu()


def margin_values(values, positive_idx, negative_idx):
    """Contributions to (logit[positive] - logit[negative]).

    The decision is an argmax, so what actually separates two classes is the
    difference of their logits, and a feature can push both up while doing
    nothing to choose between them. This is the SHAP counterpart to passing
    --class-name to src.gradcam_utils: it answers "why this rather than that",
    which is a different question from "why this".

    Exact for the same reason the rest of the module is -- a difference of two
    linear functions is linear, so the Shapley values subtract.
    """
    return values[:, positive_idx, :] - values[:, negative_idx, :]


def top_contributions(values, k=DEFAULT_TOP_K):
    """The k features with the largest |contribution|, most important first.

    Ranked by magnitude and reported with sign, rather than ranked by signed
    value: the strongest evidence *against* the predicted class is as
    interesting as the strongest evidence for it, and a signed ranking buries it
    at the far end.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    k = min(int(k), values.size)
    order = np.argsort(np.abs(values))[::-1][:k]
    return [(int(index), float(values[index])) for index in order]


def coverage(values, k=DEFAULT_TOP_K):
    """What fraction of the total |contribution| the top k account for.

    Printed beside every summary. A chart of 15 bars covering 12% of the moved
    mass is a chart that has hidden the answer, and the only way to see that
    from the picture alone is to already know it.
    """
    values = np.abs(np.asarray(values, dtype=np.float64).ravel())
    total = values.sum()
    if total == 0:
        return 0.0
    k = min(int(k), values.size)
    return float(np.sort(values)[::-1][:k].sum() / total)


# ------------------------------------------------------------------ background


def fit_background(model, backbone, loader, device):
    """The mean feature vector over a split, plus what it was fitted against.

    The reference point every contribution is measured from. Fitted on the
    training split with augmentation off, for the same reason src.ood fits its
    Gaussians that way: augmentation exists to vary what the training loop sees,
    and a reference that averages over crops and rotations describes images that
    serving never produces.

    A mean is all a linear head needs -- the closed form depends on the
    background only through E[x] -- so this is one pass and a few KB, not a
    stored sample of images.
    """
    totals = None
    count = 0

    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            _, features = forward_with_features(model, backbone, images.to(device))
            features = features.cpu().numpy().astype(np.float64)
            totals = features.sum(axis=0) if totals is None else totals + features.sum(axis=0)
            count += features.shape[0]

    if not count:
        raise ValueError("No images in the background split, so there is no mean.")

    return {
        "mean": torch.from_numpy(totals / count),
        "classes": list(CLASSES),
        "backbone": backbone,
        "feature_dim": int(totals.shape[0]),
        "images": int(count),
        # Same guard as src.ood. A background fitted against different weights
        # describes a different feature space, and using it produces
        # contributions that are wrong with no symptom other than looking odd.
        "fingerprint": fingerprint_state_dict(model.state_dict()),
    }


def save_background(path, background):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(background, path)
    return path


def load_background(path, model=None):
    """Load a fitted background, refusing one that belongs to another checkpoint.

    Copied in spirit from src.ood.load_stats, and for the same reason: the
    failure it prevents is silent. Feature index 314 means one thing to one set
    of weights and something unrelated to another, so a mismatched background
    yields a confident, well-formatted, meaningless bar chart.
    """
    background = torch.load(path, map_location="cpu", weights_only=True)

    if background.get("classes") != list(CLASSES):
        raise ValueError(
            f"SHAP background was fitted for {background.get('classes')}, but "
            f"this code expects {list(CLASSES)}."
        )

    if model is not None:
        expected = fingerprint_state_dict(model.state_dict())
        if background.get("fingerprint") != expected:
            raise ValueError(
                f"SHAP background at {path} was fitted against a different "
                f"checkpoint ({background.get('fingerprint')} vs {expected}). "
                f"Refit it with python -m src.shap_utils --fit."
            )

    return background


# ------------------------------------------------------- tracing back to pixels


def channel_map(model, backbone, batch, channel):
    """Where the channel behind one contribution fires, as an HxW array in [0,1].

    The bridge between a feature index and something a person can look at. For
    a resnet, feature i of the head's input is the spatial mean of channel i of
    the last conv block, so the unpooled channel is exactly the map behind the
    number -- `test_shap.py` pins that relationship rather than assuming it.

    For densenet121 it is close but not exact: torchvision applies norm5 and a
    ReLU between the block this reads and the pooling that produces the feature.
    The map still locates the channel; it will not average to the feature value.
    """
    captured = {}

    def capture(module, inputs, output):
        captured["activations"] = output.detach()

    handle = target_layer(model, backbone).register_forward_hook(capture)
    try:
        with torch.no_grad():
            model(batch)
    finally:
        handle.remove()

    activations = captured["activations"][0, int(channel)].cpu().numpy()
    peak = float(np.abs(activations).max())
    # A dead channel would otherwise divide by zero and render as NaN, which
    # matplotlib draws as blank -- indistinguishable from a channel that fired
    # evenly everywhere.
    return activations / peak if peak > 0 else activations


# ------------------------------------------------------------------- rendering


def save_bar_chart(contributions, path, title, base=None, total=None):
    """The paper's "SHAP value bar chart per prediction", written to a PNG.

    Horizontal bars, most important at the top, red for evidence toward the
    class and blue for evidence against it. That is the one place in this
    project where colour encodes meaning, and it is safe here for the reason it
    is not safe in the frontend: the quantity being coloured is the sign of a
    number, not a disease.
    """
    import matplotlib

    # Agg so this works over SSH and in CI, where there is no display.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = [f"feature {index}" for index, _ in contributions][::-1]
    values = [value for _, value in contributions][::-1]

    figure, axes = plt.subplots(figsize=(7, 0.32 * len(values) + 1.6))
    axes.barh(
        range(len(values)),
        values,
        color=["#c0392b" if value > 0 else "#2874a6" for value in values],
    )
    axes.set_yticks(range(len(values)), indices, fontsize=8)
    axes.axvline(0, color="black", linewidth=0.8)
    axes.set_xlabel("contribution to the logit (SHAP value)")
    axes.set_title(title, fontsize=10)

    if base is not None and total is not None:
        axes.text(
            0.99,
            0.01,
            f"base {base:+.3f}  +  contributions {total:+.3f}  =  logit {base + total:+.3f}",
            transform=axes.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def explain_image(model, backbone, pil_image, device, background, class_idx=None):
    """Full path from a PIL image to (contributions, base, probabilities, label).

    The single-image entry point the API and the CLI both go through, shaped
    like gradcam_utils.explain_image so the two read the same way at the call
    site.
    """
    tensor = build_transforms(train=False)(pil_image.convert("L"))
    tensor = tensor.unsqueeze(0).to(device)

    values, base, logits, _ = explain_tensor(model, backbone, tensor, background)
    probabilities = torch.softmax(logits, dim=1)[0].numpy()

    if class_idx is None:
        class_idx = int(probabilities.argmax())

    return values[0, class_idx], float(base[class_idx]), probabilities, CLASSES[class_idx]


# ------------------------------------------------------------------------ CLI


def _fit_and_save(args, model, backbone, device):
    loader, dataset = build_loader(
        Path(args.data_dir) / args.background_split,
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"background from {args.background_split}: {len(dataset)} images")

    background = fit_background(model, backbone, loader, device)
    path = save_background(args.background, background)

    print(f"feature dim: {background['feature_dim']}")
    print(f"saved {path}")
    print("explain an image with: python -m src.shap_utils --image <file>")


def _explain_and_report(args, model, backbone, device):
    from PIL import Image

    background = load_background(args.background, model)
    image = Image.open(args.image)

    tensor = build_transforms(train=False)(image.convert("L")).unsqueeze(0).to(device)
    values, base, logits, _ = explain_tensor(model, backbone, tensor, background)
    probabilities = torch.softmax(logits, dim=1)[0].numpy()

    predicted_idx = int(probabilities.argmax())
    explained_idx = CLASSES.index(args.class_name) if args.class_name else predicted_idx

    print(f"predicted: {CLASSES[predicted_idx]} ({probabilities[predicted_idx]:.3f})")
    for name, probability in zip(CLASSES, probabilities):
        print(f"  {name:<14}{probability:.3f}")

    if args.against:
        against_idx = CLASSES.index(args.against)
        if against_idx == explained_idx:
            raise SystemExit("--against names the class being explained.")
        selected = margin_values(values, explained_idx, against_idx)[0]
        base_value = float(base[explained_idx] - base[against_idx])
        heading = f"{CLASSES[explained_idx]} over {CLASSES[against_idx]}"
        quantity = "logit margin"
    else:
        selected = values[0, explained_idx]
        base_value = float(base[explained_idx])
        heading = CLASSES[explained_idx]
        quantity = "logit"

    total = float(selected.sum())
    covered = coverage(selected, args.top_k)
    contributions = top_contributions(selected, args.top_k)

    print(f"\nSHAP over {selected.size} features, explaining the {heading} {quantity}")
    print(f"  base value (average training image)  {base_value:+.4f}")
    print(f"  sum of contributions                 {total:+.4f}")
    print(f"  {quantity:<36} {base_value + total:+.4f}")
    print(f"\ntop {len(contributions)} of {selected.size} features "
          f"({covered:.0%} of the total movement)")
    for index, value in contributions:
        print(f"  {value:+8.4f}  feature {index}")

    save_bar_chart(
        contributions,
        args.out,
        f"SHAP: {heading} ({quantity}) -- {Path(args.image).name}",
        base=base_value,
        total=total,
    )
    print(f"\nsaved bar chart -> {args.out}")

    if args.channel_map:
        top_index = contributions[0][0]
        activation = channel_map(model, backbone, tensor, top_index)
        _save_channel_map(activation, args.channel_map, top_index, contributions[0][1])
        print(f"saved channel {top_index} map -> {args.channel_map}")
        print("a channel is not a finding -- see the module docstring")


def _save_channel_map(activation, path, channel, value):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(4, 4))
    axes.imshow(activation, cmap="inferno")
    axes.set_title(f"feature {channel} ({value:+.3f} logits)", fontsize=10)
    axes.axis("off")
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="SHAP feature attributions for one X-ray, or fit the background."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--background", default=DEFAULT_BACKGROUND_PATH)
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Fit the background mean and exit. Do this once per checkpoint.",
    )
    parser.add_argument("--image", help="The X-ray to explain.")
    parser.add_argument("--out", default="shap.png")
    parser.add_argument(
        "--class-name",
        choices=CLASSES,
        help="Explain this class instead of the predicted one.",
    )
    parser.add_argument(
        "--against",
        choices=CLASSES,
        help="Explain the margin over this class rather than the logit itself, "
        "which is what actually decides the prediction.",
    )
    parser.add_argument(
        "--channel-map",
        help="Also write the spatial map of the top feature to this path, to "
        "see where the channel behind it fires.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--background-split", default="train", choices=("train", "val", "test")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.fit and not args.image:
        parser.error("pass --image to explain one, or --fit to fit the background.")

    device = pick_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    backbone = checkpoint["backbone"]
    print(f"checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']}, {backbone})")

    if args.fit:
        _fit_and_save(args, model, backbone, device)
        return

    _explain_and_report(args, model, backbone, device)


if __name__ == "__main__":
    main()
