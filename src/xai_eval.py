"""Scoring the explanations themselves, rather than the predictions.

A heatmap that looks anatomically sensible is not evidence that it found the
evidence. Everything in this project so far measures the classifier; this
measures whether Grad-CAM and SHAP are describing what the classifier actually
did. Four numbers, answering three questions:

*Is the heatmap pointing at pixels the model relies on?* Deletion and insertion
AUC, from Petsiuk et al. 2018 ("RISE"). Remove the pixels the map ranks highest
and the predicted probability should collapse; start from a blurred image and
add those pixels back and it should recover. Both are computed against a random
ordering as a control, because an AUC on its own has no scale -- a map that
ranks pixels arbitrarily still produces a curve, and it is the gap between the
two that means anything.

*Is it pointing inside the lungs?* The fraction of Grad-CAM mass falling within
the lung fields, reported against the fraction of the image those fields
occupy. **This is not the IoU against radiologist-annotated pathology that an
explanation-fidelity protocol really wants**, and the difference matters enough
to state twice. The masks shipped with the COVID-19 Radiography Database
segment *lungs*, not findings, so a map covering both entire lungs scores
perfectly here while having localised nothing. Read the enrichment ratio, which
is the mass fraction divided by the area fraction: 1.0 is exactly what an
uninformative map achieves. True pathology IoU needs boxes this dataset does
not carry -- the RSNA Pneumonia Detection set or the 984 annotated NIH images.

*Are the SHAP explanations consistent between similar cases?* Mean pairwise
cosine similarity of the SHAP vectors within a class, against the same quantity
computed across classes as a control. Also the coverage: what fraction of the
total attributed movement the top k features account for, which is what decides
whether a bar chart of 15 bars out of 512 is a summary or a sample.

    python -m src.xai_eval --checkpoint checkpoints/best.pt --limit 200

**One caveat that applies to the deletion and insertion numbers specifically,
and that this module measures rather than merely noting.** Blanking pixels
produces an image unlike any radiograph, so a probability that falls after
deletion may be reporting "this is no longer a chest X-ray" rather than "the
evidence is gone". That objection is usually left as a footnote in papers that
use these metrics. Here it does not have to be: src.ood already knows what
in-distribution looks like, so when statistics are available this reports the
fraction of perturbation steps that the model's own detector would refuse. A
deletion curve whose steps are 90% refused is not measuring explanation
fidelity.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.dataset import CLASSES, build_loader
from src.gradcam_utils import GradCAM
from src.model import (
    forward_with_features,
    load_checkpoint,
    pick_device,
    target_layer,
)
from src.shap_utils import (
    DEFAULT_BACKGROUND_PATH,
    DEFAULT_TOP_K,
    coverage,
    head_parameters,
    load_background,
    shap_values,
)

# 50 steps over a 224x224 image is roughly 1,000 pixels a step -- fine enough
# that the curve is not a staircase, coarse enough that one image is one batched
# forward pass rather than fifty.
DEFAULT_STEPS = 50

# The blur that insertion starts from and deletion falls back to. Large enough
# to destroy the texture the model reads, small enough to keep the silhouette,
# which is what stops the baseline itself being a uniform field the model has
# never seen.
BLUR_KERNEL = 31
BLUR_SIGMA = 15.0


def blurred(batch):
    """The insertion baseline: same image, no readable detail."""
    from torchvision.transforms.functional import gaussian_blur

    return gaussian_blur(batch, kernel_size=BLUR_KERNEL, sigma=BLUR_SIGMA)


def perturbation_order(attribution, generator=None):
    """Flat pixel indices, most important first.

    Ties are broken randomly rather than by index. Grad-CAM is computed on a 7x7
    grid and bilinearly upsampled, so large blocks of pixels carry identical
    values; ordering those by raster index would delete them top-left first and
    put a spatial bias into a metric that is supposed to be about ranking.
    """
    flat = np.asarray(attribution, dtype=np.float64).ravel()
    generator = generator or np.random.default_rng(0)

    jitter = generator.random(flat.size) * 1e-12
    return np.argsort(-(flat + jitter))


def _stepped_masks(order, steps):
    """(steps+1, H*W) boolean masks, revealing `order` progressively.

    Row s has the first s/steps of the ordering set True. Built once and
    applied to both curves, so deletion and insertion are exact complements of
    each other rather than two similar-looking orderings.
    """
    total = order.size
    masks = np.zeros((steps + 1, total), dtype=bool)
    for step in range(1, steps + 1):
        masks[step, order[: int(round(total * step / steps))]] = True
    return masks


def _probabilities_for(model, batch, class_idx):
    with torch.no_grad():
        logits = model(batch)
    return torch.softmax(logits, dim=1)[:, class_idx].cpu().numpy()


def curves(model, tensor, order, class_idx, steps=DEFAULT_STEPS, batch_size=32):
    """(deletion, insertion) probability curves, each of length steps+1.

    Deletion replaces the highest-ranked pixels with the blurred image;
    insertion starts from the blurred image and restores those same pixels.
    Blur rather than zeros for both: zero is pure black, which is what the
    background of a radiograph already is, so a zero-filled region reads as
    plausible anatomy rather than as absence and the two curves stop being
    comparable.
    """
    baseline = blurred(tensor)
    height, width = tensor.shape[-2:]
    masks = torch.from_numpy(
        _stepped_masks(order, steps).reshape(steps + 1, 1, height, width)
    ).to(tensor.device)

    deletion = []
    insertion = []

    for start in range(0, steps + 1, batch_size):
        chunk = masks[start : start + batch_size]
        # Where the mask is True the pixel has been "removed" for deletion and
        # "restored" for insertion -- the same mask, used both ways round.
        deletion.append(
            _probabilities_for(
                model, torch.where(chunk, baseline, tensor), class_idx
            )
        )
        insertion.append(
            _probabilities_for(
                model, torch.where(chunk, tensor, baseline), class_idx
            )
        )

    return np.concatenate(deletion), np.concatenate(insertion)


def curve_auc(curve):
    """Normalised area under a curve sampled evenly over a fraction in [0, 1].

    Trapezoid rather than a plain mean: the endpoints are the unperturbed image
    and the fully perturbed one, and a mean would weight those as heavily as the
    interior points despite each covering half a step.

    Written out rather than calling numpy, which spells this np.trapz below 2.0
    and np.trapezoid from 2.0 on. requirements.txt allows both.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if curve.size < 2:
        raise ValueError("A curve needs at least two points to have an area.")
    return float((curve[0] / 2 + curve[1:-1].sum() + curve[-1] / 2) / (curve.size - 1))


def ood_refusal_rate(model, backbone, tensor, order, stats, steps=DEFAULT_STEPS):
    """How many deletion steps the model's own detector would refuse.

    The honesty check on the two numbers above. If the perturbed images are
    off-distribution, the probability drop they produce is not evidence that the
    map found the evidence.
    """
    from src.ood import is_out_of_distribution, score

    baseline = blurred(tensor)
    height, width = tensor.shape[-2:]
    masks = torch.from_numpy(
        _stepped_masks(order, steps).reshape(steps + 1, 1, height, width)
    ).to(tensor.device)

    with torch.no_grad():
        _, features = forward_with_features(
            model, backbone, torch.where(masks, baseline, tensor)
        )

    flagged = is_out_of_distribution(*score(features.cpu().numpy(), stats), stats)
    return float(np.mean(flagged))


# ------------------------------------------------------------ localisation


def lung_mass_fraction(cam, mask):
    """(fraction of CAM mass inside the mask, fraction of area the mask covers).

    Both, always, and never the first on its own. Lungs are roughly a quarter of
    a chest radiograph, so a map that is uniform over the whole image already
    scores about 0.25 here. The ratio of the two is the only part that carries
    information, and `enrichment` below is what should be quoted.
    """
    cam = np.asarray(cam, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    if cam.shape != mask.shape:
        raise ValueError(f"CAM is {cam.shape} but the mask is {mask.shape}.")

    total = cam.sum()
    if total <= 0:
        return 0.0, float(mask.mean())
    return float(cam[mask].sum() / total), float(mask.mean())


def enrichment(mass_fraction, area_fraction):
    """Mass inside the lungs over what an uninformative map would put there.

    1.0 means the heatmap is no more concentrated on the lungs than a uniform
    map. Above 1.0 it is localising; below, it is attending to the margins.
    """
    if area_fraction <= 0:
        return None
    return float(mass_fraction / area_fraction)


def load_lung_mask(mask_path, size):
    """A binary lung mask resized to the model's input grid.

    Nearest-neighbour for the same reason src.mask_lungs uses it: the mask is
    binary and ships at a different resolution than the images, so interpolating
    fringes the border with values that are neither lung nor background.
    """
    from PIL import Image

    mask = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    return np.array(mask) > 127


# ------------------------------------------------------- attribution stability


def pairwise_cosine(vectors):
    """Mean pairwise cosine similarity between rows. NaN-free for zero rows."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if len(vectors) < 2:
        return None

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalised = vectors / np.clip(norms, 1e-12, None)
    similarity = normalised @ normalised.T

    upper = np.triu_indices(len(vectors), k=1)
    return float(similarity[upper].mean())


def stability(vectors, labels, top_k=DEFAULT_TOP_K):
    """Within-class agreement of SHAP vectors, against a cross-class control.

    The paper's "SHAP value stability across similar cases". Reported with the
    control because the number alone cannot be read: SHAP vectors are all
    w * (x - E[x]) for the same w, so they share a direction before any
    similarity between the images is considered, and within-class agreement is
    high for arithmetic reasons before it is high for interesting ones. Only the
    margin over the cross-class figure says anything about similar cases.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    labels = np.asarray(labels)

    within = {}
    for index, name in enumerate(CLASSES):
        group = vectors[labels == index]
        within[name] = {
            "images": int(len(group)),
            "mean_pairwise_cosine": pairwise_cosine(group),
            "mean_top_k_coverage": (
                float(np.mean([coverage(row, top_k) for row in group]))
                if len(group)
                else None
            ),
        }

    present = [value["mean_pairwise_cosine"] for value in within.values()
               if value["mean_pairwise_cosine"] is not None]

    return {
        "top_k": int(top_k),
        "per_class": within,
        "mean_within_class_cosine": float(np.mean(present)) if present else None,
        # The control: every pair, ignoring class. If this matches the within-
        # class number, "similar cases get similar explanations" is not a claim
        # this data supports.
        "all_pairs_cosine": pairwise_cosine(vectors),
    }


# -------------------------------------------------------------------- runner


def evaluate_image(
    model, backbone, tensor, steps=DEFAULT_STEPS, generator=None, ood_stats=None
):
    """Every spatial metric for one image. Returns a dict plus the CAM."""
    with GradCAM(model, target_layer(model, backbone)) as extractor:
        cam, class_idx, _ = extractor(tensor)

    ordered = perturbation_order(cam, generator)
    random_order = (generator or np.random.default_rng(0)).permutation(cam.size)

    deletion, insertion = curves(model, tensor, ordered, class_idx, steps)
    random_deletion, random_insertion = curves(
        model, tensor, random_order, class_idx, steps
    )

    result = {
        "predicted": CLASSES[class_idx],
        "deletion_auc": curve_auc(deletion),
        "insertion_auc": curve_auc(insertion),
        "random_deletion_auc": curve_auc(random_deletion),
        "random_insertion_auc": curve_auc(random_insertion),
    }

    if ood_stats is not None:
        result["deletion_ood_rate"] = ood_refusal_rate(
            model, backbone, tensor, ordered, ood_stats, steps
        )

    return result, cam, class_idx


def summarise(rows):
    """Mean of every numeric column, skipping the ones that were not computed."""
    keys = {key for row in rows for key, value in row.items()
            if isinstance(value, (int, float))}
    return {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in sorted(keys)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Score the explanations: fidelity, localisation, stability."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Images to score. Each one costs about 4 batched forward passes, "
        "so the whole split is rarely worth it.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--background", default=DEFAULT_BACKGROUND_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--masks",
        help="The original download, for lung masks. Without it the "
        "localisation numbers are skipped rather than guessed at.",
    )
    parser.add_argument(
        "--ood-stats",
        help="Fitted OOD statistics, to report how far the perturbed images "
        "drift out of distribution.",
    )
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    backbone = checkpoint["backbone"]
    print(f"checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']}, {backbone})")

    _, dataset = build_loader(
        Path(args.data_dir) / args.split, train=False, batch_size=1, num_workers=0
    )

    generator = np.random.default_rng(args.seed)
    limit = min(args.limit, len(dataset))
    # Sampled, not the first N. ImageFolder walks the class directories in
    # order, so dataset[:200] on this split is 200 COVID-19 images and nothing
    # else -- the summary would be a per-class number wearing a split-wide
    # label. Sorted afterwards only so the log reads in dataset order.
    selection = np.sort(generator.choice(len(dataset), size=limit, replace=False))

    sampled = {name: 0 for name in CLASSES}
    for index in selection:
        sampled[CLASSES[dataset.samples[index][1]]] += 1
    print(f"{args.split}: scoring {limit} of {len(dataset)} images {sampled}")

    ood_stats = None
    if args.ood_stats:
        from src.ood import load_stats

        ood_stats = load_stats(args.ood_stats, model)
        print(f"OOD statistics from {args.ood_stats}")

    background = None
    if Path(args.background).is_file():
        background = load_background(args.background, model)
        weight, bias = head_parameters(model, backbone)
        print(f"SHAP background from {args.background}")
    else:
        print(f"no SHAP background at {args.background} -- stability skipped")

    mask_dirs = {}
    if args.masks:
        from src.mask_lungs import find_mask_dirs

        mask_dirs = find_mask_dirs(Path(args.masks).expanduser())
        absent = [name for name in CLASSES if name not in mask_dirs]
        if absent:
            print(f"no masks for {absent} -- those images are skipped")

    rows = []
    localisation = []
    shap_vectors = []
    shap_labels = []

    for scored, index in enumerate(selection):
        index = int(index)
        tensor, label = dataset[index]
        tensor = tensor.unsqueeze(0).to(device)

        row, cam, class_idx = evaluate_image(
            model, backbone, tensor, args.steps, generator, ood_stats
        )
        row["actual"] = CLASSES[label]
        rows.append(row)

        if mask_dirs:
            entry = _localisation_for(dataset, index, mask_dirs, cam)
            if entry is not None:
                localisation.append(entry)

        if background is not None:
            with torch.no_grad():
                _, features = forward_with_features(model, backbone, tensor)
            values, _ = shap_values(
                features.cpu().numpy(), background["mean"], weight, bias
            )
            shap_vectors.append(values[0, class_idx])
            shap_labels.append(label)

        if (scored + 1) % 25 == 0:
            print(f"  {scored + 1}/{limit}")

    summary = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "images": limit,
        "sampled": sampled,
        "seed": args.seed,
        "steps": args.steps,
        "fidelity": summarise(rows),
    }

    print("\nexplanation fidelity (Grad-CAM)")
    print(f"  deletion AUC   {summary['fidelity']['deletion_auc']:.4f}  "
          f"(random {summary['fidelity']['random_deletion_auc']:.4f}) -- lower is better")
    print(f"  insertion AUC  {summary['fidelity']['insertion_auc']:.4f}  "
          f"(random {summary['fidelity']['random_insertion_auc']:.4f}) -- higher is better")
    if "deletion_ood_rate" in summary["fidelity"]:
        print(f"  deletion steps the OOD check refuses: "
              f"{summary['fidelity']['deletion_ood_rate']:.1%}")

    if localisation:
        mass = float(np.mean([row["mass_fraction"] for row in localisation]))
        area = float(np.mean([row["area_fraction"] for row in localisation]))
        summary["localisation"] = {
            "images": len(localisation),
            "mean_mass_fraction": mass,
            "mean_area_fraction": area,
            "enrichment": enrichment(mass, area),
        }
        print(f"\nlocalisation ({len(localisation)} images with masks)")
        print(f"  CAM mass inside the lungs   {mass:.3f}")
        print(f"  lungs as a share of the image {area:.3f}")
        print(f"  enrichment                  {enrichment(mass, area):.3f}  "
              f"(1.0 = no better than uniform)")
        print("  this is lung-field mass, NOT pathology IoU -- see the docstring")

    if shap_vectors:
        summary["stability"] = stability(shap_vectors, shap_labels, args.top_k)
        print(f"\nSHAP stability over {len(shap_vectors)} images")
        print(f"  within-class cosine  "
              f"{summary['stability']['mean_within_class_cosine']:.4f}")
        print(f"  all-pairs cosine     "
              f"{summary['stability']['all_pairs_cosine']:.4f}  (control)")
        covered = [
            row["mean_top_k_coverage"]
            for row in summary["stability"]["per_class"].values()
            if row["mean_top_k_coverage"] is not None
        ]
        if covered:
            print(f"  top-{args.top_k} coverage    {np.mean(covered):.1%} of the "
                  f"total movement")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"xai_fidelity_{args.split}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {path}")


def _localisation_for(dataset, index, mask_dirs, cam):
    """Lung-mask metrics for one image, or None when it has no mask.

    Missing masks are ordinary rather than exceptional -- prepare_data renames
    files and only some downloads ship masks at all -- so this skips and the
    caller reports how many images the number covers.
    """
    from src.mask_lungs import original_name

    path = Path(dataset.samples[index][0])
    class_name = CLASSES[dataset.samples[index][1]]
    if class_name not in mask_dirs:
        return None

    mask_path = mask_dirs[class_name] / original_name(path.name)
    if not mask_path.is_file():
        return None

    mask = load_lung_mask(mask_path, cam.shape[0])
    mass_fraction, area_fraction = lung_mass_fraction(cam, mask)
    return {"mass_fraction": mass_fraction, "area_fraction": area_fraction}


if __name__ == "__main__":
    main()
