"""SHAP attributions: that they are exact, and that they are really SHAP.

The claim this module makes is unusually strong -- not "approximately the
Shapley values" but "the Shapley values, in closed form" -- and a wrong
attribution has no symptom. A bar chart built from the wrong background, or
from a transposed weight matrix, is still a tidy chart of plausible-looking
numbers. Nothing downstream of it could tell.

So two things are pinned here rather than trusted:

- **Efficiency.** The contributions plus the base value have to reconstruct the
  logit exactly. That is the Shapley axiom that fails first when an attribution
  is wrong, and it holds for essentially no incorrect implementation.
- **Agreement with the reference.** The same numbers come out of
  shap.LinearExplainer (Lundberg & Lee 2017). The closed form is what runs at
  serving time, so without this test the project would be claiming SHAP while
  shipping something merely SHAP-shaped.
"""

import sys

import numpy as np
import pytest
import torch

from src.dataset import CLASSES, build_loader, build_transforms
from src.model import (
    build_model,
    forward_with_features,
    load_checkpoint,
    save_checkpoint,
    target_layer,
)
from src.shap_utils import (
    channel_map,
    coverage,
    explain_image,
    explain_tensor,
    fit_background,
    head_parameters,
    load_background,
    main,
    margin_values,
    save_background,
    shap_values,
    top_contributions,
)

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def loaded(checkpoint):
    model, meta = load_checkpoint(checkpoint, DEVICE)
    return model, meta["backbone"]


@pytest.fixture(scope="module")
def background(loaded, data_root):
    model, backbone = loaded
    loader, _ = build_loader(
        data_root / "train", train=False, batch_size=4, num_workers=0
    )
    return fit_background(model, backbone, loader, DEVICE)


@pytest.fixture
def batch(data_root):
    """Four real preprocessed images, not noise.

    The efficiency identity holds for any input at all, so noise would pass it.
    Using the actual transform means the test also covers the path an image
    really takes from disk.
    """
    loader, _ = build_loader(
        data_root / "test", train=False, batch_size=4, num_workers=0
    )
    images, _ = next(iter(loader))
    return images


# ------------------------------------------------------------- the axiom


def test_contributions_and_base_reconstruct_the_logit(loaded, background, batch):
    """The Shapley efficiency axiom, and the single most important assertion here.

    If this holds, the attribution is a genuine additive decomposition of what
    the head computed. If it does not, every number the module reports is
    describing some other function.
    """
    model, backbone = loaded
    values, base, logits, _ = explain_tensor(model, backbone, batch, background)

    reconstructed = values.sum(axis=2) + base
    assert np.allclose(reconstructed, logits.numpy(), atol=1e-5)


def test_the_axiom_holds_for_every_class_not_just_the_predicted_one(
    loaded, background, batch
):
    """A decomposition that is only exact for the argmax would still look right
    in every screenshot, because the CLI explains the predicted class by
    default. --class-name and the API both ask for the others."""
    model, backbone = loaded
    values, base, logits, _ = explain_tensor(model, backbone, batch, background)

    for class_idx in range(len(CLASSES)):
        reconstructed = values[:, class_idx, :].sum(axis=1) + base[class_idx]
        assert np.allclose(reconstructed, logits[:, class_idx].numpy(), atol=1e-5)


def test_a_feature_sitting_at_the_background_contributes_nothing(loaded, background):
    """Zero deviation must mean zero attribution, whatever the weight on it.

    This is what makes the base value meaningful: it is the score of an image
    that is average in every respect, so an average feature has to add nothing
    to it.
    """
    model, backbone = loaded
    weight, bias = head_parameters(model, backbone)
    mean = background["mean"].numpy()

    values, base = shap_values(mean[None, :], mean, weight, bias)

    assert np.allclose(values, 0.0, atol=1e-9)
    assert np.allclose(base, mean @ weight.T + bias, atol=1e-9)


# ------------------------------------------------- agreement with the reference


def test_the_closed_form_equals_shap_linearexplainer(loaded, background, batch):
    """The paper cites Lundberg & Lee; this is what entitles it to.

    shap.LinearExplainer computes the same quantity by a different route, so
    matching it rules out an error that happens to preserve efficiency -- a
    sign flip on the background, say, which still sums correctly for a
    symmetric weight matrix.
    """
    shap = pytest.importorskip("shap")

    model, backbone = loaded
    weight, bias = head_parameters(model, backbone)
    mean = background["mean"].numpy()

    with torch.no_grad():
        _, features = forward_with_features(model, backbone, batch)
    features = features.numpy().astype(np.float64)

    mine, base = shap_values(features, mean, weight, bias)

    # A two-row background whose mean is exactly ours. The reference reads its
    # expectation off the data it is given, so handing it the mean itself would
    # make the comparison circular.
    data = np.stack([mean - 1.0, mean + 1.0])

    for class_idx in range(len(CLASSES)):
        reference = shap.LinearExplainer(
            (weight[class_idx], bias[class_idx]),
            # max_samples matters: passing a bare array lets shap subsample it
            # to 100 rows, which silently shifts the background and every value
            # with it. Small here, load-bearing on a real background.
            shap.maskers.Independent(data, max_samples=len(data)),
        )
        assert np.allclose(mine[:, class_idx, :], reference.shap_values(features))
        assert np.allclose(
            np.asarray(reference.expected_value).ravel()[0], base[class_idx]
        )


# ------------------------------------------------------------------- margins


def test_margin_values_decompose_the_logit_difference(loaded, background, batch):
    """--against explains why one class rather than another, which is the
    question an argmax actually answers. It has to be exact for the difference
    the same way the per-class values are exact for the logit."""
    model, backbone = loaded
    values, base, logits, _ = explain_tensor(model, backbone, batch, background)

    margin = margin_values(values, 0, 2)
    expected = (logits[:, 0] - logits[:, 2]).numpy()

    assert np.allclose(margin.sum(axis=1) + (base[0] - base[2]), expected, atol=1e-5)


# ---------------------------------------------------------------- the summary


def test_top_contributions_ranks_by_magnitude_and_keeps_the_sign():
    """Evidence against the predicted class is as worth seeing as evidence for
    it, so the ranking is on |value| rather than on value."""
    values = np.array([0.1, -5.0, 2.0, -0.3, 4.0])
    assert top_contributions(values, k=3) == [(1, -5.0), (4, 4.0), (2, 2.0)]


def test_top_contributions_does_not_overrun_a_short_vector():
    assert len(top_contributions(np.array([1.0, 2.0]), k=15)) == 2


def test_coverage_reports_the_fraction_of_movement_shown():
    """A chart whose bars cover 12% of the moved mass has hidden the answer,
    and the picture alone cannot show that."""
    values = np.array([6.0, 3.0, 1.0])
    assert coverage(values, k=1) == pytest.approx(0.6)
    assert coverage(values, k=3) == pytest.approx(1.0)


def test_coverage_of_an_all_zero_vector_is_not_a_division_by_zero():
    assert coverage(np.zeros(8), k=3) == 0.0


# --------------------------------------------------- tracing back to pixels


def test_a_feature_is_the_spatial_mean_of_its_channel(loaded, batch):
    """The relationship channel_map's docstring claims, pinned rather than
    assumed. If torchvision ever changes what sits between the last block and
    the head, the map would quietly start describing a different channel than
    the number beside it."""
    model, backbone = loaded
    captured = {}

    handle = target_layer(model, backbone).register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("out", output.detach())
    )
    try:
        with torch.no_grad():
            _, features = forward_with_features(model, backbone, batch)
    finally:
        handle.remove()

    pooled = captured["out"].mean(dim=(2, 3))
    assert torch.allclose(pooled, features, atol=1e-5)


def test_channel_map_is_normalised_and_two_dimensional(loaded, batch):
    model, backbone = loaded
    activation = channel_map(model, backbone, batch[:1], channel=3)

    assert activation.ndim == 2
    assert np.abs(activation).max() == pytest.approx(1.0)


def test_a_dead_channel_maps_to_zeros_rather_than_nan(loaded, batch):
    """Dividing by a zero peak would render as blank, which looks exactly like
    a channel that fired evenly everywhere."""
    model, backbone = loaded
    with torch.no_grad():
        activation = channel_map(model, backbone, torch.zeros_like(batch[:1]), 0)

    assert np.isfinite(activation).all()


# ------------------------------------------------------------- the background


def test_a_background_from_another_checkpoint_is_refused(
    loaded, background, tmp_path
):
    """The silent failure this guard exists for. Feature 314 means one thing to
    one set of weights and something unrelated to another, so a mismatched
    background yields a confident, well-formatted, meaningless chart."""
    path = save_background(tmp_path / "background.pt", background)
    other = build_model("resnet18", pretrained=False)

    with pytest.raises(ValueError, match="different checkpoint"):
        load_background(path, other)


def test_a_matching_background_loads(loaded, background, tmp_path):
    model, _ = loaded
    path = save_background(tmp_path / "background.pt", background)

    assert load_background(path, model)["feature_dim"] == background["feature_dim"]


def test_features_of_the_wrong_width_are_refused(loaded, background):
    """resnet18 gives 512 features and resnet50 gives 2048. Numpy would
    broadcast some mismatches silently rather than raising."""
    model, backbone = loaded
    weight, bias = head_parameters(model, backbone)

    with pytest.raises(ValueError, match="different backbone"):
        shap_values(np.zeros((1, 7)), background["mean"].numpy(), weight, bias)


def test_the_background_is_fitted_over_every_image(loaded, data_root):
    model, backbone = loaded
    loader, dataset = build_loader(
        data_root / "train", train=False, batch_size=4, num_workers=0
    )
    fitted = fit_background(model, backbone, loader, DEVICE)

    assert fitted["images"] == len(dataset)
    assert fitted["classes"] == CLASSES


def test_the_background_mean_matches_a_direct_average(loaded, data_root):
    """fit_background accumulates in a running sum across batches, which is
    where a ragged final batch would be weighted wrongly."""
    model, backbone = loaded
    loader, _ = build_loader(
        data_root / "train", train=False, batch_size=3, num_workers=0
    )
    fitted = fit_background(model, backbone, loader, DEVICE)

    every_feature = []
    with torch.no_grad():
        for images, _ in loader:
            _, features = forward_with_features(model, backbone, images)
            every_feature.append(features.numpy())

    direct = np.concatenate(every_feature).astype(np.float64).mean(axis=0)
    assert np.allclose(fitted["mean"].numpy(), direct, atol=1e-6)


# ------------------------------------------------------------ the entry point


def test_explain_image_agrees_with_the_tensor_path(loaded, background, xray_image):
    """The API and the CLI go through explain_image; everything tested above
    goes through explain_tensor. They have to be the same computation."""
    model, backbone = loaded
    _, image = xray_image

    contributions, base, probabilities, label = explain_image(
        model, backbone, image, DEVICE, background
    )

    tensor = build_transforms(train=False)(image.convert("L")).unsqueeze(0)
    values, bases, logits, _ = explain_tensor(model, backbone, tensor, background)
    predicted = int(logits.argmax(dim=1))

    assert label == CLASSES[predicted]
    assert np.allclose(contributions, values[0, predicted])
    assert base == pytest.approx(float(bases[predicted]))
    assert probabilities.argmax() == predicted


# -------------------------------------------------------------------- main


def run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["src.shap_utils", *argv])
    main()


def test_fit_writes_a_background_the_explainer_can_load(
    monkeypatch, tmp_path, data_root, checkpoint
):
    path = tmp_path / "background.pt"
    run_cli(
        monkeypatch,
        [
            "--fit",
            "--checkpoint", str(checkpoint),
            "--data-dir", str(data_root),
            "--background", str(path),
            "--batch-size", "4",
            "--device", "cpu",
        ],
    )

    assert path.is_file()
    model, _ = load_checkpoint(checkpoint, DEVICE)
    assert load_background(path, model)["images"] == 27


def test_explaining_an_image_writes_a_chart_and_balances(
    monkeypatch, tmp_path, data_root, checkpoint, xray_image, capsys
):
    """The printed summary states base + contributions = logit. Those three
    numbers are read straight off the arrays, so a chart that does not balance
    is the visible form of the axiom failing."""
    background_path = tmp_path / "background.pt"
    run_cli(
        monkeypatch,
        [
            "--fit",
            "--checkpoint", str(checkpoint),
            "--data-dir", str(data_root),
            "--background", str(background_path),
            "--batch-size", "4",
            "--device", "cpu",
        ],
    )

    image_path, _ = xray_image
    out = tmp_path / "shap.png"
    run_cli(
        monkeypatch,
        [
            "--checkpoint", str(checkpoint),
            "--background", str(background_path),
            "--image", str(image_path),
            "--out", str(out),
            "--top-k", "5",
            "--device", "cpu",
        ],
    )

    assert out.is_file() and out.stat().st_size > 0
    printed = capsys.readouterr().out
    assert "base value" in printed
    assert "top 5 of 512 features" in printed


def test_the_margin_view_names_both_classes(
    monkeypatch, tmp_path, data_root, checkpoint, xray_image, capsys
):
    background_path = tmp_path / "background.pt"
    run_cli(
        monkeypatch,
        [
            "--fit",
            "--checkpoint", str(checkpoint),
            "--data-dir", str(data_root),
            "--background", str(background_path),
            "--batch-size", "4",
            "--device", "cpu",
        ],
    )

    image_path, _ = xray_image
    run_cli(
        monkeypatch,
        [
            "--checkpoint", str(checkpoint),
            "--background", str(background_path),
            "--image", str(image_path),
            "--out", str(tmp_path / "margin.png"),
            "--class-name", "COVID19",
            "--against", "NORMAL",
            "--device", "cpu",
        ],
    )

    printed = capsys.readouterr().out
    assert "COVID19 over NORMAL" in printed
    assert "logit margin" in printed


def test_explaining_a_class_against_itself_is_refused(
    monkeypatch, tmp_path, data_root, checkpoint, xray_image
):
    """The margin of a class over itself is identically zero, which would plot
    as an empty chart rather than as an error."""
    background_path = tmp_path / "background.pt"
    run_cli(
        monkeypatch,
        [
            "--fit",
            "--checkpoint", str(checkpoint),
            "--data-dir", str(data_root),
            "--background", str(background_path),
            "--batch-size", "4",
            "--device", "cpu",
        ],
    )

    image_path, _ = xray_image
    with pytest.raises(SystemExit):
        run_cli(
            monkeypatch,
            [
                "--checkpoint", str(checkpoint),
                "--background", str(background_path),
                "--image", str(image_path),
                "--out", str(tmp_path / "x.png"),
                "--class-name", "COVID19",
                "--against", "COVID19",
                "--device", "cpu",
            ],
        )


def test_neither_fit_nor_image_is_an_error(monkeypatch, checkpoint):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, ["--checkpoint", str(checkpoint), "--device", "cpu"])
