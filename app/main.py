"""FastAPI service for the trained classifier.

Three endpoints: /health to check the model actually loaded, /predict for
probabilities, /explain for the Grad-CAM overlay as a PNG.

Every prediction carries a disclaimer, the same way the triage backend does.
This is a research model trained on public data; it is not a diagnostic device
and nothing it returns should be treated as a clinical finding. Sending the
number back without that context is how a demo turns into someone's decision.

/predict also reports whether the image resembles anything the model was
trained on, which the probabilities cannot say for themselves -- a fixed-class
softmax answers confidently no matter what it is handed. That check needs
statistics fitted by src.ood; without them the field is null, meaning the
question was not asked rather than answered in the negative.

    uvicorn app.main:app --port 8100 --reload
"""

import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from src.dataset import CLASSES, build_transforms
from src.gradcam_utils import explain_image
from src.model import forward_with_features, load_checkpoint, pick_device
from src.ood import DEFAULT_STATS_PATH, is_out_of_distribution, load_stats, score
from src.shap_utils import (
    DEFAULT_BACKGROUND_PATH,
    DEFAULT_TOP_K,
    coverage,
    head_parameters,
    load_background,
    shap_values,
    top_contributions,
)

CHECKPOINT_PATH = os.getenv("CXR_CHECKPOINT", "checkpoints/best.pt")
OOD_STATS_PATH = os.getenv("CXR_OOD_STATS", DEFAULT_STATS_PATH)
SHAP_BACKGROUND_PATH = os.getenv("CXR_SHAP_BACKGROUND", DEFAULT_BACKGROUND_PATH)

# 15 MB. Chest X-ray PNGs run a few hundred KB after downscaling; anything past
# this is either a mistake or someone probing the endpoint.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

DISCLAIMER = (
    "Research prototype trained on public datasets. Not a medical device, not "
    "a diagnosis, and not validated for clinical use. Any real concern about a "
    "chest X-ray belongs with a qualified radiologist or physician."
)

# Populated by the lifespan handler so the weights load once at startup rather
# than on the first request.
state = {
    "model": None,
    "backbone": None,
    "device": None,
    "checkpoint": None,
    "ood": None,
    "shap": None,
}


def _load_ood_stats(model):
    """The out-of-distribution statistics, or None with a reason printed.

    Missing statistics are not fatal -- the classifier still works without
    them -- but they are never silently substituted for. Every path that finds
    nothing usable leaves this None, and /predict then reports the check as
    unavailable rather than as passed.
    """
    try:
        return load_stats(OOD_STATS_PATH, model)
    except FileNotFoundError:
        print(
            f"no OOD stats at {OOD_STATS_PATH} -- /predict cannot tell whether "
            f"an image is a chest X-ray. Fit them with python -m src.ood"
        )
    except ValueError as error:
        # A mismatch means the stats describe a different feature space, so the
        # distances would be meaningless. Refusing them is the point.
        print(f"ignoring OOD stats at {OOD_STATS_PATH}: {error}")
    return None


def _load_shap_background(model):
    """The SHAP reference vector, or None with a reason printed.

    Same contract as _load_ood_stats, and the same reason for it: a background
    fitted against different weights describes a different feature space, so
    every contribution computed from it would be wrong with no symptom beyond a
    chart that looks a little odd. Missing or mismatched leaves this None, and
    /analyze then reports the SHAP block as null rather than as empty.
    """
    try:
        return load_background(SHAP_BACKGROUND_PATH, model)
    except FileNotFoundError:
        print(
            f"no SHAP background at {SHAP_BACKGROUND_PATH} -- /analyze will "
            f"return no attributions. Fit one with "
            f"python -m src.shap_utils --fit"
        )
    except ValueError as error:
        print(f"ignoring SHAP background at {SHAP_BACKGROUND_PATH}: {error}")
    return None


@asynccontextmanager
async def lifespan(app):
    device = pick_device()
    try:
        model, checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
    except FileNotFoundError:
        # Deliberately not fatal. The service should still start so /health can
        # say what is wrong, instead of crash-looping with a stack trace.
        print(f"no checkpoint at {CHECKPOINT_PATH} -- /predict will return 503")
        yield
        return

    state.update(
        model=model,
        backbone=checkpoint["backbone"],
        device=device,
        checkpoint=checkpoint,
        ood=_load_ood_stats(model),
        shap=_load_shap_background(model),
    )
    print(f"loaded {CHECKPOINT_PATH} ({checkpoint['backbone']}) on {device}")
    if state["ood"] is not None:
        print(f"loaded OOD stats from {OOD_STATS_PATH}")
    if state["shap"] is not None:
        print(f"loaded SHAP background from {SHAP_BACKGROUND_PATH}")
    yield


app = FastAPI(title="Chest X-ray Classifier", lifespan=lifespan)

# /explain returns a PNG and puts the labels in headers, so a browser can point
# an <img> straight at it and still read what the picture says. That only works
# cross-origin if the headers are named here: allow_headers covers the request,
# expose_headers covers the response, and without it a browser client sees
# nothing but content-type and content-length. The frontend is served from a
# different port, so this is the normal case rather than the exotic one.
EXPOSED_HEADERS = [
    "X-Prediction",
    "X-Confidence",
    "X-Explained-Class",
    "X-Explained-Confidence",
    "X-Disclaimer",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=EXPOSED_HEADERS,
)


def _require_model():
    if state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"No model loaded. Train one first, or set CXR_CHECKPOINT "
            f"(currently {CHECKPOINT_PATH!r}).",
        )


async def _read_image(upload):
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is {len(raw) // 1024} KB, limit is {MAX_UPLOAD_BYTES // 1024} KB.",
        )
    try:
        return Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Could not read that file as an image. PNG, JPEG or BMP.",
        )


def _ood_fields(features):
    """Whether the image looks like anything the model was trained on.

    All three are null when no statistics are loaded. Null means the check did
    not run -- it is not a pass, and a caller that treats it as one has the
    same bug this endpoint used to have.
    """
    stats = state["ood"]
    if stats is None:
        return {
            "out_of_distribution": None,
            "ood_score": None,
            "ood_threshold": None,
        }

    scores, nearest = score(features.cpu().numpy(), stats)
    return {
        "out_of_distribution": bool(is_out_of_distribution(scores, nearest, stats)[0]),
        "ood_score": round(float(scores[0]), 1),
        # The cutoff actually applied, which is the one for the class this
        # image is nearest to rather than a single global number.
        "ood_threshold": round(float(stats["thresholds"][int(nearest[0])]), 1),
    }


@app.get("/health")
def health():
    loaded = state["model"] is not None
    stats = state["ood"]
    return {
        "status": "ok" if loaded else "no_model",
        "checkpoint": CHECKPOINT_PATH,
        "model_loaded": loaded,
        "backbone": state["backbone"],
        "device": str(state["device"]) if state["device"] else None,
        "classes": CLASSES,
        "val_metrics": state["checkpoint"]["metrics"] if loaded else None,
        # Reported separately from the model, because the service runs happily
        # without these and a caller has no other way to find out that every
        # out_of_distribution field it is reading back is null.
        "ood_stats": OOD_STATS_PATH,
        "ood_stats_loaded": stats is not None,
        "ood_percentile": stats["percentile"] if stats else None,
        # Reported for the same reason as the OOD statistics: /analyze serves
        # happily without a background and a caller has no other way to find
        # out that every shap block it is reading back is null.
        "shap_background": SHAP_BACKGROUND_PATH,
        "shap_background_loaded": state["shap"] is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    _require_model()
    image = await _read_image(file)

    tensor = build_transforms(train=False)(image.convert("L"))
    tensor = tensor.unsqueeze(0).to(state["device"])

    # One pass for both answers: the probabilities come off the logits, the
    # distance off the features feeding the head.
    with torch.no_grad():
        logits, features = forward_with_features(
            state["model"], state["backbone"], tensor
        )
    probabilities = torch.softmax(logits, dim=1)[0].cpu()

    ranked = sorted(
        zip(CLASSES, probabilities.tolist()), key=lambda pair: pair[1], reverse=True
    )
    label, confidence = ranked[0]

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "probabilities": {name: round(value, 4) for name, value in ranked},
        # A softmax over a fixed class list always sums to 1, so it names a class for any image
        # at all. This flag catches only the case where it is visibly torn
        # between the three; it does not catch a confident answer to a question
        # that was never asked, which is what out_of_distribution is for.
        "low_confidence": confidence < 0.6,
        **_ood_fields(features),
        "disclaimer": DISCLAIMER,
    }


@app.post("/explain")
async def explain(file: UploadFile = File(...), class_name: str = None):
    """Grad-CAM overlay as a PNG. Pass class_name to explain a specific class."""
    _require_model()

    if class_name is not None and class_name not in CLASSES:
        raise HTTPException(
            status_code=400, detail=f"class_name must be one of {CLASSES}"
        )

    image = await _read_image(file)
    class_idx = CLASSES.index(class_name) if class_name else None

    overlay, explained_label, probabilities = explain_image(
        state["model"], state["backbone"], image, state["device"], class_idx
    )

    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")

    # The prediction rides along in headers so a client can render the image
    # directly in an <img> tag and still read the label.
    #
    # Prediction and explanation are reported separately because they come
    # apart whenever class_name is passed: the heatmap then answers "why not
    # pneumonia?" while the model's actual call is still NORMAL. Reporting the
    # requested class as the prediction, next to the top class's confidence,
    # would put two different classes' numbers under one label.
    top_index = int(probabilities.argmax())
    explained_index = CLASSES.index(explained_label)

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "X-Prediction": CLASSES[top_index],
            "X-Confidence": f"{float(probabilities[top_index]):.4f}",
            "X-Explained-Class": explained_label,
            "X-Explained-Confidence": f"{float(probabilities[explained_index]):.4f}",
            "X-Disclaimer": DISCLAIMER,
        },
    )


def _shap_fields(features, class_idx, top_k):
    """The SHAP block for one image, or None when no background is loaded.

    None means the attribution was not computed. It is not an empty
    explanation, and a caller that renders it as "no features contributed" has
    the same bug the out_of_distribution field is written to avoid.
    """
    background = state["shap"]
    if background is None:
        return None

    weight, bias = head_parameters(state["model"], state["backbone"])
    values, base = shap_values(
        features.cpu().numpy(), background["mean"], weight, bias
    )
    selected = values[0, class_idx]
    total = float(selected.sum())

    return {
        "base_value": round(float(base[class_idx]), 4),
        "sum_of_contributions": round(total, 4),
        # base + sum is the logit, exactly -- the Shapley efficiency axiom. Sent
        # so a client can check the decomposition it is being shown rather than
        # taking the bars on trust.
        "logit": round(float(base[class_idx]) + total, 4),
        "features": [
            {"feature": index, "value": round(value, 4)}
            for index, value in top_contributions(selected, top_k)
        ],
        "feature_count": int(selected.size),
        # What fraction of the movement the listed features account for. On this
        # model the top 15 of 512 come to about 15%, so a client that presents
        # the bars as "the reason" is overstating them by a wide margin.
        "coverage": round(coverage(selected, top_k), 4),
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    class_name: str = None,
    top_k: int = DEFAULT_TOP_K,
):
    """Prediction, Grad-CAM and SHAP in one JSON response.

    /predict and /explain remain what they were -- one returns numbers, the
    other returns a PNG, and a client that only wants the picture should not be
    made to base64-decode it. This endpoint exists because the two explanations
    are meant to be read together: the heatmap says where, the attributions say
    how much, and delivering them in separate round trips invites a caller to
    show one and drop the other.

    The overlay is base64 rather than a URL because nothing here stores images.
    An endpoint handing back a link would need somewhere to put the upload, and
    a research prototype that starts retaining chest X-rays has acquired a
    problem it does not want.
    """
    _require_model()

    if class_name is not None and class_name not in CLASSES:
        raise HTTPException(
            status_code=400, detail=f"class_name must be one of {CLASSES}"
        )
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1.")

    image = await _read_image(file)
    requested_idx = CLASSES.index(class_name) if class_name else None

    tensor = build_transforms(train=False)(image.convert("L"))
    tensor = tensor.unsqueeze(0).to(state["device"])

    with torch.no_grad():
        logits, features = forward_with_features(
            state["model"], state["backbone"], tensor
        )
    probabilities = torch.softmax(logits, dim=1)[0].cpu()

    predicted_idx = int(probabilities.argmax())
    explained_idx = requested_idx if requested_idx is not None else predicted_idx

    overlay, _, _ = explain_image(
        state["model"], state["backbone"], image, state["device"], explained_idx
    )
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")

    ranked = sorted(
        zip(CLASSES, probabilities.tolist()), key=lambda pair: pair[1], reverse=True
    )

    return {
        "prediction": CLASSES[predicted_idx],
        "confidence": round(float(probabilities[predicted_idx]), 4),
        "probabilities": {name: round(value, 4) for name, value in ranked},
        "low_confidence": float(probabilities[predicted_idx]) < 0.6,
        # Reported separately from the prediction for the reason /explain does
        # it: when class_name is passed the heatmap and the attributions answer
        # "why not X" while the model's own call is still something else.
        "explained_class": CLASSES[explained_idx],
        "explained_confidence": round(float(probabilities[explained_idx]), 4),
        "gradcam_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "shap": _shap_fields(features, explained_idx, top_k),
        **_ood_fields(features),
        "disclaimer": DISCLAIMER,
    }


# Serving the page from this process is off by default and opt-in through the
# environment. Locally the two run separately -- the API on 8100 and the page on
# 5501 -- so either can be restarted without the other, which is worth having
# while editing. A single-container deployment has nowhere to put a second
# server, and there this is the whole point.
#
# Mounted last, and only last. A mount at "/" matches every path under it, so
# registering it before the routes above would swallow /predict and /health.
if os.getenv("CXR_SERVE_FRONTEND"):
    from fastapi.staticfiles import StaticFiles

    FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
        print(f"serving the frontend from {FRONTEND_DIR}")
    else:
        print(f"CXR_SERVE_FRONTEND is set but {FRONTEND_DIR} does not exist")
