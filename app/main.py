"""FastAPI service for the trained classifier.

Three endpoints: /health to check the model actually loaded, /predict for
probabilities, /explain for the Grad-CAM overlay as a PNG.

Every prediction carries a disclaimer, the same way the triage backend does.
This is a research model trained on public data; it is not a diagnostic device
and nothing it returns should be treated as a clinical finding. Sending the
number back without that context is how a demo turns into someone's decision.

    uvicorn app.main:app --port 8100 --reload
"""

import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from src.dataset import CLASSES, build_transforms
from src.gradcam_utils import explain_image
from src.model import load_checkpoint, pick_device

CHECKPOINT_PATH = os.getenv("CXR_CHECKPOINT", "checkpoints/best.pt")

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
state = {"model": None, "backbone": None, "device": None, "checkpoint": None}


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
    )
    print(f"loaded {CHECKPOINT_PATH} ({checkpoint['backbone']}) on {device}")
    yield


app = FastAPI(title="Chest X-ray Classifier", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/health")
def health():
    loaded = state["model"] is not None
    return {
        "status": "ok" if loaded else "no_model",
        "checkpoint": CHECKPOINT_PATH,
        "model_loaded": loaded,
        "backbone": state["backbone"],
        "device": str(state["device"]) if state["device"] else None,
        "classes": CLASSES,
        "val_metrics": state["checkpoint"]["metrics"] if loaded else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    _require_model()
    image = await _read_image(file)

    tensor = build_transforms(train=False)(image.convert("L"))
    tensor = tensor.unsqueeze(0).to(state["device"])

    with torch.no_grad():
        probabilities = torch.softmax(state["model"](tensor), dim=1)[0].cpu()

    ranked = sorted(
        zip(CLASSES, probabilities.tolist()), key=lambda pair: pair[1], reverse=True
    )
    label, confidence = ranked[0]

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "probabilities": {name: round(value, 4) for name, value in ranked},
        # A 3-class softmax always sums to 1, so it will name a class even for
        # a photo of a cat. Low confidence is the only signal the caller has
        # that the image was out of distribution.
        "low_confidence": confidence < 0.6,
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
