# Chest X-ray classifier

A three-class CNN over chest radiographs: **NORMAL**, **PNEUMONIA**, **COVID19**.

It lives beside the triage app but does not import from it and is not wired into
it. The triage backend reads symptom text; this reads an image. Keeping them
apart means the backend does not have to carry a multi-gigabyte torch install.

## What this is not

This is a research prototype trained on public datasets. It is not a medical
device, it has not been validated on any clinical population, and a number it
returns is not a finding. Three specific reasons to distrust it, beyond the
usual:

- **The public COVID-19 X-ray sets are assembled from different sources per
  class.** COVID-19 images often come from one publication's collection and the
  normal images from another hospital entirely. A model can separate them by
  scanner, exposure or burned-in annotation and never look at a lung. This is a
  documented failure of published COVID-19 X-ray models, not a hypothetical.
  Run Grad-CAM before believing any score.
- **The label is not the disease.** These labels came from whoever assembled
  the dataset, by varying and mostly undocumented criteria — some RT-PCR
  confirmed, some radiologist-read, some neither.
- **The class balance is not the real prevalence.** Nothing here estimates how
  likely a given patient is to have anything.

Grad-CAM is included for exactly this reason. It is not decoration.

## Layout

```
src/dataset.py       loaders, transforms, the two imbalance corrections
src/model.py         backbone + 3-class head, checkpoint save/load
src/train.py         fine-tuning loop, selects on macro F1
src/evaluate.py      per-class report and confusion matrix
src/gradcam_utils.py heatmaps, and a CLI for one image
src/prepare_data.py  normalise a download into data/{train,val,test}/CLASS/
src/synth_data.py    drawn stand-in images, for testing the pipeline
app/main.py          FastAPI service: /health, /predict, /explain
tests/               pytest suite, no dataset and no network needed
```

## Setup

Shares one `smt` venv with the sibling `smart-healthcare-triage` checkout
rather than standing up a second one, since torch is the same multi-gigabyte
install either way. It lives **outside** both project folders, at `~/venvs/smt`.
Both projects sit under OneDrive, and a 4.8 GB venv has no business in a synced
folder: it rebuilds from `requirements.txt` in one command, it is thousands of
small files that sync slowly, and it is path- and platform-specific enough that
a synced copy would not run on another machine anyway.

Install torch from the index that matches your hardware **first** — see the
header of `requirements.txt`. On this machine (RTX 5070 Ti Laptop, Blackwell /
sm_120) that is the CUDA 12.8 build:

```bash
~/venvs/smt/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then the rest:

```bash
~/venvs/smt/Scripts/python.exe -m pip install -r requirements.txt
```

Every command below runs from `chest-xray-classifier/`. The interpreter is
called by path rather than by activating the venv, which keeps these commands
byte-identical in PowerShell and in bash — `~` expands in both.

Always `python -m <module>`, never the bare `pytest` / `uvicorn` / `kaggle`
launchers sitting in `Scripts/`. Those compile the interpreter's absolute path
in at install time and broke when the venv was relocated out of OneDrive. The
`-m` form does not care where the venv lives.

## Quick start, with no download

`src/synth_data.py` draws stand-in images so the pipeline can be exercised
before committing to a several-gigabyte download:

```bash
~/venvs/smt/Scripts/python.exe -m src.synth_data --out data --per-class 120
```

They are ellipses and Gaussian blobs with a per-class pattern drawn to be
learnable. Training on them proves gradients flow and the loaders are wired up
correctly. **It proves nothing about pneumonia.** Do not report a number that
came from these.

## Real data

Neither dataset is redistributed here; both need a Kaggle account.

**COVID-19 Radiography Database** — 21k images, the most widely used.
`kaggle datasets download -d tawsifurrahman/covid19-radiography-database`

**Chest X-ray (COVID-19 & Pneumonia)** — smaller, already split.
`kaggle datasets download -d prashant268/chest-xray-covid19-pneumonia`

Unzip anywhere, then normalise it into the layout the loaders expect:

```bash
~/venvs/smt/Scripts/python.exe -m src.prepare_data --source ~/Downloads/COVID-19_Radiography_Dataset --out data
```

`prepare_data` handles the naming differences between the two ("COVID" vs
"COVID19", "Viral Pneumonia" vs "PNEUMONIA") and skips `Lung_Opacity`, which is
a broader finding than pneumonia and is not one of our three classes.

It also **keeps one patient's images in a single split**. The Kermany pneumonia
images are named `person1_virus_6`, `person1_bacteria_1` and so on — several
films of the same chest. Split those at random and the same patient appears in
train and test, which inflates the test score by several points. This is why
the published train/test split is pooled and redone rather than used as-is.

## Train

```bash
~/venvs/smt/Scripts/python.exe -m src.train --epochs 15 --freeze-backbone --out checkpoints/stage1.pt
~/venvs/smt/Scripts/python.exe -m src.train --epochs 25 --lr 1e-4 --resume checkpoints/stage1.pt --out checkpoints/best.pt
```

The second stage lands on `checkpoints/best.pt`, which is where `src.evaluate`,
`src.gradcam_utils` and the API all look by default. `--out` is spelled out
above only because stage 1 names its own file and the asymmetry reads as though
stage 2 goes somewhere unstated; it is the default either way.

Head first, then unfreeze. Fine-tuning a whole resnet against a few hundred
images per class mostly memorises them.

`--resume` is what joins the two, and leaving it off is the quiet way to get
this wrong: the second command would otherwise rebuild from ImageNet weights
and throw the first one's epochs away, which looks identical in the logs. It
restores weights only — the optimizer and the LR schedule start clean, since
AdamW moments gathered while the backbone was frozen say nothing about the
parameters the second stage unfreezes.

Write the stages to separate files. `--resume` refuses to run when `--out`
names the same path, rather than consuming the checkpoint it started from.

Selection is on **macro F1, not accuracy**. With COVID-19 at a tenth of the
pneumonia count, a model that never predicts it can still post a high accuracy,
and selecting on accuracy would faithfully keep that model.

`--imbalance` picks the correction: `sampler` (default, oversample rare classes)
or `loss` (weight the loss). Use one. Both together over-corrects and the model
starts crying wolf.

## Evaluate

```bash
~/venvs/smt/Scripts/python.exe -m src.evaluate --checkpoint checkpoints/best.pt --split test
```

Read the per-class recall, not the accuracy. A model at 94% accuracy that
recalls 60% of COVID-19 cases is useless for the thing you would want it for,
and only the confusion matrix shows that. Writes
`reports/confusion_test.png` and `reports/metrics_test.json`.

## Grad-CAM

```bash
~/venvs/smt/Scripts/python.exe -m src.gradcam_utils --checkpoint checkpoints/best.pt --image data/test/PNEUMONIA/00001_person1_virus_6.jpeg --out cam.png
```

`--class-name` explains a class other than the predicted one, which is how you
ask why the model did *not* say pneumonia.

Check a handful of correct predictions before trusting a score. If the heat sits
on a corner marker or outside the lungs, the model found a shortcut.

## Serve

```bash
~/venvs/smt/Scripts/python.exe -m uvicorn app.main:app --port 8100
```

- `GET /health` — whether weights actually loaded, and the val metrics they
  scored. Starts and reports `no_model` rather than crash-looping when there is
  no checkpoint.
- `POST /predict` — multipart `file`. Returns the ranked distribution, plus
  `low_confidence`, which is the only signal a caller gets that the image was
  out of distribution — a 3-class softmax names a class for a photo of a cat.
- `POST /explain` — multipart `file`, optional `class_name`. Returns the overlay
  PNG. `X-Prediction` is what the model called it; `X-Explained-Class` is what
  the heatmap answers for. They differ whenever `class_name` is passed.

Port 8100, so it does not collide with the triage backend on 8000.

## Tests

```bash
~/venvs/smt/Scripts/python.exe -m pytest tests -q
```

No dataset, no network, no pretrained download — synthetic images and untrained
weights throughout.
