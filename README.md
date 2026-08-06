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
  class.** A model can separate them by scanner, exposure or burned-in
  annotation and never look at a lung. This is not a hypothetical here, and it
  is not a risk this repo merely warns about — it was measured. Every COVID-19
  image in the Radiography Database comes from BIMCV, Eurorad, SIRM or GitHub;
  every Normal and Viral Pneumonia image comes from Kaggle. Zero overlap. The
  classes are perfectly separable by provenance before any lung is examined.
  See [Results](#results). Run Grad-CAM before believing any score. Scoring
  against a second download does not settle it either — that dataset turned out
  to share 23.9% of its images with this one's training split, and the rest
  comes from the same public archives.
- **The label is not the disease.** These labels came from whoever assembled
  the dataset, by varying and mostly undocumented criteria — some RT-PCR
  confirmed, some radiologist-read, some neither.
- **The class balance is not the real prevalence.** Nothing here estimates how
  likely a given patient is to have anything.
- **Three classes is not every finding, and NORMAL does not mean clear.** Shown
  600 `Lung_Opacity` films from this same dataset — real chest X-rays with a
  finding that is not one of the three — it called them NORMAL 94.2% of the
  time at a mean confidence of 0.972. The out-of-distribution check catches
  only a third of them. See [Is it even a chest X-ray?](#is-it-even-a-chest-x-ray).

Grad-CAM is included for exactly this reason. It is not decoration.

## Layout

```
src/dataset.py         loaders, transforms, the two imbalance corrections
src/model.py           backbone + 3-class head, checkpoint save/load
src/train.py           fine-tuning loop, selects on macro F1
src/evaluate.py        per-class report and confusion matrix
src/ood.py             fits the "is this even a chest X-ray" check
src/gradcam_utils.py   heatmaps, and a CLI for one image
src/prepare_data.py    normalise a download into data/{train,val,test}/CLASS/
src/dataset_overlap.py whether two datasets share images, before trusting one
src/cross_dataset.py   score a second dataset with the shared images removed
src/mask_lungs.py      mirror a split with everything outside the lungs blacked out
src/synth_data.py      drawn stand-in images, for testing the pipeline
app/main.py            FastAPI service: /health, /predict, /explain
tests/                 pytest suite, no dataset and no network needed
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

Before scoring one against a model trained on the other, check that they are
actually different data — compiled Kaggle sets frequently re-package the same
source collections, and a resized or re-encoded copy has different bytes while
being the same radiograph:

```bash
~/venvs/smt/Scripts/python.exe -m src.dataset_overlap --left ~/Downloads/chest-xray-cp --right ~/Downloads/covid19-radiography
```

It compares contrast-normalised thumbnails, not a perceptual hash. A 64-bit
difference hash is the usual tool and it is useless here: every chest
radiograph shares a silhouette, so at 8x8 they collapse together, and 89% of
*independent* normal films landed within hamming distance 5 of some COVID film.
That fails in the direction that discards a valid experiment. See the module
docstring for the two controls the thresholds were read off.

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

Then fit the out-of-distribution check, which the API needs before it can tell
whether an upload is a chest X-ray at all — see
[Is it even a chest X-ray?](#is-it-even-a-chest-x-ray):

```bash
~/venvs/smt/Scripts/python.exe -m src.ood --checkpoint checkpoints/best.pt --out checkpoints/ood.pt
```

## Grad-CAM

```bash
~/venvs/smt/Scripts/python.exe -m src.gradcam_utils --checkpoint checkpoints/best.pt --image data/test/PNEUMONIA/00001_person1_virus_6.jpeg --out cam.png
```

`--class-name` explains a class other than the predicted one, which is how you
ask why the model did *not* say pneumonia.

Check a handful of correct predictions before trusting a score. If the heat sits
on a corner marker or outside the lungs, the model found a shortcut.

Be aware of what this check cannot do. Scanner, exposure and processing
signature is present *inside* the lung fields as well as around them, so a
model reading provenance rather than pathology still produces heatmaps that
look anatomically sensible. Heat on the lungs is necessary, not sufficient.

## Results

resnet18, COVID-19 Radiography Database, 15,153 images split 70/15/15 with the
two-stage recipe above. Test set, 2,273 images:

| | macro F1 | accuracy | COVID19 F1 | NORMAL F1 | PNEUMONIA F1 |
|---|---|---|---|---|---|
| as downloaded | 0.9816 | 0.9894 | 0.993 | 0.992 | 0.960 |
| lungs only    | 0.9615 | 0.9718 | 0.961 | 0.979 | 0.945 |

**Do not quote the first row on its own.** The classes in this dataset are
100% separable by provenance — see [What this is not](#what-this-is-not) — so a
high score is exactly what a model would produce by learning which repository
an image came from.

The second row is the same recipe trained on a mirror of the same split with
every non-lung pixel zeroed. Roughly 77% of each image is removed, including all
burned-in markers, collimation edges, soft tissue and background. The score fell
by two points rather than collapsing. To reproduce it:

```bash
~/venvs/smt/Scripts/python.exe -m src.mask_lungs --split-root data --source ~/Downloads/COVID-19_Radiography_Dataset --out data-masked
~/venvs/smt/Scripts/python.exe -m src.train --epochs 15 --freeze-backbone --data-dir data-masked --out checkpoints/masked_stage1.pt
~/venvs/smt/Scripts/python.exe -m src.train --epochs 25 --lr 1e-4 --data-dir data-masked --resume checkpoints/masked_stage1.pt --out checkpoints/masked_best.pt
~/venvs/smt/Scripts/python.exe -m src.evaluate --checkpoint checkpoints/masked_best.pt --split test --data-dir data-masked
```

It mirrors an existing split rather than re-splitting, so the two runs differ in
exactly one variable. Masks ship with the Radiography Database; most other
downloads have none.

What that establishes, and what it does not:

- It **rules out** the crude shortcut. The model is not reading annotations or
  background, because those are gone and it still scores 0.96.
- It **does not clear** the provenance confound. Acquisition signature survives
  inside lung pixels, and so does the lung silhouette — a child's lungs are
  shaped differently from an adult's, and Viral Pneumonia here is the paediatric
  Kermany collection while COVID-19 is adult European patients.

One detail argues that part of the original score *was* artifact: COVID-19 lost
roughly twice what the other classes did (−0.032 against −0.013 and −0.015),
and its recall fell 0.989 → 0.946. COVID-19 is the only class with unique
provenance, so it is the class with the most artifact to lose.

Nothing internal to this dataset can settle it, because the correlation is
total by construction. That needs a COVID-19 set from different hospitals,
where provenance no longer predicts the label.

### Scored against a second dataset

The obvious next move is to score against a different download. Done naively it
measures nothing. `prashant268/chest-xray-covid19-pneumonia` shares **23.9%** of
its 6,432 images with this model's training split — and **zero** of them match
by checksum, because every copy had been resized or re-encoded on the way in.
A hash comparison reports two completely independent datasets. See
[Real data](#real-data) for the check.

So the images are partitioned first, and three numbers come out:

```bash
~/venvs/smt/Scripts/python.exe -m src.cross_dataset --dataset ~/Downloads/chest-xray-cp/Data --exclude-against ~/cxr-data-real/train
```

| | images | macro F1 | accuracy |
|---|---|---|---|
| all, contaminated | 6,432 | 0.9473 | 0.9633 |
| overlapping only *(control)* | 1,537 | 0.9777 | 0.9850 |
| **clean** | **4,895** | **0.9288** | **0.9565** |

The middle row is the control and it is why the other two can be believed. Those
are training images: the model recalls **100%** of their pneumonia cases,
638 of 638. The clean set sits 4.9 points of macro F1 below that, so the filter
is separating the right images. Leaving them in was worth **+0.0185** macro F1.

Against 0.9816 on the held-out split of its own dataset, the clean number is
**0.9288**. Per class, F1 goes 0.993 → 0.922 for COVID19, 0.992 → 0.886 for
NORMAL, and 0.960 → 0.978 for PNEUMONIA. The last one rises partly because
pneumonia is 74% of this dataset and was 9% of the other, so the per-class
columns are not strictly like-for-like even though macro F1 absorbs most of it.

Where it fails is consistent with everything else here: NORMAL precision drops
to 0.866, with 93 pneumonia films and 35 COVID films called NORMAL. When this
model is wrong it is disproportionately wrong in the direction of "nothing
here".

**This is still not the different-hospitals experiment.** Removing shared
images removes image-level contamination and nothing else. Both datasets are
compiled from the same public archives, so much of the clean 4,895 plausibly
comes from the same collections as the training data — the same Kermany
pneumonia set, the same BIMCV series. What this establishes is that the model
does not collapse on unseen images from a differently-assembled download. It
does not establish that it reads pathology rather than provenance.

## Serve

```bash
~/venvs/smt/Scripts/python.exe -m uvicorn app.main:app --port 8100
```

- `GET /health` — whether weights actually loaded, and the val metrics they
  scored. Starts and reports `no_model` rather than crash-looping when there is
  no checkpoint. `ood_stats_loaded` says whether the out-of-distribution check
  is running at all.
- `POST /predict` — multipart `file`. Returns the ranked distribution, plus
  `out_of_distribution` with the `ood_score` and the `ood_threshold` actually
  applied. `low_confidence` is still there and is still only a weak hint: it
  catches the model being torn between its three classes, which is a much
  narrower failure than being handed something that is not a chest X-ray.

  `out_of_distribution` is `null` when no statistics are loaded — meaning the
  check did not run, **not** that the image passed. The API reads them from
  `CXR_OOD_STATS` (default `checkpoints/ood.pt`) and refuses any set fitted
  against a different checkpoint, since those measure distances from the wrong
  means and would otherwise look fine.
- `POST /explain` — multipart `file`, optional `class_name`. Returns the overlay
  PNG. `X-Prediction` is what the model called it; `X-Explained-Class` is what
  the heatmap answers for. They differ whenever `class_name` is passed.

Port 8100, so it does not collide with the triage backend on 8000.

`/explain` puts its labels in `X-Prediction` and `X-Explained-Class` so a browser
can point an `<img>` straight at the endpoint and still read what the picture
says. Those are named in the CORS `expose_headers`; without that a cross-origin
browser client sees only `content-type` and silently loses the label, which is
how it was found.

## Frontend

`frontend/` is a plain HTML/CSS/JS page with no build step, no CDN and no
framework, matching the triage app. Serve it alongside the API:

```bash
~/venvs/smt/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8100
~/venvs/smt/Scripts/python.exe -m http.server 5501 --directory frontend
```

Then open `http://localhost:5501`. It guesses the API address from its own
hostname, so serving both from the same machine needs no configuration; the
gear icon overrides it otherwise.

**From a phone on the same Wi-Fi**, open `http://<this-machine>:5501` and set
the API address to `http://<this-machine>:8100` — the `--host 0.0.0.0` above is
what makes the API reachable off localhost. Both phones can add it to the home
screen and it runs without browser chrome. Note that the service worker only
registers over HTTPS or on localhost, so over plain LAN HTTP the page works but
is not available offline.

Four deliberate choices in the UI:

- **The classes are not colour-coded.** COVID19 is not red and NORMAL is not
  green; every probability bar is the same colour and ranks by length alone. A
  green bar reading "NORMAL 94%" is an all-clear that this model is not
  entitled to give.
- **The disclaimer is not dismissible** and sits above the fold at every size.
- **Prediction and explanation are reported separately** whenever a class is
  requested, because they come apart: the map answers "why not pneumonia?"
  while the model's own call is still something else.
- **"Not a chest X-ray" and "low confidence" are separate notices, and the
  first outranks the second.** One says the scores below are unreliable, the
  other says there is nothing below worth reading. A third notice appears when
  no statistics are loaded, because an unchecked image must not look like one
  that passed.

## Is it even a chest X-ray?

`low_confidence` was documented here as the signal that an image was out of
distribution. It never did that job. Softmax over three classes normalises
whatever it is handed, so an image unlike anything in training does not come
back uncertain — it comes back wrong and certain. Measured against
`checkpoints/best.pt`:

| input | prediction | `low_confidence` |
|---|---|---|
| flat grey square | COVID19 100.00% | not flagged |
| flat black square | COVID19 100.00% | not flagged |
| flat white square | COVID19 99.67% | not flagged |
| uniform noise | COVID19 99.96% | not flagged |
| smooth colour photo | COVID19 100.00% | not flagged |
| a page of text | COVID19 99.86% | not flagged |

No cutoff on that column separates "a chest X-ray it is sure about" from "not a
chest X-ray at all", because there is no uncertainty in it to threshold.

`src/ood.py` measures a different thing. The logits are three numbers that have
already discarded everything except how much the image resembles each class; the
512-number feature vector feeding them still carries whether it resembled
anything. So: fit a Gaussian per class over the training features, take the
Mahalanobis distance to the nearest one, and refuse anything far enough out.
This is the standard construction, from Lee et al. 2018.

```bash
~/venvs/smt/Scripts/python.exe -m src.ood --checkpoint checkpoints/best.pt --out checkpoints/ood.pt
```

Fit on `train`, cutoffs calibrated on `val`, both with augmentation off. Every
one of the six inputs above is now refused, by a wide margin — they score 2,733
to 7,384 against cutoffs between 713 and 1,467.

**The cutoff is per class, and that is not a detail.** A single pooled cutoff at
the 95th percentile reported a reassuring 4.5% false-reject rate while actually
refusing 30.2% of real pneumonia films and 0.4% of normal ones — the pooled
number was set by NORMAL, which is two thirds of the split. It is the same trap
as reading accuracy instead of per-class recall, one level down. Calibrated per
class and applied by whichever class an image is nearest, on the held-out test
split:

| | cutoff | real X-rays refused |
|---|---|---|
| COVID19 | 1054.5 | 3.7% (20/542) |
| NORMAL | 712.8 | 3.9% (60/1529) |
| PNEUMONIA | 1467.3 | 3.5% (7/202) |

`--percentile` is the knob, and it has a cost on both sides: the default 95
spends about one real X-ray in twenty to catch the inputs above.

### What this does not do

It answers "unlike the training images". That is **not** the same as "not a
chest X-ray", and much further still from "the model cannot handle this".

The dataset makes the gap easy to measure. `Lung_Opacity` ships in the same
download and is excluded from training as a broader finding than pneumonia —
real chest X-rays, same repositories, showing something this model has no class
for. Of 600 of them, only 34.5% are refused. The rest are accepted and then
called **NORMAL 94.2% of the time, at a mean confidence of 0.972**.

That is the more dangerous failure, and this check does not catch it. A caller
gets "looks like a chest X-ray" and "NORMAL", about a film with a real opacity
on it. Nothing here can fix that, because a three-class head has no output for
it: NORMAL means "not the other two", never "clear". Catching it needs classes
for the findings you care about, or a model that can abstain by design.

There is also the confound running through the whole project. What the training
images have in common includes their provenance, so an ordinary chest X-ray from
a hospital outside these datasets is exactly the kind of thing that scores far
away and gets refused. The 3.5–3.9% above is measured against images from the
same four repositories, and is a floor rather than an estimate.

**How much of a floor is now measured.** On the clean images of a second
dataset — see [Scored against a second dataset](#scored-against-a-second-dataset)
— the refusal rate roughly doubles:

| | same dataset, held out | second dataset, clean |
|---|---|---|
| COVID19 | 3.7% | 10.3% |
| NORMAL | 3.9% | 1.8% |
| PNEUMONIA | 3.5% | 8.6% |
| pooled | 3.8% | 7.4% |

A cutoff calibrated at p95 on one dataset delivers roughly p92 on another, and
that dataset is not even from different hospitals. **The threshold does not
travel.** Recalibrate against images from wherever it will actually be used, or
the check quietly refuses one real film in ten while reporting one in twenty.

## Tests

```bash
~/venvs/smt/Scripts/python.exe -m pytest tests -q
```

No dataset, no network, no pretrained download — synthetic images and untrained
weights throughout.
