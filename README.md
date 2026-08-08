# Chest X-ray classifier

A four-class CNN over chest radiographs: **NORMAL**, **PNEUMONIA**, **COVID19**,
**LUNG_OPACITY**.

It lives beside the triage app but does not import from it and is not wired into
it. The triage backend reads symptom text; this reads an image. Keeping them
apart means the backend does not have to carry a multi-gigabyte torch install.

## What this is not

This is a research prototype trained on public datasets. It is not a medical
device, it has not been validated on any clinical population, and a number it
returns is not a finding. Four specific reasons to distrust it, beyond the
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
  to share 24.0% of its images with this one's training split, and the rest
  comes from the same public archives.
- **The label is not the disease.** These labels came from whoever assembled
  the dataset, by varying and mostly undocumented criteria — some RT-PCR
  confirmed, some radiologist-read, some neither.
- **The class balance is not the real prevalence.** Nothing here estimates how
  likely a given patient is to have anything.
- **Four classes is not every finding, and NORMAL does not mean clear.**
  LUNG_OPACITY is a class here only because leaving it out was measurably
  worse: the three-class model met those films anyway and called them NORMAL
  94.2% of the time at a mean confidence of 0.972. Adding a class fixes that
  one case and not the general one. Effusion, pneumothorax, nodules and
  everything else still have no output and still land on whichever class is
  nearest. **NORMAL means "not the other three", never "clear".**

Grad-CAM is included for exactly this reason. It is not decoration.

## At a glance

resnet18, two-stage fine-tuning, 21,165 images from the COVID-19 Radiography
Database split 70/15/15 by patient group. Roughly half this repository is the
model; the other half is the machinery for deciding whether its numbers mean
anything.

### How a number gets made

```mermaid
flowchart TD
    DL["COVID-19 Radiography Database<br/>21,165 images, 4 classes"]
    DL --> PD["src.prepare_data<br/>alias class names, keep a patient in one split"]
    PD --> SPLIT["train 14,814 · val 3,176 · test 3,175"]
    SPLIT --> T1["src.train --freeze-backbone<br/>stage 1, head only"]
    T1 --> T2["src.train --resume --lr 1e-4<br/>stage 2, whole network"]
    T2 --> CKPT["checkpoints/best.pt<br/>selected on val macro F1"]
    SPLIT --> OOD["src.ood<br/>fit one Gaussian per class, cutoffs at p95"]
    CKPT --> OOD
    OOD --> STATS["checkpoints/ood.pt"]
    CKPT --> EV["src.evaluate<br/>per-class F1 and confusion matrix"]
    CKPT --> API["app.main<br/>/predict · /explain · /health"]
    STATS --> API
    API --> FE["frontend<br/>PWA, class list read from /health"]
```

### Why softmax cannot say "none of the above"

One forward pass, two answers taken from different depths. The top path always
returns a class, whatever it is handed. Only the bottom path can decline.

```mermaid
flowchart LR
    IMG["any image at all"] --> TF["greyscale, resize to 224"]
    TF --> BB["resnet18 backbone"]
    BB --> FEAT["512 features<br/>what does this look like at all"]
    FEAT --> HEAD["4-class head"]
    HEAD --> LOGITS["4 logits<br/>how much like each class"]
    LOGITS --> SM["softmax"]
    SM --> PRED["prediction + confidence<br/>always answers"]
    FEAT --> MAH["Mahalanobis distance<br/>to the nearest class mean"]
    MAH --> CUT["past that class's cutoff?<br/>can refuse"]
```

A flat grey square comes back COVID19 at 99.4% with `low_confidence` unflagged.
The distance check refuses it, and the other five probes, at 1,594 to 21,485
against a cutoff of 939.8.

### Results

| four-class, 3,175 test images | macro F1 | accuracy |
|---|---|---|
| as downloaded | 0.9587 | 0.9524 |
| lungs only, 77% of each image masked | 0.9341 | 0.9298 |
| second dataset, clean, restricted | 0.9452 | 0.9675 |

**Do not quote any of these alone.** The classes are 100% separable by
provenance, so a high score is what a model would produce by learning which
repository an image came from. The rest of this file is largely about that.

### What the machinery establishes

| question | answered by | finding |
|---|---|---|
| Does it read lungs or artifacts? | `src.mask_lungs` | Masking costs COVID-19 −0.056 F1 against NORMAL's −0.009. It is the only class with unique provenance, so it has the most artifact to lose. |
| Is the second dataset different data? | `src.dataset_overlap` | 24.0% of it is already in the training split, and **zero** of those match by checksum — every copy was re-encoded on the way in. |
| Does it hold up off-dataset? | `src.cross_dataset` | 0.9452 macro F1 on the clean remainder, after the shared images are removed and reported separately. |
| Does the refusal check still work? | `src.probe_ood` | All six non-radiographs refused; all six still called COVID19 above 98.7% by the softmax. |
| What did NORMAL used to hide? | the class list itself | The three-class model called lung opacity films NORMAL 94.2% of the time at 0.972 confidence. Adding the class moved that error onto the confusion matrix. |

### Where to look

- Numbers and what they cost: [Results](#results)
- The provenance problem, measured: [The lung-masking check](#the-lung-masking-check)
- Off-dataset scoring: [Scored against a second dataset](#scored-against-a-second-dataset)
- Refusing non-radiographs: [Is it even a chest X-ray?](#is-it-even-a-chest-x-ray)

## Layout

```
src/dataset.py         loaders, transforms, the two imbalance corrections
src/model.py           backbone + 4-class head, checkpoint save/load
src/train.py           fine-tuning loop, selects on macro F1
src/evaluate.py        per-class report and confusion matrix
src/ood.py             fits the "is this even a chest X-ray" check
src/probe_ood.py       scores six non-radiographs, to check that it still works
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
"COVID19", "Viral Pneumonia" vs "PNEUMONIA", "Lung_Opacity" vs "LUNG_OPACITY").
Lung opacity is kept as its own class and never folded into pneumonia — it is a
broader finding, and merging them would change what the model claims to detect.

**Only the Radiography Database has all four.** The smaller Kermany-derived sets
carry no lung opacity directory, so they cannot train this model; `prepare_data`
stops with a message rather than writing a split with a class missing, which
ImageFolder would silently renumber the rest around.

It also **keeps one patient's images in a single split**. The Kermany pneumonia
images are named `person1_virus_6`, `person1_bacteria_1` and so on — several
films of the same chest. Split those at random and the same patient appears in
train and test, which inflates the test score by several points. This is why
the published train/test split is pooled and redone rather than used as-is.

## Train

```bash
~/venvs/smt/Scripts/python.exe -m src.train --epochs 15 --freeze-backbone --num-workers 4 --out checkpoints/stage1.pt
~/venvs/smt/Scripts/python.exe -m src.train --epochs 25 --lr 1e-4 --num-workers 4 --resume checkpoints/stage1.pt --out checkpoints/best.pt
```

**Pass `--num-workers`.** It defaults to 0 because a worker on Windows
re-imports the calling module and deadlocks unless the entry point is guarded,
and `build_loader` is called from places that are not. `python -m src.train` is
guarded, so workers are safe there — and they matter: decoding 21k JPEGs on one
thread left this GPU 12% busy at roughly 3 minutes an epoch. With four workers
it sat near 50% and under a minute, which is the difference between a coffee
and an afternoon.

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

resnet18, COVID-19 Radiography Database, 21,165 images split 70/15/15 with the
two-stage recipe above. Test set, 3,175 images:

| four-class | macro F1 | accuracy | COVID19 F1 | LUNG_OPACITY F1 | NORMAL F1 | PNEUMONIA F1 |
|---|---|---|---|---|---|---|
| as downloaded | 0.9587 | 0.9524 | 0.982 | 0.929 | 0.953 | 0.970 |

**Do not quote these on their own.** The classes in this dataset are 100%
separable by provenance — see [What this is not](#what-this-is-not) — so a high
score is exactly what a model would produce by learning which repository an
image came from.

The headline number went down when the class was added, 0.9816 → 0.9587. Read
those two side by side with care. They are macro F1 over different class sets on
different test splits, 2,273 images then and 3,175 now, so the difference is not
a like-for-like regression on a fixed task.

The two splits are not nested either, and the reason is worth knowing before you
compare any two runs here. `prepare_data` draws from one `random.Random(seed)`
inside its loop over `CLASSES`, so inserting a class shifts the stream for every
class sorting after it — the same seed puts different NORMAL and PNEUMONIA
images in test than it did with three classes. The per-class counts are
identical (542 / 1529 / 202) only because each class lands on its 15% by
construction, which makes them no evidence at all that the images are the same.
COVID19 sorts first and is unaffected.

With that said, the per-class columns are the closest thing to a like-for-like
view, and they say where the loss went:

| F1 | COVID19 | NORMAL | PNEUMONIA |
|---|---|---|---|
| three-class | 0.993 | 0.992 | 0.960 |
| four-class | 0.982 | 0.953 | 0.970 |

**NORMAL took almost all of the loss**, 0.992 → 0.953, and pneumonia went up.
That is the shape you would predict if the three-class NORMAL had been quietly
absorbing lung opacity films, which is exactly what it was doing: shown 600 of
them it called 94.2% NORMAL at a mean confidence of 0.972, and reported nothing
unusual about any of it. The four-class model gets them wrong differently — 72
of 902 still land on NORMAL, and 48 normal films now come back LUNG_OPACITY,
which is why LUNG_OPACITY is the weakest class at 0.929.

The confusion is real either way. The difference is that it is now on the
confusion matrix instead of inside the NORMAL column, so the drop is a failure
becoming visible rather than one being introduced.

### The lung-masking check

| four-class | macro F1 | accuracy | COVID19 F1 | LUNG_OPACITY F1 | NORMAL F1 | PNEUMONIA F1 |
|---|---|---|---|---|---|---|
| as downloaded | 0.9587 | 0.9524 | 0.982 | 0.929 | 0.953 | 0.970 |
| lungs only    | 0.9341 | 0.9298 | 0.926 | 0.899 | 0.944 | 0.967 |

The second row is the same recipe trained on a mirror of the same split with
every non-lung pixel zeroed. Roughly 77% of each image is removed, including all
burned-in markers, collimation edges, soft tissue and background. The score fell
by two and a half points rather than collapsing. To reproduce it:

```bash
~/venvs/smt/Scripts/python.exe -m src.mask_lungs --split-root ~/cxr-data-4class --source ~/Downloads/covid19-radiography/COVID-19_Radiography_Dataset --out ~/cxr-data-4class-masked
~/venvs/smt/Scripts/python.exe -m src.train --epochs 15 --freeze-backbone --num-workers 4 --data-dir ~/cxr-data-4class-masked --out checkpoints/masked_stage1.pt
~/venvs/smt/Scripts/python.exe -m src.train --epochs 25 --lr 1e-4 --num-workers 4 --data-dir ~/cxr-data-4class-masked --resume checkpoints/masked_stage1.pt --out checkpoints/masked_best.pt
~/venvs/smt/Scripts/python.exe -m src.evaluate --checkpoint checkpoints/masked_best.pt --split test --data-dir ~/cxr-data-4class-masked --report-dir reports/masked_4class
```

It mirrors an existing split rather than re-splitting, so the two runs differ in
exactly one variable. Masks ship with the Radiography Database, for all four
classes; most other downloads have none. Pass `--report-dir`, or the evaluation
overwrites the unmasked run's saved metrics.

What that establishes, and what it does not:

- It **rules out** the crude shortcut. The model is not reading annotations or
  background, because those are gone and it still scores 0.93.
- It **does not clear** the provenance confound. Acquisition signature survives
  inside lung pixels, and so does the lung silhouette — a child's lungs are
  shaped differently from an adult's, and Viral Pneumonia here is the paediatric
  Kermany collection while COVID-19 is adult European patients.

One detail argues that part of the original score *was* artifact, and it is the
clearest signal in this table. Masking costs the four classes wildly different
amounts:

| | COVID19 | LUNG_OPACITY | NORMAL | PNEUMONIA |
|---|---|---|---|---|
| F1 lost to masking | −0.056 | −0.030 | −0.009 | −0.003 |

COVID-19 loses six times what NORMAL does and nearly twenty times what pneumonia
does, and its recall falls 0.976 → 0.911. It is the only class with unique
provenance, so it is the class with the most artifact to lose. The three-class
version of this experiment found the same thing at half the magnitude (−0.032
against −0.013 and −0.015), so the effect has now reproduced across two
different class lists.

LUNG_OPACITY, measured this way for the first time, sits mid-table. It loses
more than NORMAL or PNEUMONIA and less than half what COVID-19 does, which is
what you would expect from a class drawn from the same repositories as NORMAL
rather than carrying a provenance signature of its own.

Nothing internal to this dataset can settle it, because the correlation is
total by construction. That needs a COVID-19 set from different hospitals,
where provenance no longer predicts the label.

### Scored against a second dataset

The obvious next move is to score against a different download. Done naively it
measures nothing. `prashant268/chest-xray-covid19-pneumonia` shares **24.0%** of
its 6,432 images with this model's training split — and **zero** of them match
by checksum, because every copy had been resized or re-encoded on the way in.
A hash comparison reports two completely independent datasets. See
[Real data](#real-data) for the check.

So the images are partitioned first, and three numbers come out:

```bash
~/venvs/smt/Scripts/python.exe -m src.cross_dataset --dataset ~/Downloads/chest-xray-cp/Data --exclude-against ~/cxr-data-4class/train
```

| | images | macro F1 | accuracy |
|---|---|---|---|
| all, contaminated | 6,432 | 0.9492 | 0.9667 |
| overlapping only *(control)* | 1,545 | 0.9864 | 0.9903 |
| **clean** | **4,887** | **0.9252** | **0.9593** |

The middle row is the control and it is why the other two can be believed. Those
are training images, and the model gets 671 of 671 of their normal films and 647
of 647 of their pneumonia right. The clean set sits 6.1 points of macro F1 below
that, so the filter is separating the right images. Leaving them in was worth
**+0.0240** macro F1.

#### The model has a class this dataset does not

`prashant268` has NORMAL, PNEUMONIA and COVID19. It has no lung opacity
directory at all, while the model has four outputs — so what is a LUNG_OPACITY
prediction here? There is no answer that is simply correct, and the two
defensible ones measure different things, so the script reports both:

| clean, 4,887 images | macro F1 | accuracy | COVID19 F1 | NORMAL F1 | PNEUMONIA F1 |
|---|---|---|---|---|---|
| open — all four outputs live | 0.9252 | 0.9593 | 0.868 | 0.923 | 0.984 |
| restricted — absent class masked | 0.9452 | 0.9675 | 0.936 | 0.916 | 0.984 |

**Open** counts a LUNG_OPACITY prediction as wrong, because these labels cannot
confirm it. That is a lower bound: some of those films may really show an
opacity this dataset had no category for and filed under something else.
**Restricted** masks the absent class out of the logits before argmax, so the
model must choose among the classes the dataset knows — the like-for-like
against a three-class score. Both average over the same three labelled classes,
and the JSON records which under `macro_f1_over`.

The gap is **+0.0200**, and it comes from **55 of 4,887 images (1.1%)** where the
model reached for a class these labels cannot express. Nearly all of them were
COVID-19 films: masking the class recovers COVID19 F1 from 0.868 to 0.936, and
its recall from 0.785 to 0.900. Quote both numbers or neither.

Where it fails is consistent with everything else here. Under the restricted
pass, NORMAL precision drops to 0.889 — 72 pneumonia films and 35 COVID films
called NORMAL. When this model is wrong it is disproportionately wrong in the
direction of "nothing here".

Against 0.9587 on the held-out split of its own dataset, the clean number is
0.9252 open and 0.9452 restricted. Per class it is not strictly like-for-like:
pneumonia is 74% of this dataset and 6% of the other, so PNEUMONIA F1 rising to
0.984 says as much about the class balance as about the model.

**This is still not the different-hospitals experiment.** Removing shared
images removes image-level contamination and nothing else. Both datasets are
compiled from the same public archives, so much of the clean 4,887 plausibly
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
  catches the model being torn between its four classes, which is a much
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
distribution. It never did that job. Softmax over a fixed set of classes
normalises whatever it is handed, so an image unlike anything in training does
not come back uncertain — it comes back wrong and certain. Measured against
`checkpoints/best.pt`:

| input | prediction | `low_confidence` |
|---|---|---|
| flat grey square | COVID19 99.42% | not flagged |
| flat black square | COVID19 99.36% | not flagged |
| flat white square | COVID19 98.75% | not flagged |
| uniform noise | COVID19 100.00% | not flagged |
| smooth colour image \* | COVID19 99.86% | not flagged |
| a page of text \* | COVID19 99.98% | not flagged |

No cutoff on that column separates "a chest X-ray it is sure about" from "not a
chest X-ray at all", because there is no uncertainty in it to threshold. Adding
a fourth class changed nothing about this: the four-class model answers COVID19
above 98.7% for every one of them, exactly as the three-class one did.

Both columns of this table, and the scores below, come from:

```bash
~/venvs/smt/Scripts/python.exe -m src.probe_ood
```

It generates the six inputs rather than storing them, so the table can be
regenerated against any checkpoint. `--save-dir` writes them out to look at.

\* The first four inputs are fully determined by their names and match what was
measured before. The colour image and the text page are stand-ins — the original
ad-hoc files predate the script and were not kept — so those two rows are new
measurements rather than a re-run of the earlier ones.

`src/ood.py` measures a different thing. The logits are four numbers that have
already discarded everything except how much the image resembles each class; the
512-number feature vector feeding them still carries whether it resembled
anything. So: fit a Gaussian per class over the training features, take the
Mahalanobis distance to the nearest one, and refuse anything far enough out.
This is the standard construction, from Lee et al. 2018.

```bash
~/venvs/smt/Scripts/python.exe -m src.ood --checkpoint checkpoints/best.pt --out checkpoints/ood.pt
```

Fit on `train`, cutoffs calibrated on `val`, both with augmentation off. Every
one of the six inputs above is refused, by a wide margin — they score 1,594 to
21,485, all nearest to COVID19, against that class's cutoff of 939.8.

**The cutoff is per class, and that is not a detail.** Measured on the
three-class model, a single pooled cutoff at the 95th percentile reported a
reassuring 4.5% false-reject rate while actually refusing 30.2% of real
pneumonia films and 0.4% of normal ones — the pooled number was set by NORMAL,
which was two thirds of that split. It is the same trap as reading accuracy
instead of per-class recall, one level down, and it is a property of the class
imbalance rather than of the class list.

Calibrated per class and applied by whichever class an image is nearest, fitted
on 14,814 training images and calibrated on 3,176 validation ones at p95. On the
held-out test split:

| | cutoff | real X-rays refused |
|---|---|---|
| COVID19 | 939.8 | 6.6% (36/542) |
| LUNG_OPACITY | 822.5 | 5.1% (46/902) |
| NORMAL | 653.4 | 4.1% (63/1529) |
| PNEUMONIA | 1310.4 | 6.4% (13/202) |
| pooled | — | 5.0% (158/3175) |

**The cost went up with the fourth class.** The three-class model refused
3.5–3.9% per class; this one refuses 4.1–6.6%. A p95 calibration implies about
5%, so it is the three-class figure that was the outlier — the four-class rates
are what this knob has been promising all along. If you were reading 3.8% as the
price of the check, the price is 5%.

`--percentile` is the knob, and it has a cost on both sides: the default 95
spends about one real X-ray in twenty to catch inputs like the six above.

### What this does not do

It answers "unlike the training images". That is **not** the same as "not a
chest X-ray", and much further still from "the model cannot handle this".

Lung opacity is how that gap got measured here. `Lung_Opacity` ships in the same
download and was excluded from training as a broader finding than pneumonia —
real chest X-rays, same repositories, showing something the model had no class
for. Of 600 of them, only 34.5% were refused. The rest were accepted and then
called **NORMAL 94.2% of the time, at a mean confidence of 0.972**.

That measurement is why LUNG_OPACITY is a class now. It does not follow that the
gap is closed. What those films demonstrated was never specific to lung opacity;
it was that the class list is shorter than the chest, and it still is. Effusion,
pneumothorax, nodules, masses, fibrosis — each is a real chest X-ray that looks
like the training data to this check, so it is accepted, and then labelled with
whichever of the four classes is nearest. The old evidence says that will
disproportionately be NORMAL, which is the direction that costs the most.

So a caller still gets "looks like a chest X-ray" and "NORMAL" about a film with
a real finding on it. Adding a class moved that boundary out by one finding; it
did not remove it, and it cannot. **NORMAL means "not the other three", never
"clear".** Catching the general case needs classes for every finding you care
about, or a model that can abstain by design.

There is also the confound running through the whole project. What the training
images have in common includes their provenance, so an ordinary chest X-ray from
a hospital outside these datasets is exactly the kind of thing that scores far
away and gets refused. A false-reject rate measured against images from the same
four repositories is a floor rather than an estimate.

**How much of a floor is measured.** On the clean images of a second dataset —
see [Scored against a second dataset](#scored-against-a-second-dataset) — the
refusal rate rises across every class:

| | same dataset, held out | second dataset, clean |
|---|---|---|
| COVID19 | 6.6% | 9.2% |
| LUNG_OPACITY | 5.1% | — *(not labelled there)* |
| NORMAL | 4.1% | 7.3% |
| PNEUMONIA | 6.4% | 6.3% |
| pooled | 5.0% | 6.7% |

A cutoff calibrated at p95 on one dataset delivers roughly p93 on another, and
that dataset is not even from different hospitals. **The threshold does not
travel.** Recalibrate against images from wherever it will actually be used, or
the check refuses more real films than the percentile you set implies.

The effect is milder than it was under the three-class model, where the same
comparison ran 3.8% to 7.4% pooled. That is because the same-dataset floor rose
rather than the second-dataset figure falling — see the cutoff table above.

## Tests

```bash
~/venvs/smt/Scripts/python.exe -m pytest tests -q
```

No dataset, no network, no pretrained download — synthetic images and untrained
weights throughout.

## Licence

MIT — see [LICENSE](LICENSE). Use it for anything, keep the copyright notice.

The licence covers the code and nothing else. It says nothing about the
datasets, which carry their own terms from their respective Kaggle
distributions, and it grants no rights to `checkpoints/best.pt`, which is not
in this repository. Read [What this is not](#what-this-is-not) before doing
anything with either.

The warranty disclaimer in that file is not boilerplate here. This is a
research prototype trained on data whose classes are separable by provenance,
and it is offered with no promise that any number it produces means what it
appears to mean.
