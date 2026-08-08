---
title: Chest X-ray Classifier
emoji: 🫁
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Chest X-ray classifier

A four-class CNN over chest radiographs — **NORMAL**, **PNEUMONIA**, **COVID19**,
**LUNG_OPACITY** — with Grad-CAM and a check for whether the image is a chest
X-ray at all.

## This is not a medical device

It is a research prototype trained on public datasets. It has not been validated
on any clinical population, and **a number it returns is not a finding**. Any
real concern about a chest X-ray belongs with a qualified radiologist.

Four specific reasons to distrust it:

- **The classes are separable by source, not just by disease.** Every COVID-19
  image in the training set came from BIMCV, Eurorad, SIRM or GitHub; every
  Normal and Viral Pneumonia image came from Kaggle. Zero overlap. A model can
  separate them by scanner or exposure and never look at a lung. Masking
  everything outside the lungs costs COVID-19 six times what it costs NORMAL,
  which says part of the score is exactly that.
- **The label is not the disease.** Labels came from whoever assembled the
  dataset, by varying and mostly undocumented criteria.
- **The class balance is not prevalence.** Nothing here estimates how likely a
  given patient is to have anything.
- **Four classes is not every finding.** Effusion, pneumothorax, nodules and
  everything else have no output and land on whichever class is nearest.
  NORMAL means "not the other three", never "clear".

## What it does that most demos do not

Softmax over a fixed class list normalises whatever it is handed, so an image
unlike anything in training does not come back uncertain — it comes back wrong
and certain. A flat grey square scores COVID19 at 99.4%, and a confidence
threshold catches none of it.

So there is a second check underneath: a Mahalanobis distance over the
penultimate features, with a per-class cutoff calibrated at the 95th percentile.
Upload something that is not a chest X-ray and it says so, rather than
confidently naming a disease.

That check costs about one real film in twenty, and it does **not** catch a real
chest X-ray showing a finding the model has no class for.

## Scores

| four-class, 3,175 held-out images | macro F1 | accuracy |
|---|---|---|
| as downloaded | 0.9587 | 0.9524 |
| lungs only, 77% of each image masked | 0.9341 | 0.9298 |

Do not quote either alone. Read the source repository for what they cost and
what they fail to establish.

**Source:** https://github.com/ardhendudebnath/chest-xray-classifier
