# Corrected paper sections

Drafted 2026-08-10 against `Explainable_ChestXray_Research_Paper.pdf`. Each
section below states what was wrong and gives replacement prose. Every number
is traceable to a file under `reports/` — the provenance is given beside each
table so nothing here has to be taken on trust.

**Reproduce every number in this document:**

```bash
~/venvs/smt/Scripts/python.exe -m src.evaluate --checkpoint checkpoints/best.pt --data-dir ~/cxr-data-4class --split test
~/venvs/smt/Scripts/python.exe -m src.evaluate --checkpoint checkpoints/r50_best.pt --data-dir ~/cxr-data-4class --split test --report-dir reports/resnet50
~/venvs/smt/Scripts/python.exe -m src.evaluate --checkpoint checkpoints/masked_best.pt --data-dir ~/cxr-data-4class-masked --split test --report-dir reports/masked_4class
```

The seed study of §4.2. `--seed` goes to **both** stages; every other argument
is identical to the runs above. Seed 42 is the default and is the checkpoint
already trained:

```bash
for S in 1 2; do
  ~/venvs/smt/Scripts/python.exe -m src.train --epochs 15 --freeze-backbone --num-workers 4 --data-dir ~/cxr-data-4class --seed $S --out checkpoints/seed${S}_stage1.pt
  ~/venvs/smt/Scripts/python.exe -m src.train --epochs 25 --lr 1e-4 --num-workers 4 --data-dir ~/cxr-data-4class --seed $S --resume checkpoints/seed${S}_stage1.pt --out checkpoints/seed${S}_best.pt
  ~/venvs/smt/Scripts/python.exe -m src.evaluate --checkpoint checkpoints/seed${S}_best.pt --data-dir ~/cxr-data-4class --split test --report-dir reports/seed$S
done
```

Add `--backbone resnet50` to the first line and rename the artefacts for the
ResNet50 arm.

Explanation fidelity, per backbone. Both use the default `--seed 0`, which is
what makes them score the identical 200 images under identical controls:

```bash
~/venvs/smt/Scripts/python.exe -m src.xai_eval --checkpoint checkpoints/best.pt --data-dir ~/cxr-data-4class --limit 200 --masks ~/Downloads/covid19-radiography/COVID-19_Radiography_Dataset --ood-stats checkpoints/ood.pt
~/venvs/smt/Scripts/python.exe -m src.xai_eval --checkpoint checkpoints/r50_best.pt --data-dir ~/cxr-data-4class --limit 200 --masks ~/Downloads/covid19-radiography/COVID-19_Radiography_Dataset --ood-stats checkpoints/r50_ood.pt --background checkpoints/r50_shap_background.pt --report-dir reports/resnet50
```

---

## Title and framing

**What changed.** The submitted title promises "Trustworthy Clinical Decision
Support". The work as built does not support that claim and the strongest
result in it is a negative one, so the title should say what was actually
established.

> **Explainable Deep Learning for Multi-Class Chest X-ray Classification:
> Measuring What Grad-CAM and SHAP Actually Establish**
>
> *An audit of a dual-explainability pipeline, and of the dataset confound
> neither method detects*

---

## Abstract (replaces the current abstract)

> Convolutional neural networks achieve high reported accuracy on chest
> radiograph classification, and explainable AI (XAI) methods are routinely
> added to such systems on the argument that a visible explanation lets a
> clinician verify the model is using genuine pathology rather than a spurious
> correlate. This paper builds that argument's standard implementation — a
> transfer-learned CNN with Grad-CAM for spatial explanation and SHAP for
> feature-level attribution, served through a FastAPI endpoint — and then
> measures whether it does what it is claimed to do.
>
> We train ResNet18 and ResNet50 classifiers on the COVID-19 Radiography
> Database (21,165 images; four classes: normal, pneumonia, COVID-19, lung
> opacity) using a two-stage fine-tuning schedule and a patient-grouped
> 70/15/15 split, three seeds each. Macro F1 is 0.9571 +/- 0.0016 for ResNet18
> and 0.9548 +/- 0.0034 for ResNet50 — a 2.1x increase in parameters that is
> indistinguishable from seed noise and doubles the run-to-run variance. SHAP
> values are computed in closed form over the network's penultimate features,
> which is exact for a linear classification head and is verified against the
> reference implementation.
>
> Three measurements qualify these results. First, the dataset's classes are
> completely separable by source archive before any lung is examined, and
> retraining on lung-masked images — 77% of each image removed — costs COVID-19
> six times the F1 it costs the normal class, identifying acquisition signature
> rather than pathology as a substantial part of what was learned. Second,
> increasing backbone capacity by 2.1x produces no change distinguishable from
> seed noise across three runs per backbone, and doubles the run-to-run
> variance, which is what one expects when a shortcut has already been fully
> exploited. Third, and
> most importantly for the XAI claim, neither explanation detects any of this:
> Grad-CAM produces anatomically plausible maps regardless, and SHAP measures
> deviation from a training-set mean that carries the same confound.
>
> We further show that the standard deletion metric for explanation fidelity is
> unreliable on this data, and quantify why: 59.7% of its perturbation steps for
> ResNet18 and 72.6% for ResNet50 are rejected as out-of-distribution by the
> respective model's own Mahalanobis detector, so the metric partly measures
> distribution shift rather than explanation quality. The two backbones test
> this directly — the model with the higher rejection rate is also the one whose
> deletion result is further in the wrong direction, while being the better
> localised of the two on insertion, so the two metrics rank the networks in
> opposite orders and the rejection rate says which ranking to believe.
> Insertion AUC, which lacks this defect, places Grad-CAM well above a random
> ordering on both networks (+0.34 and +0.60 over their respective controls). We
> conclude that visual and feature-level explanations are necessary but
> demonstrably insufficient for the verification role assigned to them; that
> explanation metrics must be reported against per-model controls, since neither
> raw AUCs nor raw attribution similarities are comparable across
> architectures; and that dataset provenance auditing must accompany rather than
> follow explainability work.
>
> **Keywords:** Explainable AI, Chest X-ray Classification, Grad-CAM, SHAP,
> Shortcut Learning, Dataset Bias, Out-of-Distribution Detection, Explanation
> Fidelity

---

## §1 Introduction — corrected contribution list

**What changed.** The submitted list claims multi-label classification and
promises an evaluation protocol rather than results. Both are now wrong: the
model is single-label four-class, and the experiments have been run.

> The specific contributions of this work are as follows.
>
> 1. A four-class chest radiograph classifier (normal, pneumonia, COVID-19,
>    lung opacity) trained by two-stage transfer learning, reported at two
>    backbone capacities so that the effect of model size can be separated from
>    the effect of the data.
> 2. A dual explainability pipeline combining Grad-CAM for spatial explanation
>    with SHAP for feature-level attribution, in which the SHAP values are exact
>    rather than approximated: because the classification head is a single
>    linear layer over the penultimate representation, the Shapley values admit
>    a closed form that we verify against the reference implementation.
> 3. **A quantitative audit of both explanations** using deletion and insertion
>    curves, lung-field localisation, and attribution stability, each reported
>    against an explicit null control rather than in isolation.
> 4. **The finding that the deletion metric is confounded by distribution shift
>    on this data**, together with a method for detecting that condition using
>    the classifier's own out-of-distribution detector.
> 5. **A dataset provenance audit** showing that the classes are separable by
>    source archive, quantified by a lung-masking ablation and a capacity
>    ablation, and the demonstration that neither explanation method surfaces
>    this.
> 6. An abstention mechanism based on Mahalanobis distance in feature space,
>    with per-class calibration, and the measurement that its threshold does not
>    transfer between datasets.

---

## §3.2 Dataset and Preprocessing (full replacement)

**What changed.** The submitted text names NIH ChestX-ray14, CheXpert and a
tuberculosis class. None of those were used. The paragraph below describes what
was actually trained on.

> **Dataset.** All experiments use the COVID-19 Radiography Database, a
> publicly available compilation of 21,165 de-identified chest radiographs
> distributed through Kaggle. We use four classes: NORMAL (10,192), LUNG_OPACITY
> (6,012), COVID19 (3,616) and PNEUMONIA (1,345). Lung opacity is retained as a
> class in its own right and is never merged into pneumonia, since it denotes a
> broader radiographic finding; §5 reports what happened when it was omitted.
> The database also ships lung segmentation masks for all four classes, which
> §4.3 uses for the masking ablation and for localisation scoring.
>
> **Splitting.** Images are partitioned 70/15/15 into training (14,814),
> validation (3,176) and test (3,175) sets. The split is **grouped by patient**:
> several of the constituent collections contain multiple films of the same
> chest under systematic filenames, and splitting those at random places the
> same patient in both training and test, which inflates the reported test score.
> For this reason the publishers' own train/test division is pooled and redone
> rather than used as distributed.
>
> **Preprocessing.** Images are converted to greyscale and replicated to three
> channels — chest radiographs carry no colour, but ImageNet-pretrained
> backbones have a three-channel input convolution whose weights are discarded
> if it is replaced — resized to 224x224, and normalised with ImageNet channel
> statistics.
>
> **Augmentation** (training split only) comprises random resized cropping
> (scale 0.85–1.0), rotation up to 10 degrees, and brightness and contrast
> jitter of 0.15, standing in respectively for framing, patient positioning and
> exposure differences between machines and operators. **Horizontal flipping is
> deliberately excluded.** It is the default augmentation for natural images and
> appears in most published chest X-ray pipelines, but a mirrored chest places
> the heart on the right, an anatomy occurring in roughly 1 in 10,000 people.
> Training a model to treat it as unremarkable is not a defensible trade for a
> marginal increase in effective dataset size.
>
> **Class imbalance** is addressed by a weighted random sampler that draws each
> class approximately equally often. Inverse-frequency loss weighting is
> implemented as an alternative; the two are not combined, since applying both
> over-corrects toward the rare classes.

---

## §3.3 Model Architecture (full replacement)

**What changed.** The submitted text claims ResNet50. Both backbones were
trained; reporting both is more informative than either alone.

> **Backbones.** We report two ImageNet-pretrained backbones, ResNet18 and
> ResNet50, each with its 1000-way ImageNet head replaced by a fresh four-way
> linear layer, giving 11.18M and 23.52M parameters respectively — a factor of
> 2.1. Reporting both is not redundancy: §4.2 uses the comparison as a capacity
> ablation, and the near-identical result is itself evidence about the dataset.
>
> **Two-stage fine-tuning.** The backbone is first frozen and only the new head
> is trained (learning rate 3x10^-4), then the whole network is unfrozen and
> training continues from those weights at a reduced rate (1x10^-4).
> Fine-tuning an entire residual network against a few thousand images per class
> from the outset largely memorises them. The second stage restores weights
> only: optimizer state is not carried across, since AdamW moments accumulated
> while the backbone was frozen do not describe the parameters the second stage
> unfreezes, and the cosine schedule belongs to the epoch budget of the run that
> declared it.
>
> **Optimisation.** AdamW, weight decay 10^-4, cosine-annealed learning rate,
> batch size 32, early stopping with patience 5.
>
> **Model selection is on validation macro F1, not accuracy.** With COVID-19 at
> roughly a tenth of the normal-class count, a model that never predicts the
> rarest class can still post a high accuracy, and selection on accuracy would
> faithfully preserve that model. Macro F1 averages the per-class scores
> equally, making an ignored class expensive.

---

## §3.4 Explainability Pipeline (full replacement)

**What changed.** The submitted text describes SHAP as applied to "learned
feature activations" — which is right, and worth making precise, because it is
what makes the two methods complementary instead of redundant. It should also
state that the values are exact.

> **Grad-CAM.** Gradient-weighted Class Activation Mapping is applied at the
> final convolutional block (7x7 for a 224-pixel input), weighting each feature
> map by the mean gradient of the target class score with respect to it,
> rectifying, and bilinearly upsampling to the input resolution. Explanations
> can be requested for a class other than the predicted one, which answers "why
> did the model *not* say pneumonia" rather than only "why did it say COVID-19".
>
> **SHAP over learned features.** SHAP is applied to the penultimate
> representation — 512 dimensions for ResNet18, 2048 for ResNet50 — rather than
> to pixels. This is a deliberate choice. Pixel-level SHAP produces another
> spatial map, which would give the system two answers to *where* and none to
> *how much*; attributing over the features the classifier actually reads gives
> a genuinely different view.
>
> This choice also makes the attribution exact. The head is a single linear
> layer, so the class score is w·x + b, and for a linear model the Shapley
> values have a closed form:
>
>     phi_i = w_i (x_i - E[x_i]),    base = w . E[x] + b
>
> with E[x] the mean feature vector over the training split, computed once per
> checkpoint. No sampling and no convergence criterion are involved. The
> decomposition satisfies the Shapley efficiency axiom exactly — the
> contributions plus the base value reconstruct the logit — and our
> implementation is verified to agree with `shap.LinearExplainer` (Lundberg and
> Lee, 2017) to floating-point precision. Attributions may also be computed for
> the *margin* between two classes, which is the quantity an argmax decision
> actually turns on.
>
> **Four limitations of the SHAP component**, stated here because they bear
> directly on how §4.4 should be read. (i) A feature index is not a clinical
> concept: "feature 453 contributed +0.44 logits" is a true statement about the
> network and conveys nothing to a radiologist, and none of the dimensions
> corresponds to a named finding. (ii) The interventional formulation treats
> features as independent, which convolutional channels are not; this is the
> standard assumption and remains an assumption. (iii) Logits are explained, not
> probabilities, since softmax is not additive. (iv) **The baseline is the mean
> training image, so the attribution inherits whatever the training set has in
> common — including its provenance.** §4.5 shows why this matters.

---

## §3.5 System Integration (full replacement)

**What changed.** The submitted text places a new endpoint on the Smart
Healthcare Triage System's backend. That integration was not built, and
deliberately so; describing it as done would misreport the architecture.

> Classification and both explanations are served from a FastAPI application
> with four endpoints: `/health` reports whether weights and auxiliary
> statistics loaded; `/predict` returns the class distribution and the
> out-of-distribution verdict; `/explain` returns the Grad-CAM overlay as a PNG;
> and `/analyze` returns the prediction, the overlay base64-encoded, and the
> SHAP summary in a single JSON response.
>
> `/analyze` exists because the two explanations are intended to be read
> together, and delivering them over separate requests permits an interface to
> render the image and silently discard the numbers. Its SHAP block carries the
> base value, the sum of contributions and the resulting logit, so that a client
> can verify the decomposition it is being shown rather than trusting the bars;
> it also carries the fraction of total attributed movement the listed features
> represent, for the reason given in §4.4.
>
> This service is kept **separate from** the Smart Healthcare Triage System
> rather than embedded in it. The triage backend performs rule-based reasoning
> over symptom text and has no need of a multi-gigabyte deep learning runtime;
> coupling them would impose that cost on every deployment of the triage
> component. The two are designed to interoperate over HTTP, and the radiograph
> service runs on a separate port for that reason.
>
> Every response carries a non-dismissible disclaimer. The user interface
> deliberately does **not** colour-code the disease classes: a green bar reading
> "NORMAL 94%" constitutes an all-clear this system is not entitled to give, so
> all probability bars share one colour and rank by length alone.

---

## §4 Evaluation Protocol — corrected Table 1

**What changed.** The "IoU with annotated regions" cell cannot be honoured with
this dataset and must not be reported as if it were. The masks distributed with
the Radiography Database segment *lungs*, not findings.

> **Table 1. Evaluation dimensions, metrics and controls.** Each metric is
> reported against an explicit null, since none of them is interpretable alone.
>
> | Dimension | Metric | Control it is read against |
> |---|---|---|
> | Classification performance | Accuracy, macro/per-class precision, recall, F1, one-vs-rest AUC | Per-class recall, since accuracy is set by the majority class |
> | Explanation fidelity | Deletion AUC, insertion AUC | The same curves under a random pixel ordering |
> | Perturbation validity | Fraction of perturbation steps rejected by the OOD detector | — (a diagnostic on the two metrics above) |
> | Explanation localisation | Fraction of Grad-CAM mass inside the lung fields | The lung fields' share of image area (enrichment = ratio) |
> | Attribution consistency | Mean pairwise cosine similarity of SHAP vectors within a class | The same statistic over all pairs regardless of class |
> | Attribution compactness | Share of total attributed movement in the top-k features | k / d, the uniform expectation |
> | Shortcut sensitivity | Macro and per-class F1 under lung masking | The identical recipe on unmasked images |
> | Abstention cost | False-reject rate per class at a calibrated percentile | The nominal percentile |
>
> **On localisation.** We report the fraction of Grad-CAM mass falling inside
> the lung fields, *not* intersection-over-union against annotated pathology.
> The distinction is material: the available masks segment the lungs, so a
> heatmap covering both lungs entirely scores perfectly while having localised
> nothing. Because the lung fields occupy roughly a quarter of a chest
> radiograph, an uninformative map already achieves a mass fraction near 0.24,
> and only the enrichment ratio — mass fraction divided by area fraction, where
> 1.0 denotes no better than uniform — carries information. Genuine
> pathology-level IoU requires region annotations this dataset does not carry,
> such as the RSNA Pneumonia Detection Challenge bounding boxes or the 984
> annotated images in NIH ChestX-ray14; we identify this as the principal
> extension of the present protocol.

---

## §4.1 Results — classification (replaces the placeholder)

> All figures are on the held-out test split of 3,175 images (COVID19 542,
> LUNG_OPACITY 902, NORMAL 1,529, PNEUMONIA 202), from `reports/metrics_test.json`
> and `reports/resnet50/metrics_test.json`.
>
> **Table 2. Classification performance, both backbones.**
>
> | Backbone | Params | Macro F1 | Accuracy | Macro AUC | Macro precision | Macro recall |
> |---|---|---|---|---|---|---|
> | ResNet18 | 11.18M | 0.9587 | 0.9524 | 0.9928 | 0.9640 | 0.9536 |
> | ResNet50 | 23.52M | 0.9576 | 0.9512 | 0.9937 | 0.9630 | 0.9527 |
>
> **These are single runs at seed 42**, and are the checkpoints every subsequent
> section analyses, so the per-class breakdown and the explanation metrics all
> refer to these two specific models. §4.2 reports three seeds per backbone and
> should be consulted before any two numbers in this table are compared: the
> run-to-run spread is larger than the difference between the rows.
>
> **Table 3. Per-class results.**
>
> | | ResNet18 P / R / F1 / AUC | ResNet50 P / R / F1 / AUC |
> |---|---|---|
> | COVID19 | 0.989 / 0.976 / 0.982 / 0.9993 | 0.985 / 0.980 / 0.982 / 0.9991 |
> | LUNG_OPACITY | 0.939 / 0.920 / 0.929 / 0.9858 | 0.951 / 0.901 / 0.925 / 0.9887 |
> | NORMAL | 0.944 / 0.963 / 0.953 / 0.9870 | 0.936 / 0.969 / 0.952 / 0.9873 |
> | PNEUMONIA | 0.985 / 0.955 / 0.970 / 0.9992 | 0.980 / 0.960 / 0.970 / 0.9995 |
>
> These figures are comparable to published results on this dataset. **They
> should not be quoted without §4.5**, which establishes that the classes are
> separable by source archive, so a score in this range is what a model would
> produce by learning which repository an image came from.
>
> Lung opacity is the weakest class under both backbones, and the confusion is
> almost entirely with NORMAL: 86 of 902 lung-opacity films are called normal by
> the ResNet50 model, and 39 normal films are called lung opacity. This is the
> expected behaviour for two classes that differ by a graded radiographic
> finding rather than a categorical one.

---

## §4.2 Results — capacity ablation (new section)

> Each configuration was trained three times under the identical two-stage
> recipe, varying only the seed, which governs head initialisation, sampler
> draws and augmentation order. The train/validation/test partition is fixed on
> disk and is therefore *not* resampled, so what follows is training-run
> variance on one split.
>
> **Table 3b. Three seeds per backbone, test split.**
>
> | | seed 42 | seed 1 | seed 2 | Mean | SD | Range |
> |---|---|---|---|---|---|---|
> | ResNet18 macro F1 | 0.9587 | 0.9570 | 0.9556 | **0.9571** | 0.0016 | 0.0031 |
> | ResNet50 macro F1 | 0.9576 | 0.9510 | 0.9558 | **0.9548** | 0.0034 | 0.0065 |
> | ResNet18 macro AUC | 0.9928 | 0.9926 | 0.9919 | 0.9924 | — | — |
> | ResNet50 macro AUC | 0.9937 | 0.9930 | 0.9934 | **0.9933** | — | — |
>
> **The capacity difference is not distinguishable from seed noise.** The means
> differ by −0.0023 macro F1 against a pooled standard deviation of 0.0025, an
> effect of 0.93 standard deviations, and the two ranges overlap substantially
> ([0.9556, 0.9587] against [0.9510, 0.9576]). With three runs per arm this is
> nowhere near separation. For scale, the lung-masking ablation of §4.5 moves
> macro F1 by 0.0246 — an order of magnitude larger than either the backbone
> difference or the noise it sits in.
>
> **This is why the study was necessary rather than tidy.** Comparing the two
> seed-42 runs alone gives −0.0011, less than half the difference between the
> means, because that particular ResNet18 run is the best of its three and that
> particular ResNet50 run the best of its three. A single-run comparison here
> would have reported a number that is out by a factor of two, and depending on
> which pair of runs happened to be trained, could have reported either sign. We
> note this because single-run backbone comparisons are common, and on this task
> the run-to-run spread exceeds the effect being compared.
>
> **ResNet50 is markedly less stable.** Its standard deviation is 0.0034 against
> ResNet18's 0.0016 and its range is twice as wide. The larger model is not
> merely no better here; it is more dependent on initialisation, which is the
> opposite of what additional capacity is usually expected to buy.
>
> **One difference does survive, and it is in AUC rather than F1.** Every
> ResNet50 run scores a higher macro AUC than every ResNet18 run — the ranges do
> not overlap at all, 0.9930–0.9937 against 0.9919–0.9928 — while macro F1 shows
> no such separation. The larger model ranks the classes more reliably and
> converts that ranking into decisions no better, and less consistently. This is
> the same dissociation that appears in §4.5, where masking costs eight times
> more macro F1 than macro AUC, and in §4.3, where the two fidelity metrics rank
> the backbones oppositely: on this data, what separates the classes and what
> places the decision boundary come apart repeatedly.
>
> The conventional reading of a flat capacity curve is that the task saturates.
> A second reading is available given §4.5 and we consider it better supported:
> **if a substantial part of the achievable score is obtainable from acquisition
> signature, the smaller network has already extracted it and additional
> capacity has nothing left to buy.** Under this interpretation the flatness is
> a property of the dataset rather than the task, and would not be expected to
> hold where provenance and label are decorrelated.
>
> We report the comparison chiefly as a caution. A negative capacity ablation is
> frequently presented as evidence that a small model suffices; here it is at
> least equally consistent with the conclusion that neither model is doing what
> the class names suggest.
>
> The conventional reading is that the task saturates at this scale. A second
> reading is available given §4.5, and we consider it better supported: **if a
> substantial part of the achievable score is obtainable from acquisition
> signature, then the smaller network has already extracted it, and additional
> capacity has nothing left to buy.** Under this interpretation the flatness of
> the capacity curve is not a property of the task but of the dataset, and would
> not be expected to hold on data where provenance and label are decorrelated.
>
> We report the comparison chiefly as a caution. A negative capacity ablation is
> frequently presented as evidence that a small model is sufficient; here it is
> at least equally consistent with the conclusion that neither model is doing
> what the class names suggest.

---

## §4.3 Results — explanation fidelity (new section)

> Measured over 200 test images drawn at random from the split (33 COVID19, 48
> LUNG_OPACITY, 102 NORMAL, 17 PNEUMONIA), from `reports/xai_fidelity_test.json`
> and `reports/resnet50/xai_fidelity_test.json`. Both backbones are scored on
> **the identical 200 images under the identical random control orderings**, the
> sampling and the controls being drawn from a seeded generator, so differences
> below are attributable to the models alone.
>
> **Table 4. Deletion and insertion (Petsiuk et al., 2018), each against a
> random pixel ordering on the same image.**
>
> | | ResNet18 | | | ResNet50 | | |
> |---|---|---|---|---|---|---|
> | | Grad-CAM | Random | Gap | Grad-CAM | Random | Gap |
> | Deletion AUC (lower better) | 0.4503 | 0.4400 | +0.0103 | 0.4282 | 0.2738 | **+0.1543** |
> | Insertion AUC (higher better) | 0.7764 | 0.4364 | +0.3400 | **0.8700** | 0.2740 | **+0.5960** |
> | Deletion steps rejected as OOD | | | 59.7% | | | **72.6%** |
>
> **The raw AUCs are not comparable between the two models, and this is the
> first thing the table shows.** ResNet50's predicted probability collapses far
> faster under random perturbation than ResNet18's — its random insertion AUC is
> 0.2740 against 0.4364 — so its curves start from a different place entirely. A
> paper reporting ResNet50's insertion AUC of 0.8700 beside ResNet18's 0.7764
> and concluding that the larger model is better explained would be comparing
> two quantities measured against different baselines. Only the gap over the
> control is interpretable, and by that measure the larger model genuinely is
> better localised: +0.5960 against +0.3400.
>
> **Insertion succeeds and deletion fails, on both models.** Restoring the
> pixels Grad-CAM ranks highest recovers the predicted probability far faster
> than restoring random ones. Removing those same pixels destroys it no faster
> than removing random ones — in fact slower, the deletion gap being positive
> when it should be negative for both networks. The two metrics are constructed
> to agree, and their disagreement requires explanation.
>
> We are able to supply one, and the second backbone turns it from a conjecture
> into a tested prediction. Blanking pixels produces an image unlike any
> radiograph, so a probability that falls under deletion may report that the
> input is no longer a chest X-ray rather than that the evidence has been
> removed. This objection is well known and is normally left as a caveat. Here
> it is measurable, because each model carries a detector for exactly this
> condition (§4.6): **59.7% of ResNet18's deletion steps and 72.6% of
> ResNet50's are rejected as out-of-distribution by the respective model's own
> Mahalanobis check.**
>
> The prediction this licenses is that the model whose perturbed images are more
> off-distribution should have the more corrupted deletion metric, and that is
> what is observed: ResNet50 has both the higher rejection rate (72.6% against
> 59.7%) and the deletion gap that is further in the wrong direction (+0.1543
> against +0.0103), while simultaneously being the *better*-localised model on
> insertion. Deletion and insertion rank the two networks in opposite orders,
> and the out-of-distribution rate says which ranking to believe.
>
> We therefore recommend that deletion-style metrics be reported alongside a
> distributional validity statistic, and that insertion be preferred where only
> one can be reported. The diagnostic costs nothing beyond a forward pass
> wherever an out-of-distribution detector is already present, and without it a
> deletion AUC cannot be distinguished from a measurement of how brittle the
> model is to blur.
>
> **Table 5. Localisation within the lung fields.**
>
> | Quantity | ResNet18 | ResNet50 |
> |---|---|---|
> | Grad-CAM mass inside the lung fields | 0.324 | 0.304 |
> | Lung fields as a share of image area | 0.238 | 0.238 |
> | **Enrichment** | **1.359** | **1.274** |
>
> Both models' heatmaps concentrate on the lungs, and both do so modestly — 1.36
> and 1.27 times what an uninformative map achieves. Reported alone, a mass
> fraction of 0.324 would convey a misleading impression of poor localisation;
> reported without the area baseline it would convey nothing at all. Per §4,
> this is lung-field mass and not pathology IoU. Note that the larger model,
> which is better localised on insertion, is *slightly worse* by this measure —
> consistent with lung-field containment and evidence localisation being
> different properties.

---

## §4.4 Results — attribution consistency and compactness (new section)

> **Table 6. SHAP vector similarity, same 200 images, both backbones.**
>
> | Mean pairwise cosine | ResNet18 (512-d) | ResNet50 (2048-d) |
> |---|---|---|
> | Within class | 0.6076 | 0.3118 |
> | All pairs (control) | 0.3142 | 0.1479 |
> | **Ratio** | **1.93** | **2.11** |
>
> **The absolute figures halve between the two models and the ratio does not
> move.** This is the clearest demonstration in the paper of why the control is
> load-bearing. Cosine similarity between high-dimensional vectors falls as
> dimension rises — 2048-dimensional attributions are more spread out than
> 512-dimensional ones simply as geometry — so a within-class cosine of 0.31
> read on its own would suggest ResNet50's explanations are half as consistent
> as ResNet18's. Measured against its own control, the consistency is if
> anything marginally higher. **An absolute attribution-similarity figure is not
> comparable across architectures and should not be reported without its
> null.**
>
> The control is also necessary within a single model. Every SHAP vector in this
> construction is w ⊙ (x − E[x]) for one shared w, so any two vectors agree in
> direction before anything about the underlying images is considered; a
> within-class figure alone is substantially an artefact of the method. At
> roughly twice the control on both networks, the within-class consistency is
> real.
>
> **Table 7. Attribution compactness — share of total attributed movement in the
> fifteen largest contributions.**
>
> | | ResNet18 | ResNet50 |
> |---|---|---|
> | Top-15 coverage | 15.1% | 14.2% |
> | Features | 512 | 2048 |
> | Uniform expectation | 2.93% | 0.73% |
> | Concentration over uniform | 5.2x | **19.3x** |
>
> The two models look alike in the first row and are very different underneath
> it. ResNet50 spreads its attribution over four times as many features, so
> reaching a comparable 14.2% in fifteen of them represents far greater
> concentration relative to chance. Both readings are needed: the raw coverage
> governs how a chart should be captioned, the concentration governs whether the
> representation is distributed.
>
> This is the least comfortable result in the paper and it bears directly on how
> such figures are presented. **A bar chart of the fifteen largest SHAP values
> is a sample of the model's reasoning, not a summary of it.** A reader shown
> fifteen bars will reasonably infer they constitute the explanation; on either
> network they constitute about a seventh of it. Our `/analyze` endpoint returns
> this coverage fraction alongside the attributions so a client cannot present
> them as "the reason" without contradicting the payload it received, and we
> suggest any published per-prediction SHAP chart carry the equivalent figure.

---

## §4.5 Results — the provenance confound (new section; the paper's core)

> **The classes in this dataset are separable by source before any lung is
> examined.** Every COVID-19 image in the COVID-19 Radiography Database
> originates from BIMCV, Eurorad, SIRM or a GitHub collection; every normal and
> viral pneumonia image originates from Kaggle-hosted collections. The overlap
> is empty. Scanner, exposure, collimation, burned-in annotation and post-
> processing character all carry that origin, so a model can achieve a high
> score by identifying the repository rather than the disease, and would exhibit
> no symptom of having done so on any metric in §4.1.
>
> **The masking ablation.** We retrained the identical recipe on a mirror of the
> same split with every non-lung pixel zeroed using the shipped segmentation
> masks — approximately 77% of each image removed, including all burned-in
> markers, collimation edges, soft tissue and background. The split is mirrored
> rather than redrawn, so the two runs differ in exactly one variable.
>
> **Table 8. Cost of removing everything outside the lungs (ResNet18).**
>
> | | Macro F1 | Accuracy | Macro AUC |
> |---|---|---|---|
> | As distributed | 0.9587 | 0.9524 | 0.9928 |
> | Lungs only | 0.9341 | 0.9298 | 0.9898 |
>
> | Per-class loss | COVID19 | LUNG_OPACITY | NORMAL | PNEUMONIA |
> |---|---|---|---|---|
> | F1 | **−0.056** | −0.030 | −0.009 | −0.003 |
> | AUC | −0.0040 | −0.0057 | −0.0015 | −0.0011 |
>
> The score does not collapse, which **rules out** the crude shortcut: the model
> is not simply reading annotations or background, because those are gone and it
> still achieves 0.934. But the per-class F1 losses are grossly unequal.
> COVID-19 loses six times what the normal class loses and nearly twenty times
> what pneumonia loses, and its recall falls from 0.976 to 0.911. It is the only
> class with unique provenance, and it is the class with the most to lose.
> An earlier three-class version of this experiment found the same ordering at
> half the magnitude, so the effect has reproduced across two class definitions.
>
> **The AUC row qualifies this and should be reported with it.** Macro AUC falls
> only 0.0030 where macro F1 falls 0.0246, and on AUC the per-class ordering does
> not hold — lung opacity loses marginally more than COVID-19. Most of what
> masking costs the COVID-19 class is therefore the placement of the decision
> boundary rather than the separability of the class itself. We state this
> because the F1 row alone supports a stronger claim than the evidence warrants.
>
> **Neither explanation detects any of this, and this is the paper's central
> negative result.** Acquisition signature is present *inside* the lung fields —
> in noise characteristics and processing response — as is the lung silhouette
> itself, and a paediatric chest differs in outline from an adult one, which
> alone separates the paediatric pneumonia collection from the adult European
> COVID-19 series. A model reading provenance rather than pathology therefore
> produces heatmaps that fall on the lungs and look entirely reasonable. Our
> enrichment figure of 1.359 is consistent with a model reading pathology and
> equally consistent with one reading scanner. SHAP fares no better: its
> baseline is the mean training feature vector, so it measures deviation from
> the average image *in this dataset*, and a large contribution is as consistent
> with "unlike the typical scanner here" as with "unlike a healthy lung".
>
> **Heat on the lungs is necessary, not sufficient.** The verification role
> routinely assigned to Grad-CAM in the clinical XAI literature — that a
> clinician can confirm the model used genuine evidence by inspecting the map —
> is not supported by these measurements. The confound that most threatens this
> dataset is invisible to both explanation methods applied to it.
>
> **Cross-dataset validation, and why it does not settle the question.** Scoring
> against a second public dataset appears to be the obvious remedy, and done
> naively it measures nothing: `prashant268/chest-xray-covid19-pneumonia` shares
> **24.0%** of its 6,432 images with this model's training split, and **none of
> those matches by checksum**, every copy having been resized or re-encoded in
> compilation. A hash comparison reports two independent datasets. After
> partitioning:
>
> | | Images | Macro F1 | Accuracy |
> |---|---|---|---|
> | All, contaminated | 6,432 | 0.9492 | 0.9667 |
> | Overlapping only (control) | 1,545 | 0.9864 | 0.9903 |
> | **Clean** | **4,887** | **0.9252** | 0.9593 |
>
> Leaving the contamination in was worth a spurious +0.0240 macro F1. But
> **this is still not the required experiment.** Both datasets are compiled from
> the same public archives, so much of the clean remainder plausibly originates
> from the same collections as the training data. What this establishes is that
> the model does not collapse on unseen images from a differently assembled
> compilation. It does not establish that the model reads pathology. Only a
> COVID-19 cohort from hospitals where provenance does not predict the label can
> settle that, and we identify it as the necessary next experiment.

---

## §4.6 Results — abstention (new section)

> A softmax over a fixed class list normalises whatever it is given, so an input
> unlike anything in training does not return an uncertain answer; it returns a
> confident wrong one. Against the trained ResNet18, a flat grey square is
> classified COVID-19 at 99.42%, uniform noise at 100.00%, and a page of text at
> 99.98% — none flagged by any confidence threshold, because there is no
> uncertainty present to threshold.
>
> We fit a Mahalanobis-distance detector over the penultimate features following
> Lee et al. (2018): one Gaussian per class with a tied, Ledoit-Wolf-shrunk
> covariance, fitted on the training split and calibrated on validation.
> **Thresholds are per class**, applied by whichever class an input is nearest.
> This is not a detail: a single pooled 95th-percentile cutoff, measured on the
> three-class model, reported a reassuring 4.5% false-reject rate while actually
> rejecting 30.2% of genuine pneumonia films and 0.4% of normal ones, the pooled
> figure having been set by the majority class.
>
> **Table 9. Calibrated cutoffs and the cost of the check.**
>
> | | ResNet18 cutoff | Rejected (test) | Rejected (2nd dataset) | ResNet50 cutoff | Rejected (test) |
> |---|---|---|---|---|---|
> | COVID19 | 939.8 | 6.6% | 9.2% | 4663.8 | 3.9% |
> | LUNG_OPACITY | 822.5 | 5.1% | not labelled | 4084.0 | 5.5% |
> | NORMAL | 653.4 | 4.1% | 7.3% | 2602.2 | 4.6% |
> | PNEUMONIA | 1310.4 | 6.4% | 6.3% | 6436.5 | **8.9%** |
> | Pooled | — | 5.0% | 6.7% | — | 5.0% |
>
> All six non-radiograph probes are rejected by a wide margin. Three properties
> of this table are worth stating explicitly.
>
> **The cutoffs are not comparable between the two models.** Mahalanobis
> distance is measured in 512 dimensions for ResNet18 and 2048 for ResNet50, so
> the raw thresholds differ by roughly a factor of four for reasons that have
> nothing to do with detection quality. Only rejection rates transfer.
>
> **The pooled rate is 5.0% for both models by construction**, being the
> complement of the 95th-percentile calibration, and is therefore uninformative.
> The per-class rows are where the two models differ, and they differ in both
> directions: ResNet50 rejects fewer COVID-19 films (3.9% against 6.6%) and
> substantially more pneumonia films (8.9% against 6.4%). Pneumonia is the
> smallest class in the training set, and the larger model's abstention is
> hardest on it — a cost invisible in the pooled figure and the same trap, one
> level down, as reading accuracy instead of per-class recall.
>
> **The threshold does not transfer between datasets**: a cutoff calibrated at
> the 95th percentile on one delivers approximately the 93rd on another, and
> that second dataset is not even from different hospitals. Recalibration
> against images from the intended deployment site is required, or the detector
> rejects more genuine films than the nominal percentile implies.
>
> **What abstention does not solve.** The detector answers "unlike the training
> images", which is not "not a chest X-ray" and is much further from "the model
> cannot handle this". Lung opacity measured that gap: of 600 such films, real
> radiographs from the same repositories showing a finding the then-three-class
> model had no output for, only 34.5% were rejected. The remainder were accepted
> and then classified **NORMAL 94.2% of the time at a mean confidence of 0.972.**
> That measurement is why lung opacity is a class in the present model. It does
> not close the gap. Effusion, pneumothorax, nodules and fibrosis remain real
> radiographs that resemble the training data, and will be accepted and assigned
> the nearest available class — disproportionately, on this evidence, the one a
> reader is most likely to act on. **NORMAL here means "not the other three",
> never "clear".**

---

## §5 Discussion (full replacement)

> This work set out to add a dual explainability layer to a chest radiograph
> classifier and to evaluate it. The evaluation produced a less comfortable
> result than the design anticipated, and we think the discrepancy is the most
> useful thing in the paper.
>
> **The explanations work, in the narrow sense in which they can be tested.**
> Grad-CAM's insertion AUC is far above a random ordering, its heat is enriched
> on the lung fields, and SHAP attributions are roughly twice as consistent
> within a class as across classes. On every measurement we could construct, the
> two methods are doing something real.
>
> **They do not do the job the literature assigns them.** The justification for
> adding XAI to a clinical classifier is normally that a clinician can inspect
> the explanation and detect a model relying on a spurious correlate. On this
> dataset the dominant spurious correlate is source archive, we can demonstrate
> by ablation that the model uses it, and neither explanation shows any sign of
> it. Grad-CAM cannot, because acquisition signature is present within the lung
> fields, so a shortcut-driven model produces an anatomically plausible map.
> SHAP cannot, because its baseline is drawn from the same confounded
> distribution. A clinician following the recommended procedure would inspect a
> reasonable-looking heatmap and a coherent attribution profile and conclude,
> incorrectly, that the prediction was grounded.
>
> This is not an argument against explainability. It is an argument that
> explanation quality and dataset validity are separate axes, that a system can
> score well on the first while failing on the second, and that XAI work on
> medical imaging should report a provenance audit alongside its explanations
> rather than treating explanation as the audit.
>
> **On methodology.** Three of our findings concern measurement rather than this
> model, and having two backbones is what makes them checkable rather than
> anecdotal.
>
> First, the deletion metric is unreliable where perturbation drives inputs
> off-distribution, and the condition is detectable at no extra cost wherever an
> OOD detector exists. The two networks provide a test rather than an assertion:
> the one with more off-distribution perturbations has the more corrupted
> deletion result, while being the better-localised model on insertion. We
> recommend reporting the rejected fraction beside any deletion AUC.
>
> Second, **explanation metrics are not comparable across architectures in raw
> form, and this is easy to get wrong.** ResNet50's raw insertion AUC of 0.8700
> exceeds ResNet18's 0.7764, but its random control is 0.2740 against 0.4364, so
> the two numbers are measured from different origins. The same applies to
> attribution similarity, where the within-class cosine halves between the two
> models purely because 2048-dimensional vectors are more spread out than
> 512-dimensional ones, while the ratio to the control barely moves (1.93 to
> 2.11). A comparison of raw figures would report the larger model as both
> better explained and less consistent; both conclusions would be artefacts.
>
> Third, per-prediction SHAP bar charts should carry the share of total
> attributed movement they represent. At 15.1% and 14.2% for a fifteen-bar chart
> on our two models, the difference between a chart that is a summary and one
> that is a sample is not something a reader can infer from the chart.
>
> **Limitations.** The provenance confound cannot be resolved within this
> dataset, because the correlation is total by construction; the necessary
> experiment requires a COVID-19 cohort from hospitals where source does not
> predict label. Labels derive from dataset compilers under varying and largely
> undocumented criteria, so agreement with them is not agreement with a
> diagnosis. Class balance does not reflect prevalence, so no output estimates
> the probability that a patient has anything. The localisation metric uses lung
> masks rather than pathology annotations. Explanation fidelity was measured on
> 200 images per backbone at 50 perturbation steps, on the seed-42 checkpoints
> only — the seed study of §4.2 covers classification metrics, not explanation
> metrics, so we cannot say how much of the fidelity difference between the two
> backbones is itself run-to-run variance. The two-backbone agreement on the
> deletion/OOD relationship is likewise two points rather than a trend. Three
> seeds per arm supports the claim that the capacity difference is not
> distinguishable from noise, but is too few to estimate that noise precisely.
> And no clinician evaluation was conducted: the
> usability study proposed in the original design remains outstanding, and we
> note that our findings raise a specific question for it — whether clinicians
> shown a plausible heatmap from a shortcut-driven model correctly withhold
> trust. Our results predict they would not.

---

## §6 Conclusion (full replacement)

> We built a four-class chest radiograph classifier with a dual explainability
> pipeline — Grad-CAM for spatial explanation, exact closed-form SHAP for
> feature attribution — served through a single endpoint, and then measured what
> the explanations establish. Both backbones reach a macro F1 near 0.958 and a
> macro AUC near 0.993. Both explanations pass the fidelity tests we could
> construct. Neither detects the shortcut we can independently prove the model
> uses, and a 2.1x increase in capacity buys nothing distinguishable from seed
> noise across three runs per backbone — while doubling the variance between
> runs — which is what one expects when the shortcut is already exhausted.
>
> We also report two methodological findings of wider applicability: that the
> deletion metric is confounded by distribution shift in a way that is
> measurable using an out-of-distribution detector, and that per-prediction SHAP
> charts should state the fraction of attribution they represent.
>
> Future work is, in priority order: validation on a chest radiograph cohort in
> which acquisition source does not predict the label, which is the only
> experiment that can settle the central question; explanation fidelity against
> genuine pathology annotations, using RSNA or NIH bounding boxes; and a
> clinician study designed specifically to test whether a plausible explanation
> from a shortcut-driven model induces unwarranted trust.

---

## References — corrections

Replace placeholder [7]. Add [8]–[10], cited above.

> [7] Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W.,
> Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural
> networks. *Nature Machine Intelligence*, 2(11), 665–673.
>
> [8] Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized Input Sampling
> for Explanation of Black-box Models. *British Machine Vision Conference
> (BMVC)*.
>
> [9] Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A Simple Unified Framework
> for Detecting Out-of-Distribution Samples and Adversarial Attacks. *Advances
> in Neural Information Processing Systems (NeurIPS)*.
>
> [10] DeGrave, A. J., Janizek, J. D., & Lee, S.-I. (2021). AI for radiographic
> COVID-19 detection selects shortcuts over signal. *Nature Machine
> Intelligence*, 3(7), 610–619.

**Note on [10].** This is the closest prior work to §4.5 and should be
positioned in §2.3 rather than only cited: DeGrave et al. demonstrate shortcut
learning in COVID-19 radiograph classifiers using saliency methods and a
generative approach. The present contribution relative to it is the ablation-
plus-capacity evidence, and specifically the finding that the dual XAI pipeline
routinely proposed as the remedy does not surface the problem.

Also recommended for §2.2: Adebayo et al. (2018), *Sanity Checks for Saliency
Maps* (NeurIPS), which motivates the control-based reporting used throughout §4.
