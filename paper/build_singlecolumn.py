"""Build the corrected paper as a .docx, from paper/draft_sections.md content.

Writes a NEW file beside the original; the original is not modified.
"""
import os
import re
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches, RGBColor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'Explainable_ChestXray_Paper_singlecolumn.docx')

doc = Document()

# Match the original: US Letter, 1" margins.
for s in doc.sections:
    s.page_width, s.page_height = Inches(8.5), Inches(11)
    s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)

normal = doc.styles['Normal']
normal.font.name = 'Times New Roman'
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

for name, size in (('Heading 1', 14), ('Heading 2', 12), ('Heading 3', 11)):
    st = doc.styles[name]
    st.font.name = 'Times New Roman'
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.paragraph_format.space_before = Pt(12)
    st.paragraph_format.space_after = Pt(4)


def rich(par, text):
    """Add text to a paragraph, honouring **bold** markers."""
    for chunk in re.split(r'(\*\*[^*]+\*\*)', text):
        if not chunk:
            continue
        run = par.add_run(chunk[2:-2] if chunk.startswith('**') else chunk)
        run.bold = chunk.startswith('**')
    return par


def para(text='', style=None, align=None, italic=False, size=None, bold=False):
    p = doc.add_paragraph(style=style)
    if text:
        rich(p, text)
    if align is not None:
        p.alignment = align
    for r in p.runs:
        if italic:
            r.italic = True
        if size:
            r.font.size = Pt(size)
        if bold:
            r.bold = True
    return p


def heading(text, level):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p


def bullets(items):
    for it in items:
        p = doc.add_paragraph(style='List Bullet')
        rich(p, it)
        p.paragraph_format.space_after = Pt(3)


def table(headers, rows, caption=None, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    t.autofit = True
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9)
        r.font.name = 'Times New Roman'
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i else WD_ALIGN_PARAGRAPH.LEFT
            rich(p, str(v))
            for r in p.runs:
                r.font.size = Pt(9)
                r.font.name = 'Times New Roman'
    if caption:
        c = para(caption, align=WD_ALIGN_PARAGRAPH.LEFT, italic=True, size=9)
        c.paragraph_format.space_before = Pt(3)
        c.paragraph_format.space_after = Pt(10)
    return t


# ----------------------------------------------------------------- title block

t = para('Explainable Deep Learning for Multi-Class Chest X-ray Classification: '
         'Measuring What Grad-CAM and SHAP Actually Establish',
         align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16)
t.paragraph_format.space_after = Pt(4)

para('An audit of a dual-explainability pipeline, and of the dataset confound neither method detects',
     align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=11)
para('Ardhendu', align=WD_ALIGN_PARAGRAPH.CENTER, size=11)
para('Department of Computer Science Engineering (AI & ML), Jain University',
     align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=10)

# -------------------------------------------------------------------- abstract

heading('Abstract', 1)
para('Convolutional neural networks achieve high reported accuracy on chest radiograph '
     'classification, and explainable AI (XAI) methods are routinely added to such systems on the '
     'argument that a visible explanation lets a clinician verify the model is using genuine '
     'pathology rather than a spurious correlate. This paper builds that argument\u2019s standard '
     'implementation \u2014 a transfer-learned CNN with Grad-CAM for spatial explanation and SHAP for '
     'feature-level attribution, served through a FastAPI endpoint \u2014 and then measures whether it '
     'does what it is claimed to do.')
para('We train ResNet18 and ResNet50 classifiers on the COVID-19 Radiography Database (21,165 '
     'images; four classes: normal, pneumonia, COVID-19, lung opacity) using a two-stage '
     'fine-tuning schedule and a patient-grouped 70/15/15 split, three seeds each. Macro F1 is '
     '0.9571 \u00b1 0.0016 for ResNet18 and 0.9548 \u00b1 0.0034 for ResNet50 \u2014 a 2.1\u00d7 increase in '
     'parameters that is indistinguishable from seed noise and doubles the run-to-run variance. '
     'SHAP values are computed in closed form over the network\u2019s penultimate features, which is '
     'exact for a linear classification head and is verified against the reference implementation.')
para('Three measurements qualify these results. First, the dataset\u2019s classes are completely '
     'separable by source archive before any lung is examined, and retraining on lung-masked '
     'images \u2014 77% of each image removed \u2014 costs COVID-19 six times the F1 it costs the normal '
     'class, identifying acquisition signature rather than pathology as a substantial part of what '
     'was learned. Second, increasing backbone capacity by 2.1\u00d7 produces no change distinguishable '
     'from seed noise across three runs per backbone, and doubles the run-to-run variance, which is '
     'what one expects when a shortcut has already been fully exploited. Third, and most '
     'importantly for the XAI claim, neither explanation detects any of this: Grad-CAM produces '
     'anatomically plausible maps regardless, and SHAP measures deviation from a training-set mean '
     'that carries the same confound.')
para('We further show that the standard deletion metric for explanation fidelity is unreliable on '
     'this data, and quantify why: 59.7% of its perturbation steps for ResNet18 and 72.6% for '
     'ResNet50 are rejected as out-of-distribution by the respective model\u2019s own Mahalanobis '
     'detector, so the metric partly measures distribution shift rather than explanation quality. '
     'The two backbones test this directly \u2014 the model with the higher rejection rate is also the '
     'one whose deletion result is further in the wrong direction, while being the better localised '
     'of the two on insertion, so the two metrics rank the networks in opposite orders and the '
     'rejection rate says which ranking to believe. Insertion AUC, which lacks this defect, places '
     'Grad-CAM well above a random ordering on both networks (+0.34 and +0.60 over their respective '
     'controls). We conclude that visual and feature-level explanations are necessary but '
     'demonstrably insufficient for the verification role assigned to them; that explanation '
     'metrics must be reported against per-model controls, since neither raw AUCs nor raw '
     'attribution similarities are comparable across architectures; and that dataset provenance '
     'auditing must accompany rather than follow explainability work.')
p = para()
rich(p, '**Keywords:** Explainable AI, Chest X-ray Classification, Grad-CAM, SHAP, Shortcut '
        'Learning, Dataset Bias, Out-of-Distribution Detection, Explanation Fidelity')

# ---------------------------------------------------------------- introduction

heading('1. Introduction', 1)
para('Chest radiography remains one of the most widely used diagnostic tools for detecting '
     'pulmonary conditions such as pneumonia, COVID-19 and other thoracic abnormalities, '
     'particularly where access to specialist radiologists is limited. Convolutional neural '
     'networks pretrained on large image corpora and fine-tuned by transfer learning have '
     'demonstrated classification accuracy rivalling human readers on several benchmark datasets.')
para('Despite this, clinical adoption has been slow, largely because these models function as '
     'black boxes: a clinician receives a label and a confidence score with no insight into which '
     'visual evidence drove the decision. This creates a specific risk \u2014 that a model relies on '
     'spurious artifacts such as text markers, positioning hardware or scanner-specific noise '
     'rather than genuine pathology. Explainable AI addresses this by producing human-interpretable '
     'justifications alongside predictions, on the argument that a clinician can then verify, trust '
     'or challenge a diagnosis.')
para('This paper takes that argument seriously enough to test it. We build the standard '
     'implementation and then measure, rather than assert, what its explanations establish. The '
     'specific contributions are as follows.')
bullets([
    'A four-class chest radiograph classifier (normal, pneumonia, COVID-19, lung opacity) trained '
    'by two-stage transfer learning, reported at two backbone capacities and three seeds each, so '
    'that the effect of model size can be separated from the effect of the data and from '
    'run-to-run noise.',
    'A dual explainability pipeline combining Grad-CAM for spatial explanation with SHAP for '
    'feature-level attribution, in which the SHAP values are **exact rather than approximated**: '
    'because the classification head is a single linear layer over the penultimate representation, '
    'the Shapley values admit a closed form that we verify against the reference implementation.',
    '**A quantitative audit of both explanations** using deletion and insertion curves, lung-field '
    'localisation, and attribution stability, each reported against an explicit null control '
    'rather than in isolation.',
    '**The finding that the deletion metric is confounded by distribution shift on this data**, '
    'together with a method for detecting that condition using the classifier\u2019s own '
    'out-of-distribution detector.',
    '**A dataset provenance audit** showing that the classes are separable by source archive, '
    'quantified by a lung-masking ablation and a capacity ablation, and the demonstration that '
    'neither explanation method surfaces this.',
    'An abstention mechanism based on Mahalanobis distance in feature space, with per-class '
    'calibration, and the measurement that its threshold does not transfer between datasets.',
])

# ---------------------------------------------------------------- related work

heading('2. Related Work', 1)
heading('2.1 Deep Learning for Chest X-ray Classification', 2)
para('Transfer learning from ImageNet-pretrained CNN architectures, including ResNet, DenseNet and '
     'VGG variants, is the standard approach for chest X-ray classification given the limited size '
     'of labelled medical imaging datasets relative to natural image corpora. Prior work has '
     'applied these architectures to pneumonia, COVID-19 and other pulmonary conditions, generally '
     'reporting accuracies in the 83\u201390 percent range depending on the disease set and evaluation '
     'methodology.')
heading('2.2 Explainability Techniques in Medical Imaging', 2)
para('Grad-CAM [1] is among the most widely used visual explanation techniques for CNN-based '
     'medical image classifiers, producing coarse localisation maps over the regions most '
     'responsible for a prediction. Complementary approaches such as LIME [6] and SHAP [2] provide '
     'feature-level attributions. Several studies combine multiple XAI techniques and evaluate the '
     'resulting explanations against radiologist-identified regions of interest. Adebayo et al. '
     '[11] show that several saliency methods pass visual inspection while failing basic sanity '
     'checks, which motivates the control-based reporting we adopt throughout Section 4.')
heading('2.3 Research Gap', 2)
para('Two gaps motivate this work. First, while individual XAI techniques have been applied to '
     'chest X-ray diagnosis, comparatively few studies integrate a dual spatial and '
     'feature-attribution pipeline within an end-to-end serving workflow. Second, and more '
     'importantly, explanation quality is usually asserted from visual plausibility rather than '
     'measured against controls. DeGrave et al. [10] demonstrate shortcut learning in COVID-19 '
     'radiograph classifiers using saliency and generative methods; our contribution relative to '
     'that work is the ablation-plus-capacity evidence, and specifically the finding that the dual '
     'XAI pipeline routinely proposed as the remedy does not surface the problem.')

# ----------------------------------------------------------------- methodology

heading('3. Proposed Methodology', 1)
heading('3.1 System Overview', 2)
para('The system comprises four stages: (1) data acquisition and preprocessing, (2) classification '
     'using a fine-tuned residual network, (3) a dual explainability pipeline combining Grad-CAM '
     'and SHAP, and (4) exposure of predictions and both explanations through a FastAPI service. A '
     'fifth component, an out-of-distribution detector, allows the system to abstain and is also '
     'used in Section 4.3 as a diagnostic on the fidelity metrics themselves.')

heading('3.2 Dataset and Preprocessing', 2)
para('**Dataset.** All experiments use the COVID-19 Radiography Database, a publicly available '
     'compilation of 21,165 de-identified chest radiographs distributed through Kaggle. We use '
     'four classes: NORMAL (10,192), LUNG_OPACITY (6,012), COVID19 (3,616) and PNEUMONIA (1,345). '
     'Lung opacity is retained as a class in its own right and never merged into pneumonia, since '
     'it denotes a broader radiographic finding; Section 4.6 reports what happened when it was '
     'omitted. The database also ships lung segmentation masks for all four classes, used in '
     'Sections 4.3 and 4.5.')
para('**Splitting.** Images are partitioned 70/15/15 into training (14,814), validation (3,176) '
     'and test (3,175) sets. The split is **grouped by patient**: several constituent collections '
     'contain multiple films of the same chest under systematic filenames, and splitting those at '
     'random places the same patient in both training and test, inflating the reported score. The '
     'publishers\u2019 own train/test division is therefore pooled and redone rather than used as '
     'distributed.')
para('**Preprocessing.** Images are converted to greyscale and replicated to three channels \u2014 '
     'chest radiographs carry no colour, but ImageNet-pretrained backbones have a three-channel '
     'input convolution whose weights are discarded if it is replaced \u2014 resized to 224\u00d7224, and '
     'normalised with ImageNet channel statistics.')
para('**Augmentation** (training split only) comprises random resized cropping (scale 0.85\u20131.0), '
     'rotation up to 10 degrees, and brightness and contrast jitter of 0.15, standing in '
     'respectively for framing, patient positioning and exposure differences between machines and '
     'operators. **Horizontal flipping is deliberately excluded.** It is the default augmentation '
     'for natural images and appears in most published chest X-ray pipelines, but a mirrored chest '
     'places the heart on the right, an anatomy occurring in roughly 1 in 10,000 people. Training '
     'a model to treat it as unremarkable is not a defensible trade for a marginal increase in '
     'effective dataset size.')
para('**Class imbalance** is addressed by a weighted random sampler that draws each class '
     'approximately equally often. Inverse-frequency loss weighting is implemented as an '
     'alternative; the two are not combined, since applying both over-corrects toward the rare '
     'classes.')

heading('3.3 Model Architecture', 2)
para('**Backbones.** We report two ImageNet-pretrained backbones, ResNet18 and ResNet50, each with '
     'its 1000-way ImageNet head replaced by a fresh four-way linear layer, giving 11.18M and '
     '23.52M parameters respectively \u2014 a factor of 2.1. Reporting both is not redundancy: Section '
     '4.2 uses the comparison as a capacity ablation, and the near-identical result is itself '
     'evidence about the dataset.')
para('**Two-stage fine-tuning.** The backbone is first frozen and only the new head is trained '
     '(learning rate 3\u00d710\u207b\u2074), then the whole network is unfrozen and training continues from those '
     'weights at a reduced rate (1\u00d710\u207b\u2074). Fine-tuning an entire residual network against a few '
     'thousand images per class from the outset largely memorises them. The second stage restores '
     'weights only: optimizer state is not carried across, since AdamW moments accumulated while '
     'the backbone was frozen do not describe the parameters the second stage unfreezes, and the '
     'cosine schedule belongs to the epoch budget of the run that declared it.')
para('**Optimisation.** AdamW, weight decay 10\u207b\u2074, cosine-annealed learning rate, batch size 32, '
     'early stopping with patience 5.')
para('**Model selection is on validation macro F1, not accuracy.** With COVID-19 at roughly a '
     'tenth of the normal-class count, a model that never predicts the rarest class can still post '
     'a high accuracy, and selection on accuracy would faithfully preserve that model. Macro F1 '
     'averages the per-class scores equally, making an ignored class expensive.')

heading('3.4 Explainability Pipeline', 2)
para('**Grad-CAM.** Gradient-weighted Class Activation Mapping is applied at the final '
     'convolutional block (7\u00d77 for a 224-pixel input), weighting each feature map by the mean '
     'gradient of the target class score with respect to it, rectifying, and bilinearly upsampling '
     'to the input resolution. Explanations can be requested for a class other than the predicted '
     'one, which answers \u201cwhy did the model not say pneumonia\u201d rather than only \u201cwhy did it say '
     'COVID-19\u201d.')
para('**SHAP over learned features.** SHAP is applied to the penultimate representation \u2014 512 '
     'dimensions for ResNet18, 2048 for ResNet50 \u2014 rather than to pixels. This is deliberate. '
     'Pixel-level SHAP produces another spatial map, which would give the system two answers to '
     'where and none to how much; attributing over the features the classifier actually reads '
     'gives a genuinely different view.')
para('This choice also makes the attribution exact. The head is a single linear layer, so the '
     'class score is w\u00b7x + b, and for a linear model the Shapley values have a closed form: '
     '\u03c6\u1d62 = w\u1d62(x\u1d62 \u2212 E[x\u1d62]), with base value w\u00b7E[x] + b, where E[x] is the mean feature vector over '
     'the training split. No sampling and no convergence criterion are involved. The decomposition '
     'satisfies the Shapley efficiency axiom exactly \u2014 the contributions plus the base value '
     'reconstruct the logit \u2014 and our implementation is verified to agree with the reference '
     'LinearExplainer of Lundberg and Lee [2] to floating-point precision. Attributions may also be '
     'computed for the margin between two classes, which is the quantity an argmax decision '
     'actually turns on.')
para('**Four limitations of the SHAP component**, stated here because they bear directly on how '
     'Section 4.4 should be read. (i) A feature index is not a clinical concept: \u201cfeature 453 '
     'contributed +0.44 logits\u201d is a true statement about the network and conveys nothing to a '
     'radiologist, and none of the dimensions corresponds to a named finding. (ii) The '
     'interventional formulation treats features as independent, which convolutional channels are '
     'not; this is the standard assumption and remains an assumption. (iii) Logits are explained, '
     'not probabilities, since softmax is not additive. (iv) **The baseline is the mean training '
     'image, so the attribution inherits whatever the training set has in common \u2014 including its '
     'provenance.** Section 4.5 shows why this matters.')

heading('3.5 System Integration', 2)
para('Classification and both explanations are served from a FastAPI application with four '
     'endpoints: /health reports whether weights and auxiliary statistics loaded; /predict returns '
     'the class distribution and the out-of-distribution verdict; /explain returns the Grad-CAM '
     'overlay as a PNG; and /analyze returns the prediction, the overlay base64-encoded, and the '
     'SHAP summary in a single JSON response.')
para('/analyze exists because the two explanations are intended to be read together, and '
     'delivering them over separate requests permits an interface to render the image and silently '
     'discard the numbers. Its SHAP block carries the base value, the sum of contributions and the '
     'resulting logit, so a client can verify the decomposition it is being shown rather than '
     'trusting the bars; it also carries the fraction of total attributed movement the listed '
     'features represent, for the reason given in Section 4.4.')
para('This service is kept **separate from** the companion Smart Healthcare Triage System rather '
     'than embedded in it. The triage backend performs rule-based reasoning over symptom text and '
     'has no need of a multi-gigabyte deep learning runtime; coupling them would impose that cost '
     'on every deployment of the triage component. The two interoperate over HTTP, and the '
     'radiograph service runs on a separate port for that reason.')
para('Every response carries a non-dismissible disclaimer. The user interface deliberately does '
     '**not** colour-code the disease classes: a green bar reading \u201cNORMAL 94%\u201d constitutes an '
     'all-clear this system is not entitled to give, so all probability bars share one colour and '
     'rank by length alone.')

# ---------------------------------------------------------- evaluation protocol

heading('4. Evaluation Protocol and Results', 1)
para('Each metric below is reported against an explicit null, since none of them is interpretable '
     'alone. Table 1 summarises the dimensions and their controls.')
table(
    ['Dimension', 'Metric', 'Control it is read against'],
    [
        ['Classification performance', 'Accuracy, macro/per-class precision, recall, F1, one-vs-rest AUC',
         'Per-class recall, since accuracy is set by the majority class'],
        ['Explanation fidelity', 'Deletion AUC, insertion AUC', 'The same curves under a random pixel ordering'],
        ['Perturbation validity', 'Fraction of perturbation steps rejected by the OOD detector',
         'A diagnostic on the two metrics above'],
        ['Explanation localisation', 'Fraction of Grad-CAM mass inside the lung fields',
         'The lung fields\u2019 share of image area (enrichment = ratio)'],
        ['Attribution consistency', 'Mean pairwise cosine similarity of SHAP vectors within a class',
         'The same statistic over all pairs regardless of class'],
        ['Attribution compactness', 'Share of total attributed movement in the top-k features',
         'k / d, the uniform expectation'],
        ['Shortcut sensitivity', 'Macro and per-class F1 under lung masking',
         'The identical recipe on unmasked images'],
        ['Abstention cost', 'False-reject rate per class at a calibrated percentile', 'The nominal percentile'],
    ],
    'Table 1. Evaluation dimensions, metrics and the control each is read against.')

para('**On localisation.** We report the fraction of Grad-CAM mass falling inside the lung fields, '
     '**not** intersection-over-union against annotated pathology. The distinction is material: the '
     'available masks segment the lungs, so a heatmap covering both lungs entirely scores perfectly '
     'while having localised nothing. Because the lung fields occupy roughly a quarter of a chest '
     'radiograph, an uninformative map already achieves a mass fraction near 0.24, and only the '
     'enrichment ratio \u2014 mass fraction divided by area fraction, where 1.0 denotes no better than '
     'uniform \u2014 carries information. Genuine pathology-level IoU requires region annotations this '
     'dataset does not carry, such as the RSNA Pneumonia Detection Challenge bounding boxes or the '
     '984 annotated images in NIH ChestX-ray14; we identify this as the principal extension of the '
     'present protocol.')

heading('4.1 Classification Performance', 2)
para('All figures are on the held-out test split of 3,175 images (COVID19 542, LUNG_OPACITY 902, '
     'NORMAL 1,529, PNEUMONIA 202).')
table(['Backbone', 'Params', 'Macro F1', 'Accuracy', 'Macro AUC', 'Macro P', 'Macro R'],
      [['ResNet18', '11.18M', '0.9587', '0.9524', '0.9928', '0.9640', '0.9536'],
       ['ResNet50', '23.52M', '0.9576', '0.9512', '0.9937', '0.9630', '0.9527']],
      'Table 2. Classification performance, seed-42 runs.')
para('**These are single runs at seed 42**, and are the checkpoints every subsequent section '
     'analyses, so the per-class breakdown and the explanation metrics all refer to these two '
     'specific models. Section 4.2 reports three seeds per backbone and should be consulted before '
     'any two numbers in this table are compared: the run-to-run spread is larger than the '
     'difference between the rows.')
table(['Class', 'ResNet18 P / R / F1 / AUC', 'ResNet50 P / R / F1 / AUC'],
      [['COVID19', '0.989 / 0.976 / 0.982 / 0.9993', '0.985 / 0.980 / 0.982 / 0.9991'],
       ['LUNG_OPACITY', '0.939 / 0.920 / 0.929 / 0.9858', '0.951 / 0.901 / 0.925 / 0.9887'],
       ['NORMAL', '0.944 / 0.963 / 0.953 / 0.9870', '0.936 / 0.969 / 0.952 / 0.9873'],
       ['PNEUMONIA', '0.985 / 0.955 / 0.970 / 0.9992', '0.980 / 0.960 / 0.970 / 0.9995']],
      'Table 3. Per-class precision, recall, F1 and one-vs-rest AUC.')
para('These figures are comparable to published results on this dataset. **They should not be '
     'quoted without Section 4.5**, which establishes that the classes are separable by source '
     'archive, so a score in this range is what a model would produce by learning which repository '
     'an image came from.')
para('Lung opacity is the weakest class under both backbones, and the confusion is almost entirely '
     'with NORMAL: 86 of 902 lung-opacity films are called normal by the ResNet50 model, and 39 '
     'normal films are called lung opacity. This is expected behaviour for two classes differing by '
     'a graded radiographic finding rather than a categorical one.')

heading('4.2 Capacity Ablation and Seed Variance', 2)
para('Each configuration was trained three times under the identical two-stage recipe, varying '
     'only the seed, which governs head initialisation, sampler draws and augmentation order. The '
     'train/validation/test partition is fixed on disk and is therefore not resampled, so what '
     'follows is training-run variance on one split.')
table(['Metric', 'seed 42', 'seed 1', 'seed 2', 'Mean', 'SD', 'Range'],
      [['ResNet18 macro F1', '0.9587', '0.9570', '0.9556', '**0.9571**', '0.0016', '0.0031'],
       ['ResNet50 macro F1', '0.9576', '0.9510', '0.9558', '**0.9548**', '0.0034', '0.0065'],
       ['ResNet18 macro AUC', '0.9928', '0.9926', '0.9919', '0.9924', '\u2014', '\u2014'],
       ['ResNet50 macro AUC', '0.9937', '0.9930', '0.9934', '**0.9933**', '\u2014', '\u2014']],
      'Table 4. Three seeds per backbone, test split.')
para('**The capacity difference is not distinguishable from seed noise.** The means differ by '
     '\u22120.0023 macro F1 against a pooled standard deviation of 0.0025, an effect of 0.93 standard '
     'deviations, and the two ranges overlap substantially ([0.9556, 0.9587] against [0.9510, '
     '0.9576]). With three runs per arm this is nowhere near separation. For scale, the '
     'lung-masking ablation of Section 4.5 moves macro F1 by 0.0246 \u2014 an order of magnitude larger '
     'than either the backbone difference or the noise it sits in.')
para('**This is why the study was necessary rather than tidy.** Comparing the two seed-42 runs '
     'alone gives \u22120.0011, less than half the difference between the means, because that particular '
     'ResNet18 run is the best of its three and that particular ResNet50 run the best of its three. '
     'A single-run comparison here would have reported a number out by a factor of two, and '
     'depending on which pair of runs happened to be trained, could have reported either sign. We '
     'note this because single-run backbone comparisons are common, and on this task the '
     'run-to-run spread exceeds the effect being compared.')
para('**ResNet50 is markedly less stable.** Its standard deviation is 0.0034 against ResNet18\u2019s '
     '0.0016 and its range is twice as wide. The larger model is not merely no better here; it is '
     'more dependent on initialisation, which is the opposite of what additional capacity is '
     'usually expected to buy.')
para('**One difference does survive, and it is in AUC rather than F1.** Every ResNet50 run scores a '
     'higher macro AUC than every ResNet18 run \u2014 the ranges do not overlap at all, 0.9930\u20130.9937 '
     'against 0.9919\u20130.9928 \u2014 while macro F1 shows no such separation. The larger model ranks the '
     'classes more reliably and converts that ranking into decisions no better, and less '
     'consistently. This is the same dissociation that appears in Section 4.5, where masking costs '
     'eight times more macro F1 than macro AUC, and in Section 4.3, where the two fidelity metrics '
     'rank the backbones oppositely: on this data, what separates the classes and what places the '
     'decision boundary come apart repeatedly.')
para('The conventional reading of a flat capacity curve is that the task saturates. A second '
     'reading is available given Section 4.5 and we consider it better supported: **if a '
     'substantial part of the achievable score is obtainable from acquisition signature, the '
     'smaller network has already extracted it and additional capacity has nothing left to buy.** '
     'Under this interpretation the flatness is a property of the dataset rather than the task, and '
     'would not be expected to hold where provenance and label are decorrelated. We report the '
     'comparison chiefly as a caution: a negative capacity ablation is frequently presented as '
     'evidence that a small model suffices; here it is at least equally consistent with the '
     'conclusion that neither model is doing what the class names suggest.')

heading('4.3 Explanation Fidelity', 2)
para('Measured over 200 test images drawn at random from the split (33 COVID19, 48 LUNG_OPACITY, '
     '102 NORMAL, 17 PNEUMONIA). Both backbones are scored on **the identical 200 images under the '
     'identical random control orderings**, the sampling and the controls being drawn from a seeded '
     'generator, so differences below are attributable to the models alone.')
table(['Metric', 'R18 Grad-CAM', 'R18 random', 'R18 gap', 'R50 Grad-CAM', 'R50 random', 'R50 gap'],
      [['Deletion AUC (lower better)', '0.4503', '0.4400', '+0.0103', '0.4282', '0.2738', '**+0.1543**'],
       ['Insertion AUC (higher better)', '0.7764', '0.4364', '+0.3400', '**0.8700**', '0.2740', '**+0.5960**'],
       ['Deletion steps rejected as OOD', '\u2014', '\u2014', '59.7%', '\u2014', '\u2014', '**72.6%**']],
      'Table 5. Deletion and insertion [8], each against a random pixel ordering on the same image.')
para('**The raw AUCs are not comparable between the two models, and this is the first thing the '
     'table shows.** ResNet50\u2019s predicted probability collapses far faster under random '
     'perturbation than ResNet18\u2019s \u2014 its random insertion AUC is 0.2740 against 0.4364 \u2014 so its '
     'curves start from a different place entirely. A paper reporting ResNet50\u2019s insertion AUC of '
     '0.8700 beside ResNet18\u2019s 0.7764 and concluding that the larger model is better explained '
     'would be comparing two quantities measured against different baselines. Only the gap over the '
     'control is interpretable, and by that measure the larger model genuinely is better localised: '
     '+0.5960 against +0.3400.')
para('**Insertion succeeds and deletion fails, on both models.** Restoring the pixels Grad-CAM '
     'ranks highest recovers the predicted probability far faster than restoring random ones. '
     'Removing those same pixels destroys it no faster than removing random ones \u2014 in fact slower, '
     'the deletion gap being positive when it should be negative for both networks. The two metrics '
     'are constructed to agree, and their disagreement requires explanation.')
para('We are able to supply one, and the second backbone turns it from a conjecture into a tested '
     'prediction. Blanking pixels produces an image unlike any radiograph, so a probability that '
     'falls under deletion may report that the input is no longer a chest X-ray rather than that '
     'the evidence has been removed. This objection is well known and is normally left as a caveat. '
     'Here it is measurable, because each model carries a detector for exactly this condition '
     '(Section 4.6): **59.7% of ResNet18\u2019s deletion steps and 72.6% of ResNet50\u2019s are rejected as '
     'out-of-distribution by the respective model\u2019s own Mahalanobis check.**')
para('The prediction this licenses is that the model whose perturbed images are more '
     'off-distribution should have the more corrupted deletion metric, and that is what is '
     'observed: ResNet50 has both the higher rejection rate (72.6% against 59.7%) and the deletion '
     'gap that is further in the wrong direction (+0.1543 against +0.0103), while simultaneously '
     'being the better-localised model on insertion. Deletion and insertion rank the two networks '
     'in opposite orders, and the out-of-distribution rate says which ranking to believe.')
para('We therefore recommend that deletion-style metrics be reported alongside a distributional '
     'validity statistic, and that insertion be preferred where only one can be reported. The '
     'diagnostic costs nothing beyond a forward pass wherever an out-of-distribution detector is '
     'already present, and without it a deletion AUC cannot be distinguished from a measurement of '
     'how brittle the model is to blur.')
table(['Quantity', 'ResNet18', 'ResNet50'],
      [['Grad-CAM mass inside the lung fields', '0.324', '0.304'],
       ['Lung fields as a share of image area', '0.238', '0.238'],
       ['**Enrichment**', '**1.359**', '**1.274**']],
      'Table 6. Localisation within the lung fields.')
para('Both models\u2019 heatmaps concentrate on the lungs, and both do so modestly \u2014 1.36 and 1.27 '
     'times what an uninformative map achieves. Reported alone, a mass fraction of 0.324 would '
     'convey a misleading impression of poor localisation; reported without the area baseline it '
     'would convey nothing at all. As stated above, this is lung-field mass and not pathology IoU. '
     'Note that the larger model, better localised on insertion, is slightly worse by this measure '
     '\u2014 consistent with lung-field containment and evidence localisation being different '
     'properties.')

heading('4.4 Attribution Consistency and Compactness', 2)
table(['Mean pairwise cosine', 'ResNet18 (512-d)', 'ResNet50 (2048-d)'],
      [['Within class', '0.6076', '0.3118'],
       ['All pairs (control)', '0.3142', '0.1479'],
       ['**Ratio**', '**1.93**', '**2.11**']],
      'Table 7. SHAP vector similarity, same 200 images, both backbones.')
para('**The absolute figures halve between the two models and the ratio does not move.** This is '
     'the clearest demonstration in the paper of why the control is load-bearing. Cosine similarity '
     'between high-dimensional vectors falls as dimension rises \u2014 2048-dimensional attributions are '
     'more spread out than 512-dimensional ones simply as geometry \u2014 so a within-class cosine of '
     '0.31 read on its own would suggest ResNet50\u2019s explanations are half as consistent as '
     'ResNet18\u2019s. Measured against its own control, the consistency is if anything marginally '
     'higher. **An absolute attribution-similarity figure is not comparable across architectures '
     'and should not be reported without its null.**')
para('The control is also necessary within a single model. Every SHAP vector in this construction '
     'is w \u2299 (x \u2212 E[x]) for one shared w, so any two vectors agree in direction before anything '
     'about the underlying images is considered; a within-class figure alone is substantially an '
     'artefact of the method. At roughly twice the control on both networks, the within-class '
     'consistency is real.')
table(['Quantity', 'ResNet18', 'ResNet50'],
      [['Top-15 coverage', '15.1%', '14.2%'],
       ['Features', '512', '2048'],
       ['Uniform expectation', '2.93%', '0.73%'],
       ['Concentration over uniform', '5.2\u00d7', '**19.3\u00d7**']],
      'Table 8. Attribution compactness \u2014 share of total attributed movement in the fifteen largest '
      'contributions.')
para('The two models look alike in the first row and are very different underneath it. ResNet50 '
     'spreads its attribution over four times as many features, so reaching a comparable 14.2% in '
     'fifteen of them represents far greater concentration relative to chance. Both readings are '
     'needed: the raw coverage governs how a chart should be captioned, the concentration governs '
     'whether the representation is distributed.')
para('This is the least comfortable result in the paper and it bears directly on how such figures '
     'are presented. **A bar chart of the fifteen largest SHAP values is a sample of the model\u2019s '
     'reasoning, not a summary of it.** A reader shown fifteen bars will reasonably infer they '
     'constitute the explanation; on either network they constitute about a seventh of it. Our '
     '/analyze endpoint returns this coverage fraction alongside the attributions so a client '
     'cannot present them as \u201cthe reason\u201d without contradicting the payload it received, and we '
     'suggest any published per-prediction SHAP chart carry the equivalent figure.')

heading('4.5 The Provenance Confound', 2)
para('**The classes in this dataset are separable by source before any lung is examined.** Every '
     'COVID-19 image in the COVID-19 Radiography Database originates from BIMCV, Eurorad, SIRM or a '
     'GitHub collection; every normal and viral pneumonia image originates from Kaggle-hosted '
     'collections. The overlap is empty. Scanner, exposure, collimation, burned-in annotation and '
     'post-processing character all carry that origin, so a model can achieve a high score by '
     'identifying the repository rather than the disease, and would exhibit no symptom of having '
     'done so on any metric in Section 4.1.')
para('**The masking ablation.** We retrained the identical recipe on a mirror of the same split '
     'with every non-lung pixel zeroed using the shipped segmentation masks \u2014 approximately 77% of '
     'each image removed, including all burned-in markers, collimation edges, soft tissue and '
     'background. The split is mirrored rather than redrawn, so the two runs differ in exactly one '
     'variable.')
table(['Configuration', 'Macro F1', 'Accuracy', 'Macro AUC'],
      [['As distributed', '0.9587', '0.9524', '0.9928'],
       ['Lungs only', '0.9341', '0.9298', '0.9898']],
      'Table 9. Cost of removing everything outside the lungs (ResNet18).')
table(['Per-class loss', 'COVID19', 'LUNG_OPACITY', 'NORMAL', 'PNEUMONIA'],
      [['F1', '**\u22120.056**', '\u22120.030', '\u22120.009', '\u22120.003'],
       ['AUC', '\u22120.0040', '\u22120.0057', '\u22120.0015', '\u22120.0011']],
      'Table 10. Per-class cost of masking.')
para('The score does not collapse, which **rules out** the crude shortcut: the model is not simply '
     'reading annotations or background, because those are gone and it still achieves 0.934. But '
     'the per-class F1 losses are grossly unequal. COVID-19 loses six times what the normal class '
     'loses and nearly twenty times what pneumonia loses, and its recall falls from 0.976 to 0.911. '
     'It is the only class with unique provenance, and it is the class with the most to lose. An '
     'earlier three-class version of this experiment found the same ordering at half the magnitude, '
     'so the effect has reproduced across two class definitions.')
para('**The AUC row qualifies this and should be reported with it.** Macro AUC falls only 0.0030 '
     'where macro F1 falls 0.0246, and on AUC the per-class ordering does not hold \u2014 lung opacity '
     'loses marginally more than COVID-19. Most of what masking costs the COVID-19 class is '
     'therefore the placement of the decision boundary rather than the separability of the class '
     'itself. We state this because the F1 row alone supports a stronger claim than the evidence '
     'warrants.')
para('**Neither explanation detects any of this, and this is the paper\u2019s central negative '
     'result.** Acquisition signature is present inside the lung fields \u2014 in noise characteristics '
     'and processing response \u2014 as is the lung silhouette itself, and a paediatric chest differs in '
     'outline from an adult one, which alone separates the paediatric pneumonia collection from the '
     'adult European COVID-19 series. A model reading provenance rather than pathology therefore '
     'produces heatmaps that fall on the lungs and look entirely reasonable. Our enrichment figure '
     'of 1.359 is consistent with a model reading pathology and equally consistent with one reading '
     'scanner. SHAP fares no better: its baseline is the mean training feature vector, so it '
     'measures deviation from the average image in this dataset, and a large contribution is as '
     'consistent with \u201cunlike the typical scanner here\u201d as with \u201cunlike a healthy lung\u201d.')
para('**Heat on the lungs is necessary, not sufficient.** The verification role routinely assigned '
     'to Grad-CAM in the clinical XAI literature \u2014 that a clinician can confirm the model used '
     'genuine evidence by inspecting the map \u2014 is not supported by these measurements. The confound '
     'that most threatens this dataset is invisible to both explanation methods applied to it.')
para('**Cross-dataset validation, and why it does not settle the question.** Scoring against a '
     'second public dataset appears to be the obvious remedy, and done naively it measures nothing: '
     'a widely used second Kaggle compilation shares **24.0%** of its 6,432 images with this '
     'model\u2019s training split, and **none of those matches by checksum**, every copy having been '
     'resized or re-encoded in compilation. A hash comparison reports two independent datasets.')
table(['Partition', 'Images', 'Macro F1', 'Accuracy'],
      [['All, contaminated', '6,432', '0.9492', '0.9667'],
       ['Overlapping only (control)', '1,545', '0.9864', '0.9903'],
       ['**Clean**', '**4,887**', '**0.9252**', '0.9593']],
      'Table 11. Second-dataset scoring after partitioning by overlap.')
para('Leaving the contamination in was worth a spurious +0.0240 macro F1. But **this is still not '
     'the required experiment.** Both datasets are compiled from the same public archives, so much '
     'of the clean remainder plausibly originates from the same collections as the training data. '
     'What this establishes is that the model does not collapse on unseen images from a differently '
     'assembled compilation. It does not establish that the model reads pathology. Only a COVID-19 '
     'cohort from hospitals where provenance does not predict the label can settle that, and we '
     'identify it as the necessary next experiment.')

heading('4.6 Abstention', 2)
para('A softmax over a fixed class list normalises whatever it is given, so an input unlike '
     'anything in training does not return an uncertain answer; it returns a confident wrong one. '
     'Against the trained ResNet18, a flat grey square is classified COVID-19 at 99.42%, uniform '
     'noise at 100.00%, and a page of text at 99.98% \u2014 none flagged by any confidence threshold, '
     'because there is no uncertainty present to threshold.')
para('We fit a Mahalanobis-distance detector over the penultimate features following Lee et al. '
     '[9]: one Gaussian per class with a tied, Ledoit-Wolf-shrunk covariance, fitted on the '
     'training split and calibrated on validation. **Thresholds are per class**, applied by '
     'whichever class an input is nearest. This is not a detail: a single pooled 95th-percentile '
     'cutoff, measured on the three-class model, reported a reassuring 4.5% false-reject rate while '
     'actually rejecting 30.2% of genuine pneumonia films and 0.4% of normal ones, the pooled '
     'figure having been set by the majority class.')
table(['Class', 'R18 cutoff', 'R18 rejected (test)', 'R18 rejected (2nd set)', 'R50 cutoff', 'R50 rejected (test)'],
      [['COVID19', '939.8', '6.6%', '9.2%', '4663.8', '3.9%'],
       ['LUNG_OPACITY', '822.5', '5.1%', 'not labelled', '4084.0', '5.5%'],
       ['NORMAL', '653.4', '4.1%', '7.3%', '2602.2', '4.6%'],
       ['PNEUMONIA', '1310.4', '6.4%', '6.3%', '6436.5', '**8.9%**'],
       ['Pooled', '\u2014', '5.0%', '6.7%', '\u2014', '5.0%']],
      'Table 12. Calibrated cutoffs and the cost of the check.')
para('All six non-radiograph probes are rejected by a wide margin. Three properties of this table '
     'are worth stating explicitly. **The cutoffs are not comparable between the two models**: '
     'Mahalanobis distance is measured in 512 dimensions for ResNet18 and 2048 for ResNet50, so the '
     'raw thresholds differ by roughly a factor of four for reasons unrelated to detection quality. '
     'Only rejection rates transfer. **The pooled rate is 5.0% for both models by construction**, '
     'being the complement of the 95th-percentile calibration, and is therefore uninformative; the '
     'per-class rows are where the two models differ, and they differ in both directions, with '
     'ResNet50 rejecting fewer COVID-19 films and substantially more pneumonia films. Pneumonia is '
     'the smallest class in the training set, and the larger model\u2019s abstention is hardest on it \u2014 '
     'a cost invisible in the pooled figure. **The threshold does not transfer between datasets**: '
     'a cutoff calibrated at the 95th percentile on one delivers approximately the 93rd on another, '
     'and that second dataset is not even from different hospitals.')
para('**What abstention does not solve.** The detector answers \u201cunlike the training images\u201d, which '
     'is not \u201cnot a chest X-ray\u201d and is much further from \u201cthe model cannot handle this\u201d. Lung '
     'opacity measured that gap: of 600 such films, real radiographs from the same repositories '
     'showing a finding the then-three-class model had no output for, only 34.5% were rejected. The '
     'remainder were accepted and then classified **NORMAL 94.2% of the time at a mean confidence '
     'of 0.972.** That measurement is why lung opacity is a class in the present model. It does not '
     'close the gap: effusion, pneumothorax, nodules and fibrosis remain real radiographs that '
     'resemble the training data, and will be accepted and assigned the nearest available class \u2014 '
     'disproportionately, on this evidence, the one a reader is most likely to act on. **NORMAL '
     'here means \u201cnot the other three\u201d, never \u201cclear\u201d.**')

# ------------------------------------------------------------------ discussion

heading('5. Discussion', 1)
para('This work set out to add a dual explainability layer to a chest radiograph classifier and to '
     'evaluate it. The evaluation produced a less comfortable result than the design anticipated, '
     'and we think the discrepancy is the most useful thing in the paper.')
para('**The explanations work, in the narrow sense in which they can be tested.** Grad-CAM\u2019s '
     'insertion AUC is far above a random ordering on both backbones, its heat is enriched on the '
     'lung fields, and SHAP attributions are roughly twice as consistent within a class as across '
     'classes. On every measurement we could construct, the two methods are doing something real.')
para('**They do not do the job the literature assigns them.** The justification for adding XAI to a '
     'clinical classifier is normally that a clinician can inspect the explanation and detect a '
     'model relying on a spurious correlate. On this dataset the dominant spurious correlate is '
     'source archive, we can demonstrate by ablation that the model uses it, and neither '
     'explanation shows any sign of it. Grad-CAM cannot, because acquisition signature is present '
     'within the lung fields, so a shortcut-driven model produces an anatomically plausible map. '
     'SHAP cannot, because its baseline is drawn from the same confounded distribution. A clinician '
     'following the recommended procedure would inspect a reasonable-looking heatmap and a coherent '
     'attribution profile and conclude, incorrectly, that the prediction was grounded.')
para('This is not an argument against explainability. It is an argument that explanation quality '
     'and dataset validity are separate axes, that a system can score well on the first while '
     'failing on the second, and that XAI work on medical imaging should report a provenance audit '
     'alongside its explanations rather than treating explanation as the audit.')
para('**On methodology.** Three of our findings concern measurement rather than this model, and '
     'having two backbones is what makes them checkable rather than anecdotal. First, the deletion '
     'metric is unreliable where perturbation drives inputs off-distribution, and the condition is '
     'detectable at no extra cost wherever an OOD detector exists; the two networks provide a test '
     'rather than an assertion, since the one with more off-distribution perturbations has the more '
     'corrupted deletion result while being the better-localised model on insertion. Second, '
     '**explanation metrics are not comparable across architectures in raw form.** ResNet50\u2019s raw '
     'insertion AUC of 0.8700 exceeds ResNet18\u2019s 0.7764, but its random control is 0.2740 against '
     '0.4364, so the two numbers are measured from different origins; the same applies to '
     'attribution similarity, where the within-class cosine halves between models purely because '
     '2048-dimensional vectors are more spread out than 512-dimensional ones, while the ratio to '
     'the control barely moves. A comparison of raw figures would report the larger model as both '
     'better explained and less consistent, and both conclusions would be artefacts. Third, '
     'per-prediction SHAP bar charts should carry the share of total attributed movement they '
     'represent.')
para('**Limitations.** The provenance confound cannot be resolved within this dataset, because the '
     'correlation is total by construction; the necessary experiment requires a COVID-19 cohort '
     'from hospitals where source does not predict label. Labels derive from dataset compilers '
     'under varying and largely undocumented criteria, so agreement with them is not agreement with '
     'a diagnosis. Class balance does not reflect prevalence, so no output estimates the '
     'probability that a patient has anything. The localisation metric uses lung masks rather than '
     'pathology annotations. Explanation fidelity was measured on 200 images per backbone at 50 '
     'perturbation steps, on the seed-42 checkpoints only \u2014 the seed study covers classification '
     'metrics, not explanation metrics, so we cannot say how much of the fidelity difference '
     'between backbones is itself run-to-run variance; the two-backbone agreement on the '
     'deletion/OOD relationship is likewise two points rather than a trend, and three seeds per arm '
     'supports the claim that the capacity difference is not distinguishable from noise but is too '
     'few to estimate that noise precisely. And no clinician evaluation was conducted: the '
     'usability study proposed in the original design remains outstanding, and our findings raise a '
     'specific question for it \u2014 whether clinicians shown a plausible heatmap from a '
     'shortcut-driven model correctly withhold trust. Our results predict they would not.')

# ------------------------------------------------------------------ conclusion

heading('6. Conclusion and Future Work', 1)
para('We built a four-class chest radiograph classifier with a dual explainability pipeline \u2014 '
     'Grad-CAM for spatial explanation, exact closed-form SHAP for feature attribution \u2014 served '
     'through a single endpoint, and then measured what the explanations establish. Both backbones '
     'reach a macro F1 near 0.955 and a macro AUC near 0.993. Both explanations pass the fidelity '
     'tests we could construct. Neither detects the shortcut we can independently prove the model '
     'uses, and a 2.1\u00d7 increase in capacity buys nothing distinguishable from seed noise across '
     'three runs per backbone \u2014 while doubling the variance between runs \u2014 which is what one '
     'expects when the shortcut is already exhausted.')
para('We also report two methodological findings of wider applicability: that the deletion metric '
     'is confounded by distribution shift in a way that is measurable using an out-of-distribution '
     'detector, and that explanation metrics require per-model controls before they can be compared '
     'across architectures.')
para('Future work is, in priority order: validation on a chest radiograph cohort in which '
     'acquisition source does not predict the label, which is the only experiment that can settle '
     'the central question; explanation fidelity against genuine pathology annotations, using RSNA '
     'or NIH bounding boxes; and a clinician study designed specifically to test whether a '
     'plausible explanation from a shortcut-driven model induces unwarranted trust.')

# ------------------------------------------------------------------ references

heading('References', 1)
refs = [
    '[1] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). '
    'Grad-CAM: Visual explanations from deep networks via gradient-based localization. Proceedings '
    'of the IEEE International Conference on Computer Vision (ICCV).',
    '[2] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model '
    'predictions. Advances in Neural Information Processing Systems (NeurIPS).',
    '[3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image '
    'recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition '
    '(CVPR).',
    '[4] Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: '
    'Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and '
    'localization of common thorax diseases. Proceedings of the IEEE Conference on Computer Vision '
    'and Pattern Recognition (CVPR).',
    '[5] Irvin, J., Rajpurkar, P., et al. (2019). CheXpert: A large chest radiograph dataset with '
    'uncertainty labels and expert comparison. Proceedings of the AAAI Conference on Artificial '
    'Intelligence.',
    '[6] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). \u201cWhy should I trust you?\u201d: Explaining '
    'the predictions of any classifier. Proceedings of the ACM SIGKDD International Conference on '
    'Knowledge Discovery and Data Mining.',
    '[7] Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & '
    'Wichmann, F. A. (2020). Shortcut learning in deep neural networks. Nature Machine '
    'Intelligence, 2(11), 665\u2013673.',
    '[8] Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized Input Sampling for Explanation '
    'of Black-box Models. British Machine Vision Conference (BMVC).',
    '[9] Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A Simple Unified Framework for Detecting '
    'Out-of-Distribution Samples and Adversarial Attacks. Advances in Neural Information Processing '
    'Systems (NeurIPS).',
    '[10] DeGrave, A. J., Janizek, J. D., & Lee, S.-I. (2021). AI for radiographic COVID-19 '
    'detection selects shortcuts over signal. Nature Machine Intelligence, 3(7), 610\u2013619.',
    '[11] Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity '
    'Checks for Saliency Maps. Advances in Neural Information Processing Systems (NeurIPS).',
]
for r in refs:
    p = para(r, size=10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)

doc.save(OUT)
print('saved', OUT)
print('paragraphs:', len(doc.paragraphs), '| tables:', len(doc.tables))
