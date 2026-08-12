"""Rebuild the corrected paper in IJSR two-column format.

Uses IJSR_PaperFormat.docx as the base so its styles, page setup and column
geometry are inherited rather than reconstructed. Wide tables are transposed or
split, because a 3.42in column cannot hold seven numeric columns.
"""
import copy
import os
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, Inches

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.expanduser(r'~\Downloads\IJSR_PaperFormat.docx')
OUT = os.path.join(HERE, 'Explainable_ChestXray_Paper_IJSR.docx')

doc = Document(TEMPLATE)

# Strip the template's sample content, keeping the trailing sectPr so page size,
# margins and column geometry survive.
body = doc.element.body
for child in list(body):
    if child.tag == qn('w:sectPr'):
        continue
    body.remove(child)

# The remaining section carries the title block, which is single column.
first = doc.sections[0]
cols = first._sectPr.find(qn('w:cols'))
cols.set(qn('w:num'), '1')

# The template numbers its heading styles automatically — I., II. on Heading 1 and
# A., B. on Heading 2 — which printed on top of the numbers written into the heading
# text, giving "IV. 4. Evaluation and Results". The text's own numbers are the ones
# that stay: the prose says "Section 4.5" throughout, and roman numerals cannot carry
# a subsection. So drop the numbering from the styles rather than from the headings.
for style_name in ('Heading 1', 'Heading 2', 'Heading 3'):
    try:
        pPr = doc.styles[style_name].element.find(qn('w:pPr'))
    except KeyError:
        continue
    numPr = pPr.find(qn('w:numPr')) if pPr is not None else None
    if numPr is not None:
        pPr.remove(numPr)

COL_W = Inches(3.42)   # usable width inside one column


def set_grid(t, widths):
    """Write the table grid, not only the cells.

    Under a fixed layout Word measures a table from w:tblGrid. python-docx sets
    that grid when the table is created, dividing the *section* text width — 7.04in
    here — and setting cell widths afterwards does not touch it. Cell widths alone
    therefore read back correct while the table renders at twice the column width,
    spilling across the gutter. Set the grid and w:tblW from the same numbers.
    """
    tw = [int(round(w * 1440)) for w in widths]
    tw[-1] += int(round(COL_W.inches * 1440)) - sum(tw)   # absorb rounding in the last

    tblPr = t._tbl.tblPr
    tblW = tblPr.find(qn('w:tblW'))
    if tblW is None:
        tblW = tblPr.makeelement(qn('w:tblW'), {})
        tblPr.append(tblW)
    tblW.set(qn('w:w'), str(sum(tw)))
    tblW.set(qn('w:type'), 'dxa')

    for col, w in zip(t._tbl.tblGrid.findall(qn('w:gridCol')), tw):
        col.set(qn('w:w'), str(w))
    return [w / 1440 for w in tw]


def set_cell_font(cell, size, bold=False, align=None):
    for p in cell.paragraphs:
        if align is not None:
            p.alignment = align
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.name = 'Times New Roman'
            if bold:
                r.bold = True


def rich(par, text):
    for chunk in re.split(r'(\*\*[^*]+\*\*)', text):
        if not chunk:
            continue
        run = par.add_run(chunk[2:-2] if chunk.startswith('**') else chunk)
        run.bold = chunk.startswith('**')
    return par


def para(text='', style='Text', align=None, size=None, bold=False, italic=False):
    p = doc.add_paragraph(style=style)
    if text:
        rich(p, text)
    if align is not None:
        p.alignment = align
    for r in p.runs:
        if size:
            r.font.size = Pt(size)
        if bold:
            r.bold = True
        if italic:
            r.italic = True
        r.font.name = 'Times New Roman'
    return p


def heading(text, level):
    p = doc.add_paragraph(style='Heading %d' % level)
    r = p.add_run(text)
    r.bold = True
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12 if level == 1 else 10.5)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p


def bullets(items):
    """Numbered, hanging-indent paragraphs.

    The IJSR template defines no list style, and a literal bullet character is
    the wrong way to fake one. Numbering also reads better in a narrow column
    and lets the text refer back to a specific contribution.
    """
    for n, it in enumerate(items, 1):
        p = doc.add_paragraph(style='Text')
        lead = p.add_run('%d) ' % n)
        lead.bold = True
        rich(p, it)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Inches(0.2)
        p.paragraph_format.first_line_indent = Inches(-0.2)
        for r in p.runs:
            r.font.size = Pt(10)
            r.font.name = 'Times New Roman'


def table(caption, headers, rows, widths=None, size=8):
    """IJSR puts the caption ABOVE the table."""
    cap = doc.add_paragraph(style='TableCaption')
    cr = cap.add_run(caption)
    cr.bold = True
    cr.font.size = Pt(9)
    cr.font.name = 'Times New Roman'
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(6)
    cap.paragraph_format.space_after = Pt(2)

    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    t.autofit = False

    if widths is None:
        widths = [COL_W.inches / len(headers)] * len(headers)
    total = sum(widths)
    widths = [w * COL_W.inches / total for w in widths]
    widths = set_grid(t, widths)

    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ''
        c.width = Inches(widths[i])
        rich(c.paragraphs[0], h)
        set_cell_font(c, size, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            cells[i].width = Inches(widths[i])
            rich(cells[i].paragraphs[0], str(v))
            set_cell_font(cells[i], size,
                          align=WD_ALIGN_PARAGRAPH.CENTER if i else WD_ALIGN_PARAGRAPH.LEFT)

    doc.add_paragraph(style='Text').paragraph_format.space_after = Pt(4)
    return t


# ============================================================== title block (1 col)

t = para('Explainable Deep Learning for Multi-Class Chest X-ray Classification: '
         'Measuring What Grad-CAM and SHAP Actually Establish',
         style='Text', align=WD_ALIGN_PARAGRAPH.CENTER, size=20, bold=True)
t.paragraph_format.space_after = Pt(8)

para('Ardhendu', style='Text', align=WD_ALIGN_PARAGRAPH.CENTER, size=11, bold=True)
para('Department of Computer Science Engineering (AI & ML), Jain University, Bengaluru, India',
     style='Normal', align=WD_ALIGN_PARAGRAPH.CENTER, size=9)

# ============================================================== switch to 2 columns

sec = doc.add_section(WD_SECTION.CONTINUOUS)
c = sec._sectPr.find(qn('w:cols'))
c.set(qn('w:num'), '2')
c.set(qn('w:space'), '288')

# ------------------------------------------------------------------- abstract

p = doc.add_paragraph(style='Abstract')
r = p.add_run('Abstract: ')
r.bold = True
r.font.size = Pt(10)
r.font.name = 'Times New Roman'
rich(p, 'Convolutional neural networks achieve high reported accuracy on chest radiograph '
        'classification, and explainable AI (XAI) methods are routinely added to such systems on '
        'the argument that a visible explanation lets a clinician verify the model is using genuine '
        'pathology rather than a spurious correlate. This paper builds that argument\u2019s standard '
        'implementation \u2014 a transfer-learned CNN with Grad-CAM for spatial explanation and SHAP for '
        'feature-level attribution, served through a FastAPI endpoint \u2014 and then measures whether it '
        'does what it is claimed to do. We train ResNet18 and ResNet50 classifiers on the COVID-19 '
        'Radiography Database (21,165 images; four classes) using a two-stage fine-tuning schedule '
        'and a patient-grouped 70/15/15 split, three seeds each. Macro F1 is 0.9571 \u00b1 0.0016 for '
        'ResNet18 and 0.9548 \u00b1 0.0034 for ResNet50 \u2014 a 2.1\u00d7 increase in parameters that is '
        'indistinguishable from seed noise and doubles the run-to-run variance. SHAP values are '
        'computed in closed form over the penultimate features, which is exact for a linear head '
        'and verified against the reference implementation. Three measurements qualify these '
        'results. The dataset\u2019s classes are completely separable by source archive before any lung '
        'is examined, and retraining on lung-masked images \u2014 77% of each image removed \u2014 costs '
        'COVID-19 six times the F1 it costs the normal class. Increasing capacity produces no '
        'improvement distinguishable from noise, as expected when a shortcut is already exhausted. '
        'And neither explanation detects any of this: Grad-CAM produces anatomically plausible maps '
        'regardless, and SHAP measures deviation from a training-set mean carrying the same '
        'confound. We further show the standard deletion metric is unreliable here and quantify '
        'why: 59.7% of its perturbation steps for ResNet18 and 72.6% for ResNet50 are rejected as '
        'out-of-distribution by each model\u2019s own Mahalanobis detector. Insertion AUC, which lacks '
        'this defect, places Grad-CAM well above a random ordering on both networks. We conclude '
        'that visual and feature-level explanations are necessary but demonstrably insufficient for '
        'the verification role assigned to them, and that dataset provenance auditing must '
        'accompany rather than follow explainability work.')
for r_ in p.runs:
    r_.font.size = Pt(10)
    r_.font.name = 'Times New Roman'

p = doc.add_paragraph(style='IndexTerms')
r = p.add_run('Keywords: ')
r.bold = True
rich(p, 'Explainable AI, Chest X-ray Classification, Grad-CAM, SHAP, Shortcut Learning, '
        'Out-of-Distribution Detection')
for r_ in p.runs:
    r_.font.size = Pt(10)
    r_.font.name = 'Times New Roman'

# --------------------------------------------------------------- 1 introduction

heading('1. Introduction', 1)
para('Chest radiography remains one of the most widely used diagnostic tools for detecting '
     'pulmonary conditions such as pneumonia and COVID-19, particularly where access to specialist '
     'radiologists is limited. Convolutional neural networks pretrained on large image corpora and '
     'fine-tuned by transfer learning have demonstrated classification accuracy rivalling human '
     'readers on several benchmark datasets [3], [4], [5].')
para('Despite this, clinical adoption has been slow, largely because these models function as '
     'black boxes: a clinician receives a label and a confidence score with no insight into which '
     'visual evidence drove the decision. This creates a specific risk \u2014 that a model relies on '
     'spurious artifacts such as text markers, positioning hardware or scanner-specific noise '
     'rather than genuine pathology [7], [10]. Explainable AI addresses this by producing '
     'human-interpretable justifications alongside predictions, on the argument that a clinician '
     'can then verify, trust or challenge a diagnosis.')
para('This paper takes that argument seriously enough to test it. We build the standard '
     'implementation and then measure, rather than assert, what its explanations establish. The '
     'specific contributions are as follows.')
bullets([
    'A four-class chest radiograph classifier trained by two-stage transfer learning, reported at '
    'two backbone capacities and three seeds each, so that the effect of model size is separated '
    'from the effect of the data and from run-to-run noise.',
    'A dual explainability pipeline combining Grad-CAM with SHAP in which the SHAP values are '
    '**exact rather than approximated**, verified against the reference implementation.',
    '**A quantitative audit of both explanations**, each metric reported against an explicit null '
    'control rather than in isolation.',
    '**The finding that the deletion metric is confounded by distribution shift**, with a method '
    'for detecting that condition using the classifier\u2019s own out-of-distribution detector.',
    '**A dataset provenance audit**, and the demonstration that neither explanation surfaces it.',
    'An abstention mechanism with per-class calibration, and the measurement that its threshold '
    'does not transfer between datasets.',
])

# --------------------------------------------------------------- 2 related work

heading('2. Related Work', 1)
heading('2.1 Deep Learning for Chest X-ray Classification', 2)
para('Transfer learning from ImageNet-pretrained CNN architectures is the standard approach for '
     'chest X-ray classification given the limited size of labelled medical imaging datasets '
     'relative to natural image corpora [3]. Prior work has applied these architectures to '
     'pneumonia and COVID-19, generally reporting accuracies in the 83\u201390 percent range depending '
     'on the disease set and evaluation methodology [4], [5].')
heading('2.2 Explainability Techniques in Medical Imaging', 2)
para('Grad-CAM [1] is among the most widely used visual explanation techniques for CNN-based '
     'medical image classifiers. Complementary approaches such as LIME [6] and SHAP [2] provide '
     'feature-level attributions. Several studies combine multiple XAI techniques and evaluate the '
     'resulting explanations against radiologist-identified regions of interest. Adebayo et al. '
     '[11] show that several saliency methods pass visual inspection while failing basic sanity '
     'checks, which motivates the control-based reporting we adopt throughout Section 4.')
heading('2.3 Research Gap', 2)
para('Two gaps motivate this work. First, comparatively few studies integrate a dual spatial and '
     'feature-attribution pipeline within an end-to-end serving workflow. Second, and more '
     'importantly, explanation quality is usually asserted from visual plausibility rather than '
     'measured against controls. DeGrave et al. [10] demonstrate shortcut learning in COVID-19 '
     'radiograph classifiers; our contribution relative to that work is the ablation-plus-capacity '
     'evidence, and specifically the finding that the dual XAI pipeline routinely proposed as the '
     'remedy does not surface the problem.')

# --------------------------------------------------------------- 3 methodology

heading('3. Proposed Methodology', 1)
heading('3.1 System Overview', 2)
para('The system comprises four stages: data acquisition and preprocessing; classification using a '
     'fine-tuned residual network; a dual explainability pipeline combining Grad-CAM and SHAP; and '
     'exposure of predictions and both explanations through a FastAPI service. A fifth component, '
     'an out-of-distribution detector, allows the system to abstain and is also used in Section 4.3 '
     'as a diagnostic on the fidelity metrics themselves.')

heading('3.2 Dataset and Preprocessing', 2)
para('**Dataset.** All experiments use the COVID-19 Radiography Database, a publicly available '
     'compilation of 21,165 de-identified chest radiographs. We use four classes: NORMAL (10,192), '
     'LUNG_OPACITY (6,012), COVID19 (3,616) and PNEUMONIA (1,345). Lung opacity is retained as its '
     'own class and never merged into pneumonia, since it denotes a broader radiographic finding. '
     'The database also ships lung segmentation masks for all four classes, used in Sections 4.3 '
     'and 4.5.')
para('**Splitting.** Images are partitioned 70/15/15 into training (14,814), validation (3,176) '
     'and test (3,175) sets. The split is **grouped by patient**: several constituent collections '
     'contain multiple films of the same chest, and splitting those at random places the same '
     'patient in both training and test, inflating the reported score.')
para('**Preprocessing.** Images are converted to greyscale and replicated to three channels, '
     'resized to 224\u00d7224, and normalised with ImageNet channel statistics. Augmentation on the '
     'training split comprises random resized cropping (scale 0.85\u20131.0), rotation up to 10 degrees, '
     'and brightness and contrast jitter of 0.15. **Horizontal flipping is deliberately excluded**: '
     'a mirrored chest places the heart on the right, an anatomy occurring in roughly 1 in 10,000 '
     'people, and training a model to treat it as unremarkable is not a defensible trade for a '
     'marginal increase in effective dataset size.')
para('**Class imbalance** is addressed by a weighted random sampler drawing each class '
     'approximately equally often. Inverse-frequency loss weighting is available as an alternative; '
     'the two are not combined, since applying both over-corrects toward the rare classes.')

heading('3.3 Model Architecture', 2)
para('**Backbones.** We report ResNet18 and ResNet50, each with its 1000-way ImageNet head replaced '
     'by a fresh four-way linear layer, giving 11.18M and 23.52M parameters \u2014 a factor of 2.1.')
para('**Two-stage fine-tuning.** The backbone is first frozen and only the head trained (learning '
     'rate 3\u00d710\u207b\u2074), then the whole network unfrozen and training continued from those weights at a '
     'reduced rate (1\u00d710\u207b\u2074). The second stage restores weights only: optimizer state is not carried '
     'across, since AdamW moments accumulated while the backbone was frozen do not describe the '
     'parameters the second stage unfreezes.')
para('**Optimisation.** AdamW, weight decay 10\u207b\u2074, cosine-annealed learning rate, batch size 32, '
     'early stopping with patience 5. Model selection is on **validation macro F1, not accuracy**: '
     'with COVID-19 at roughly a tenth of the normal-class count, a model that never predicts the '
     'rarest class can still post a high accuracy, and selection on accuracy would faithfully '
     'preserve that model.')

heading('3.4 Explainability Pipeline', 2)
para('**Grad-CAM** is applied at the final convolutional block (7\u00d77 for a 224-pixel input), '
     'weighting each feature map by the mean gradient of the target class score with respect to it, '
     'rectifying, and bilinearly upsampling to the input resolution. Explanations can be requested '
     'for a class other than the predicted one.')
para('**SHAP over learned features.** SHAP is applied to the penultimate representation \u2014 512 '
     'dimensions for ResNet18, 2048 for ResNet50 \u2014 rather than to pixels. Pixel-level SHAP would '
     'produce another spatial map, giving the system two answers to \u201cwhere\u201d and none to \u201chow '
     'much\u201d; attributing over the features the classifier actually reads gives a genuinely '
     'different view.')
para('This choice also makes the attribution exact. The head is a single linear layer, so the class '
     'score is w\u00b7x + b, and for a linear model the Shapley values have a closed form, '
     '\u03c6\u1d62 = w\u1d62(x\u1d62 \u2212 E[x\u1d62]) with base value w\u00b7E[x] + b, where E[x] is the mean feature vector over the '
     'training split. No sampling and no convergence criterion are involved. The decomposition '
     'satisfies the Shapley efficiency axiom exactly, and our implementation is verified to agree '
     'with the reference LinearExplainer of Lundberg and Lee [2] to floating-point precision.')
para('**Four limitations**, which bear directly on Section 4.4. (i) A feature index is not a '
     'clinical concept: \u201cfeature 453 contributed +0.44 logits\u201d conveys nothing to a radiologist. '
     '(ii) The interventional formulation treats features as independent, which convolutional '
     'channels are not. (iii) Logits are explained, not probabilities, since softmax is not '
     'additive. (iv) **The baseline is the mean training image, so the attribution inherits '
     'whatever the training set has in common \u2014 including its provenance.**')

heading('3.5 System Integration', 2)
para('Predictions and both explanations are served from a FastAPI application with four endpoints: '
     '/health, /predict, /explain, and /analyze, the last returning the prediction, the Grad-CAM '
     'overlay base64-encoded, and the SHAP summary in a single JSON response. /analyze exists '
     'because the two explanations are intended to be read together, and delivering them over '
     'separate requests permits an interface to render the image and silently discard the numbers. '
     'Its SHAP block carries the base value, the sum of contributions and the resulting logit, so a '
     'client can verify the decomposition it is shown, together with the fraction of total '
     'attributed movement the listed features represent.')
para('Every response carries a non-dismissible disclaimer, and the interface deliberately does '
     '**not** colour-code the disease classes: a green bar reading \u201cNORMAL 94%\u201d constitutes an '
     'all-clear the system is not entitled to give.')

# ------------------------------------------------------------ 4 evaluation

heading('4. Evaluation and Results', 1)
para('Each metric is reported against an explicit null, since none is interpretable alone.')

table('Table 1: Metrics and the control each is read against',
      ['Metric', 'Control'],
      [['Accuracy, precision, recall, F1, AUC', 'Per-class recall, since accuracy is set by the majority class'],
       ['Deletion / insertion AUC', 'The same curves under a random pixel ordering'],
       ['Steps rejected by OOD detector', 'A diagnostic on the two metrics above'],
       ['Grad-CAM mass in lung fields', 'The lung fields\u2019 share of image area'],
       ['Within-class SHAP cosine', 'The same statistic over all pairs regardless of class'],
       ['Top-k attribution coverage', 'k / d, the uniform expectation'],
       ['F1 under lung masking', 'The identical recipe on unmasked images'],
       ['Per-class false-reject rate', 'The nominal percentile']],
      widths=[1.35, 2.05], size=8)

para('**On localisation.** We report the fraction of Grad-CAM mass inside the lung fields, **not** '
     'intersection-over-union against annotated pathology. The available masks segment the lungs, '
     'so a heatmap covering both lungs entirely scores perfectly while having localised nothing. '
     'Because the lung fields occupy roughly a quarter of a radiograph, an uninformative map '
     'already achieves a mass fraction near 0.24, and only the enrichment ratio \u2014 mass fraction '
     'divided by area fraction, where 1.0 denotes no better than uniform \u2014 carries information. '
     'Genuine pathology IoU requires annotations this dataset does not carry.')

heading('4.1 Classification Performance', 2)
para('All figures are on the held-out test split of 3,175 images (COVID19 542, LUNG_OPACITY 902, '
     'NORMAL 1,529, PNEUMONIA 202).')
table('Table 2: Classification performance, seed-42 runs',
      ['Quantity', 'ResNet18', 'ResNet50'],
      [['Parameters', '11.18M', '23.52M'],
       ['Macro F1', '0.9587', '0.9576'],
       ['Accuracy', '0.9524', '0.9512'],
       ['Macro AUC', '0.9928', '0.9937'],
       ['Macro precision', '0.9640', '0.9630'],
       ['Macro recall', '0.9536', '0.9527']],
      widths=[1.4, 1.0, 1.0])
para('**These are single runs at seed 42**, and are the checkpoints every subsequent section '
     'analyses. Section 4.2 reports three seeds per backbone and should be consulted before any two '
     'numbers here are compared: the run-to-run spread is larger than the difference between the '
     'columns.')
table('Table 3: Per-class precision / recall / F1 / AUC',
      ['Class', 'ResNet18', 'ResNet50'],
      [['COVID19', '.989 / .976 / .982 / .9993', '.985 / .980 / .982 / .9991'],
       ['LUNG_OPACITY', '.939 / .920 / .929 / .9858', '.951 / .901 / .925 / .9887'],
       ['NORMAL', '.944 / .963 / .953 / .9870', '.936 / .969 / .952 / .9873'],
       ['PNEUMONIA', '.985 / .955 / .970 / .9992', '.980 / .960 / .970 / .9995']],
      # 0.86in split LUNG_OPACITY across two lines; the class names are one
      # unbreakable token each, so this column is sized by its longest label.
      widths=[1.0, 1.21, 1.21], size=7.5)
para('These figures are comparable to published results on this dataset. **They should not be '
     'quoted without Section 4.5**, which establishes that the classes are separable by source '
     'archive. Lung opacity is the weakest class under both backbones, and the confusion is almost '
     'entirely with NORMAL.')

heading('4.2 Capacity Ablation and Seed Variance', 2)
para('Each configuration was trained three times under the identical recipe, varying only the seed, '
     'which governs head initialisation, sampler draws and augmentation order. The split is fixed '
     'on disk and is not resampled, so this is training-run variance on one partition.')
table('Table 4: Three seeds per backbone, macro F1 on test',
      ['Statistic', 'ResNet18', 'ResNet50'],
      [['seed 42', '0.9587', '0.9576'],
       ['seed 1', '0.9570', '0.9510'],
       ['seed 2', '0.9556', '0.9558'],
       ['**Mean**', '**0.9571**', '**0.9548**'],
       ['Std. deviation', '0.0016', '0.0034'],
       ['Range', '0.0031', '0.0065']],
      widths=[1.4, 1.0, 1.0])
table('Table 5: Macro AUC across the same runs',
      ['Statistic', 'ResNet18', 'ResNet50'],
      [['seed 42', '0.9928', '0.9937'],
       ['seed 1', '0.9926', '0.9930'],
       ['seed 2', '0.9919', '0.9934'],
       ['**Mean**', '**0.9924**', '**0.9933**']],
      widths=[1.4, 1.0, 1.0])
para('**The capacity difference is not distinguishable from seed noise.** The means differ by '
     '\u22120.0023 macro F1 against a pooled standard deviation of 0.0025 \u2014 an effect of 0.93 standard '
     'deviations \u2014 and the ranges overlap substantially. For scale, the lung-masking ablation of '
     'Section 4.5 moves macro F1 by 0.0246, an order of magnitude further.')
para('**This is why the study was necessary.** Comparing the two seed-42 runs alone gives \u22120.0011, '
     'less than half the difference between the means, because that ResNet18 run is the best of its '
     'three and that ResNet50 run the best of its three. A single-run comparison would have '
     'reported a number out by a factor of two and, depending which pair was trained, either sign. '
     'Single-run backbone comparisons are common, and on this task the spread exceeds the effect.')
para('**ResNet50 is markedly less stable**, with a standard deviation of 0.0034 against 0.0016 and '
     'twice the range. The larger model is not merely no better; it is more dependent on '
     'initialisation, the opposite of what capacity is usually expected to buy.')
para('**One difference does survive, in AUC rather than F1.** Every ResNet50 run scores a higher '
     'macro AUC than every ResNet18 run \u2014 0.9930\u20130.9937 against 0.9919\u20130.9928, no overlap \u2014 while '
     'macro F1 shows no separation. The larger model ranks the classes more reliably and converts '
     'that ranking into decisions no better and less consistently. The same dissociation appears in '
     'Section 4.5, where masking costs eight times more F1 than AUC, and in Section 4.3, where the '
     'two fidelity metrics rank the backbones oppositely.')
para('The conventional reading of a flat capacity curve is task saturation. Given Section 4.5 a '
     'second reading is better supported: **if a substantial part of the achievable score comes '
     'from acquisition signature, the smaller network has already extracted it and additional '
     'capacity has nothing left to buy.**')

heading('4.3 Explanation Fidelity', 2)
para('Measured over 200 test images drawn at random (33 COVID19, 48 LUNG_OPACITY, 102 NORMAL, 17 '
     'PNEUMONIA). Both backbones are scored on **the identical images under identical random '
     'control orderings**, so differences are attributable to the models alone.')
table('Table 6: Deletion and insertion [8], against a random ordering',
      ['Quantity', 'Deletion', 'Insertion'],
      [['ResNet18 Grad-CAM', '0.4503', '0.7764'],
       ['ResNet18 random', '0.4400', '0.4364'],
       ['**ResNet18 gap**', '+0.0103', '**+0.3400**'],
       ['ResNet50 Grad-CAM', '0.4282', '0.8700'],
       ['ResNet50 random', '0.2738', '0.2740'],
       ['**ResNet50 gap**', '**+0.1543**', '**+0.5960**']],
      widths=[1.6, 0.9, 0.9])
para('Lower is better for deletion, higher for insertion. **The raw AUCs are not comparable between '
     'the two models**, which is the first thing the table shows. ResNet50\u2019s probability collapses '
     'far faster under random perturbation \u2014 its random insertion AUC is 0.2740 against 0.4364 \u2014 so '
     'its curves start from a different place. Reporting 0.8700 beside 0.7764 and concluding the '
     'larger model is better explained compares two quantities measured against different '
     'baselines. Only the gap is interpretable, and by that measure ResNet50 genuinely is better '
     'localised.')
para('**Insertion succeeds and deletion fails, on both models.** Removing the highest-ranked pixels '
     'destroys the prediction no faster than removing random ones \u2014 slower, the deletion gap being '
     'positive when it should be negative. The two metrics are constructed to agree, and their '
     'disagreement requires explanation.')
para('We supply one, and the second backbone turns it from conjecture into a tested prediction. '
     'Blanking pixels produces an image unlike any radiograph, so a probability that falls under '
     'deletion may report that the input is no longer a chest X-ray rather than that the evidence '
     'has been removed. Here that is measurable: **59.7% of ResNet18\u2019s deletion steps and 72.6% of '
     'ResNet50\u2019s are rejected as out-of-distribution by the respective model\u2019s own Mahalanobis '
     'check.** The model with more off-distribution perturbations has the more corrupted deletion '
     'result while being the better-localised model on insertion. Deletion and insertion rank the '
     'networks oppositely, and the rejection rate says which ranking to believe.')
para('We therefore recommend that deletion-style metrics be reported alongside a distributional '
     'validity statistic, and that insertion be preferred where only one can be reported. The '
     'diagnostic costs nothing beyond a forward pass wherever an OOD detector is present.')
table('Table 7: Localisation within the lung fields',
      ['Quantity', 'ResNet18', 'ResNet50'],
      [['Mass inside lung fields', '0.324', '0.304'],
       ['Lung share of image area', '0.238', '0.238'],
       ['**Enrichment**', '**1.359**', '**1.274**']],
      widths=[1.6, 0.9, 0.9])
para('Both models\u2019 heatmaps concentrate on the lungs modestly \u2014 1.36 and 1.27 times what an '
     'uninformative map achieves. Note that the larger model, better localised on insertion, is '
     'slightly worse by this measure: lung-field containment and evidence localisation are '
     'different properties.')

heading('4.4 Attribution Consistency and Compactness', 2)
table('Table 8: SHAP vector similarity, same 200 images',
      ['Mean pairwise cosine', 'R18 (512-d)', 'R50 (2048-d)'],
      [['Within class', '0.6076', '0.3118'],
       ['All pairs (control)', '0.3142', '0.1479'],
       ['**Ratio**', '**1.93**', '**2.11**']],
      widths=[1.5, 0.95, 0.95])
para('**The absolute figures halve between the two models and the ratio does not move.** Cosine '
     'similarity between high-dimensional vectors falls as dimension rises, so a within-class '
     'cosine of 0.31 read alone would suggest ResNet50\u2019s explanations are half as consistent. '
     'Measured against its own control the consistency is if anything marginally higher. **An '
     'absolute attribution-similarity figure is not comparable across architectures and should not '
     'be reported without its null.**')
para('The control is also necessary within a single model. Every SHAP vector here is w \u2299 (x \u2212 E[x]) '
     'for one shared w, so any two vectors agree in direction before anything about the images is '
     'considered. At roughly twice the control on both networks, the within-class consistency is '
     'real.')
table('Table 9: Attribution compactness, fifteen largest contributions',
      ['Quantity', 'ResNet18', 'ResNet50'],
      [['Top-15 coverage', '15.1%', '14.2%'],
       ['Features', '512', '2048'],
       ['Uniform expectation', '2.93%', '0.73%'],
       ['**Concentration**', '5.2\u00d7', '**19.3\u00d7**']],
      widths=[1.5, 0.95, 0.95])
para('The two models look alike in the first row and differ underneath it: ResNet50 spreads its '
     'attribution over four times as many features, so a comparable 14.2% in fifteen of them '
     'represents far greater concentration relative to chance.')
para('This is the least comfortable result in the paper. **A bar chart of the fifteen largest SHAP '
     'values is a sample of the model\u2019s reasoning, not a summary of it.** A reader shown fifteen '
     'bars will reasonably infer they constitute the explanation; on either network they constitute '
     'about a seventh of it. Our /analyze endpoint returns this coverage fraction alongside the '
     'attributions so a client cannot present them as \u201cthe reason\u201d without contradicting the '
     'payload it received, and we suggest any published per-prediction SHAP chart carry the '
     'equivalent figure.')

heading('4.5 The Provenance Confound', 2)
para('**The classes in this dataset are separable by source before any lung is examined.** Every '
     'COVID-19 image originates from BIMCV, Eurorad, SIRM or a GitHub collection; every normal and '
     'viral pneumonia image originates from Kaggle-hosted collections. The overlap is empty. '
     'Scanner, exposure, collimation, burned-in annotation and post-processing character all carry '
     'that origin, so a model can achieve a high score by identifying the repository rather than '
     'the disease, and would exhibit no symptom on any metric in Section 4.1.')
para('**The masking ablation.** We retrained the identical recipe on a mirror of the same split '
     'with every non-lung pixel zeroed \u2014 approximately 77% of each image removed. The split is '
     'mirrored rather than redrawn, so the two runs differ in exactly one variable.')
table('Table 10: Cost of removing everything outside the lungs (ResNet18)',
      ['Configuration', 'Macro F1', 'Accuracy', 'AUC'],
      [['As distributed', '0.9587', '0.9524', '0.9928'],
       ['Lungs only', '0.9341', '0.9298', '0.9898']],
      widths=[1.2, 0.78, 0.78, 0.66])
table('Table 11: Per-class cost of masking',
      ['Class', 'F1 lost', 'AUC lost'],
      [['COVID19', '**\u22120.056**', '\u22120.0040'],
       ['LUNG_OPACITY', '\u22120.030', '\u22120.0057'],
       ['NORMAL', '\u22120.009', '\u22120.0015'],
       ['PNEUMONIA', '\u22120.003', '\u22120.0011']],
      widths=[1.5, 0.95, 0.95])
para('The score does not collapse, which **rules out** the crude shortcut: the model is not simply '
     'reading annotations or background, because those are gone and it still achieves 0.934. But '
     'the per-class F1 losses are grossly unequal. COVID-19 loses six times what the normal class '
     'loses and nearly twenty times what pneumonia loses, and its recall falls from 0.976 to 0.911. '
     'It is the only class with unique provenance, and it is the class with the most to lose. An '
     'earlier three-class version of this experiment found the same ordering at half the magnitude, '
     'so the effect has reproduced across two class definitions.')
para('**The AUC column qualifies this and should be reported with it.** Macro AUC falls only 0.0030 '
     'where macro F1 falls 0.0246, and on AUC the per-class ordering does not hold \u2014 lung opacity '
     'loses marginally more than COVID-19. Most of what masking costs the COVID-19 class is '
     'therefore the placement of the decision boundary rather than the separability of the class '
     'itself. We state this because the F1 column alone supports a stronger claim than the evidence '
     'warrants.')
para('**Neither explanation detects any of this, and this is the paper\u2019s central negative result.** '
     'Acquisition signature is present inside the lung fields, as is the lung silhouette itself, '
     'and a paediatric chest differs in outline from an adult one \u2014 which alone separates the '
     'paediatric pneumonia collection from the adult European COVID-19 series. A model reading '
     'provenance rather than pathology therefore produces heatmaps that fall on the lungs and look '
     'entirely reasonable. Our enrichment figure of 1.359 is consistent with a model reading '
     'pathology and equally consistent with one reading scanner. SHAP fares no better: its baseline '
     'is the mean training feature vector, so a large contribution is as consistent with \u201cunlike '
     'the typical scanner here\u201d as with \u201cunlike a healthy lung\u201d.')
para('**Heat on the lungs is necessary, not sufficient.** The verification role routinely assigned '
     'to Grad-CAM in the clinical XAI literature is not supported by these measurements: the '
     'confound that most threatens this dataset is invisible to both explanation methods applied '
     'to it.')
para('**Cross-dataset validation does not settle it either.** A widely used second Kaggle '
     'compilation shares **24.0%** of its 6,432 images with this model\u2019s training split, and **none '
     'of those matches by checksum**, every copy having been resized or re-encoded. A hash '
     'comparison reports two independent datasets.')
table('Table 12: Second-dataset scoring after partitioning by overlap',
      ['Partition', 'Images', 'Macro F1', 'Acc.'],
      [['All, contaminated', '6,432', '0.9492', '0.9667'],
       ['Overlap only (control)', '1,545', '0.9864', '0.9903'],
       ['**Clean**', '**4,887**', '**0.9252**', '0.9593']],
      widths=[1.35, 0.65, 0.75, 0.67])
para('Leaving the contamination in was worth a spurious +0.0240 macro F1. But both datasets are '
     'compiled from the same public archives, so much of the clean remainder plausibly originates '
     'from the same collections as the training data. This establishes that the model does not '
     'collapse on unseen images from a differently assembled compilation. It does not establish '
     'that the model reads pathology. Only a cohort from hospitals where provenance does not '
     'predict the label can settle that.')

heading('4.6 Abstention', 2)
para('A softmax over a fixed class list normalises whatever it is given, so an input unlike '
     'anything in training does not return an uncertain answer; it returns a confident wrong one. '
     'Against the trained ResNet18, a flat grey square is classified COVID-19 at 99.42%, uniform '
     'noise at 100.00%, and a page of text at 99.98% \u2014 none flagged by any confidence threshold, '
     'because there is no uncertainty present to threshold.')
para('We fit a Mahalanobis-distance detector over the penultimate features following Lee et al. '
     '[9]: one Gaussian per class with a tied, Ledoit-Wolf-shrunk covariance, fitted on training '
     'and calibrated on validation. **Thresholds are per class.** This is not a detail: a single '
     'pooled 95th-percentile cutoff, measured on the three-class model, reported a reassuring 4.5% '
     'false-reject rate while actually rejecting 30.2% of genuine pneumonia films and 0.4% of '
     'normal ones, the pooled figure having been set by the majority class.')
table('Table 13: Calibrated cutoffs and false-reject rates',
      ['Class', 'R18 cut', 'R18 rej.', 'R50 cut', 'R50 rej.'],
      [['COVID19', '939.8', '6.6%', '4663.8', '3.9%'],
       ['LUNG_OPAC.', '822.5', '5.1%', '4084.0', '5.5%'],
       ['NORMAL', '653.4', '4.1%', '2602.2', '4.6%'],
       ['PNEUMONIA', '1310.4', '6.4%', '6436.5', '**8.9%**'],
       ['Pooled', '\u2014', '5.0%', '\u2014', '5.0%']],
      widths=[1.0, 0.62, 0.6, 0.62, 0.58], size=7.5)
para('All six non-radiograph probes are rejected by a wide margin. **The cutoffs are not comparable '
     'between models**: Mahalanobis distance is measured in 512 dimensions for ResNet18 and 2048 '
     'for ResNet50, so the raw thresholds differ by roughly a factor of four for reasons unrelated '
     'to detection quality; only rejection rates transfer. **The pooled rate is 5.0% for both by '
     'construction**, being the complement of the calibration percentile, and is therefore '
     'uninformative. The per-class rows differ in both directions, with ResNet50 rejecting fewer '
     'COVID-19 films and substantially more pneumonia films \u2014 the smallest class, on which the '
     'larger model\u2019s abstention falls hardest, a cost invisible in the pooled figure.')
para('**The threshold does not transfer between datasets.** On the clean images of the second '
     'dataset the ResNet18 rejection rate rises to 9.2% for COVID-19, 7.3% for NORMAL and 6.7% '
     'pooled: a cutoff calibrated at the 95th percentile on one dataset delivers approximately the '
     '93rd on another, and that dataset is not even from different hospitals.')
para('**What abstention does not solve.** The detector answers \u201cunlike the training images\u201d, which '
     'is not \u201cnot a chest X-ray\u201d and is much further from \u201cthe model cannot handle this\u201d. Lung '
     'opacity measured that gap: of 600 such films, real radiographs showing a finding the '
     'then-three-class model had no output for, only 34.5% were rejected. The remainder were '
     'accepted and classified **NORMAL 94.2% of the time at a mean confidence of 0.972.** That '
     'measurement is why lung opacity is a class in the present model. It does not close the gap: '
     'effusion, pneumothorax, nodules and fibrosis remain real radiographs that resemble the '
     'training data and will be assigned the nearest available class \u2014 disproportionately, on this '
     'evidence, the one a reader is most likely to act on. **NORMAL here means \u201cnot the other '
     'three\u201d, never \u201cclear\u201d.**')

# ------------------------------------------------------------------ discussion

heading('5. Discussion', 1)
para('**The explanations work, in the narrow sense in which they can be tested.** Grad-CAM\u2019s '
     'insertion AUC is far above a random ordering on both backbones, its heat is enriched on the '
     'lung fields, and SHAP attributions are roughly twice as consistent within a class as across '
     'classes. On every measurement we could construct, the two methods are doing something real.')
para('**They do not do the job the literature assigns them.** The justification for adding XAI to a '
     'clinical classifier is normally that a clinician can inspect the explanation and detect a '
     'model relying on a spurious correlate. On this dataset the dominant spurious correlate is '
     'source archive, we can demonstrate by ablation that the model uses it, and neither '
     'explanation shows any sign of it. Grad-CAM cannot, because acquisition signature is present '
     'within the lung fields. SHAP cannot, because its baseline is drawn from the same confounded '
     'distribution. A clinician following the recommended procedure would inspect a '
     'reasonable-looking heatmap and a coherent attribution profile and conclude, incorrectly, that '
     'the prediction was grounded.')
para('This is not an argument against explainability. It is an argument that explanation quality '
     'and dataset validity are separate axes, that a system can score well on the first while '
     'failing on the second, and that XAI work on medical imaging should report a provenance audit '
     'alongside its explanations rather than treating explanation as the audit.')
para('**On methodology.** Three findings concern measurement rather than this model. First, the '
     'deletion metric is unreliable where perturbation drives inputs off-distribution, and the '
     'condition is detectable wherever an OOD detector exists. Second, **explanation metrics are '
     'not comparable across architectures in raw form**: ResNet50\u2019s raw insertion AUC exceeds '
     'ResNet18\u2019s, but its random control is 0.2740 against 0.4364, so the two are measured from '
     'different origins; the same applies to attribution similarity, where the within-class cosine '
     'halves between models purely as geometry while the ratio to the control barely moves. A '
     'comparison of raw figures would report the larger model as both better explained and less '
     'consistent, and both conclusions would be artefacts. Third, per-prediction SHAP charts should '
     'carry the share of attributed movement they represent.')
para('**Limitations.** The provenance confound cannot be resolved within this dataset, because the '
     'correlation is total by construction. Labels derive from dataset compilers under varying and '
     'largely undocumented criteria, so agreement with them is not agreement with a diagnosis. '
     'Class balance does not reflect prevalence. The localisation metric uses lung masks rather '
     'than pathology annotations. Explanation fidelity was measured on 200 images per backbone at '
     '50 perturbation steps, on the seed-42 checkpoints only \u2014 the seed study covers classification '
     'metrics, not explanation metrics, so we cannot say how much of the fidelity difference '
     'between backbones is itself run-to-run variance; the two-backbone agreement on the '
     'deletion/OOD relationship is likewise two points rather than a trend, and three seeds per arm '
     'is too few to estimate the noise precisely. No clinician evaluation was conducted, and our '
     'findings raise a specific question for one \u2014 whether clinicians shown a plausible heatmap '
     'from a shortcut-driven model correctly withhold trust. Our results predict they would not.')

heading('6. Conclusion and Future Work', 1)
para('We built a four-class chest radiograph classifier with a dual explainability pipeline and '
     'then measured what the explanations establish. Both backbones reach a macro F1 near 0.955 and '
     'a macro AUC near 0.993. Both explanations pass the fidelity tests we could construct. Neither '
     'detects the shortcut we can independently prove the model uses, and a 2.1\u00d7 increase in '
     'capacity buys nothing distinguishable from seed noise while doubling the variance between '
     'runs \u2014 which is what one expects when the shortcut is already exhausted.')
para('We also report two methodological findings of wider applicability: that the deletion metric '
     'is confounded by distribution shift in a way measurable using an out-of-distribution '
     'detector, and that explanation metrics require per-model controls before they can be compared '
     'across architectures.')
para('Future work is, in priority order: validation on a cohort in which acquisition source does '
     'not predict the label, the only experiment that can settle the central question; explanation '
     'fidelity against genuine pathology annotations using RSNA or NIH bounding boxes; and a '
     'clinician study designed to test whether a plausible explanation from a shortcut-driven model '
     'induces unwarranted trust.')

# ------------------------------------------------------------------ references

heading('References', 1)
refs = [
    'R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, \u201cGrad-CAM: Visual '
    'Explanations from Deep Networks via Gradient-Based Localization,\u201d Proc. IEEE ICCV, 2017.',
    'S. M. Lundberg and S. I. Lee, \u201cA Unified Approach to Interpreting Model Predictions,\u201d Advances '
    'in Neural Information Processing Systems (NeurIPS), 2017.',
    'K. He, X. Zhang, S. Ren, and J. Sun, \u201cDeep Residual Learning for Image Recognition,\u201d Proc. '
    'IEEE CVPR, 2016.',
    'X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, \u201cChestX-ray8: Hospital-Scale '
    'Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of '
    'Common Thorax Diseases,\u201d Proc. IEEE CVPR, 2017.',
    'J. Irvin, P. Rajpurkar, et al., \u201cCheXpert: A Large Chest Radiograph Dataset with Uncertainty '
    'Labels and Expert Comparison,\u201d Proc. AAAI, 2019.',
    'M. T. Ribeiro, S. Singh, and C. Guestrin, \u201cWhy Should I Trust You?: Explaining the Predictions '
    'of Any Classifier,\u201d Proc. ACM SIGKDD, 2016.',
    'R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann, '
    '\u201cShortcut Learning in Deep Neural Networks,\u201d Nature Machine Intelligence, vol. 2, no. 11, pp. '
    '665\u2013673, 2020.',
    'V. Petsiuk, A. Das, and K. Saenko, \u201cRISE: Randomized Input Sampling for Explanation of '
    'Black-box Models,\u201d Proc. British Machine Vision Conference (BMVC), 2018.',
    'K. Lee, K. Lee, H. Lee, and J. Shin, \u201cA Simple Unified Framework for Detecting '
    'Out-of-Distribution Samples and Adversarial Attacks,\u201d Advances in Neural Information '
    'Processing Systems (NeurIPS), 2018.',
    'A. J. DeGrave, J. D. Janizek, and S.-I. Lee, \u201cAI for Radiographic COVID-19 Detection Selects '
    'Shortcuts Over Signal,\u201d Nature Machine Intelligence, vol. 3, no. 7, pp. 610\u2013619, 2021.',
    'J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, \u201cSanity Checks for '
    'Saliency Maps,\u201d Advances in Neural Information Processing Systems (NeurIPS), 2018.',
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph(style='Text')
    run = p.add_run('[%d] %s' % (i, r))
    run.font.size = Pt(9)
    run.font.name = 'Times New Roman'
    p.paragraph_format.left_indent = Inches(0.22)
    p.paragraph_format.first_line_indent = Inches(-0.22)
    p.paragraph_format.space_after = Pt(2)

doc.save(OUT)
print('saved', OUT)
print('sections:', len(doc.sections), '| paragraphs:', len(doc.paragraphs), '| tables:', len(doc.tables))
for i, s in enumerate(doc.sections):
    cols = s._sectPr.find(qn('w:cols'))
    print('  section %d: %.2f x %.2f in, cols=%s'
          % (i, s.page_width.inches, s.page_height.inches,
             cols.get(qn('w:num')) if cols is not None else '?'))
