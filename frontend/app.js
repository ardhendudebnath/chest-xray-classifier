/* Chest X-ray Classifier — frontend logic.
 *
 * Talks to the FastAPI service: GET /health and POST /analyze, which returns
 * the prediction, the Grad-CAM overlay and the SHAP attributions in one
 * response. /predict and /explain remain on the service for callers that want
 * only one of the three; this page always wants all of them.
 * No framework and no build step, matching the triage app.
 *
 * The API address is configurable and stored locally, because the interesting
 * case is a phone on the same Wi-Fi talking to a laptop — which means the page
 * and the API are on different origins. The service sets CORS to allow any
 * origin, so this works without a proxy.
 */

'use strict';

/* Must match src/dataset.py CLASSES, in the same order. /health returns the
 * server's list, and checkHealth replaces this with it -- these values are only
 * the fallback for the moments before that lands, and a mismatch between the
 * two would mislabel every bar. */
let CLASSES = ['COVID19', 'LUNG_OPACITY', 'NORMAL', 'PNEUMONIA'];
const MAX_BYTES = 15 * 1024 * 1024;

const $ = (id) => document.getElementById(id);

const els = {
  drop: $('drop'), preview: $('preview'), file: $('file'), camera: $('camera'),
  pick: $('pick'), shoot: $('shoot'), clear: $('clear'), fileName: $('file-name'),
  run: $('run'), status: $('status'),
  results: $('results'), verdict: $('verdict'), bars: $('bars'),
  warnOod: $('warn-ood'), warnLowConf: $('warn-lowconf'), warnNoCheck: $('warn-nocheck'),
  camCard: $('cam-card'), cam: $('cam'), camClass: $('cam-class'),
  camRun: $('cam-run'), camCaption: $('cam-caption'),
  shapCard: $('shap-card'), shapSubject: $('shap-subject'),
  shapBars: $('shap-bars'), shapCoverage: $('shap-coverage'),
  health: $('health'), healthDetail: $('health-detail'),
  settings: $('settings'), settingsOpen: $('settings-open'), api: $('api'),
};

let selected = null;      // the File the user chose
let previewUrl = null;    // object URL for the preview, revoked on replace

/* ------------------------------------------------------------------ config */

/* Default to this machine on the API's port. Serving the page from the laptop
 * and opening it on a phone means location.hostname is already the LAN address,
 * so this guesses right without being told. */
function defaultApi() {
  const host = location.hostname || 'localhost';
  return `${location.protocol}//${host}:8100`;
}

function apiBase() {
  return (localStorage.getItem('cxr-api') || defaultApi()).replace(/\/+$/, '');
}

/* --------------------------------------------------------------- selection */

function selectFile(file) {
  if (!file) return;

  if (!/^image\//.test(file.type)) {
    setStatus('That is not an image file. PNG, JPEG or BMP.', true);
    return;
  }
  if (file.size > MAX_BYTES) {
    setStatus(`Image is ${Math.round(file.size / 1024)} KB; the limit is 15 MB.`, true);
    return;
  }

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  selected = file;

  els.preview.src = previewUrl;
  els.preview.hidden = false;
  els.drop.classList.add('has-image');
  els.drop.querySelector('.drop-idle').hidden = true;
  els.clear.hidden = false;
  els.fileName.textContent = `${file.name} · ${Math.round(file.size / 1024)} KB`;
  els.run.disabled = false;

  hideResults();
  setStatus('');
}

function clearFile() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  selected = null;

  els.preview.hidden = true;
  els.preview.removeAttribute('src');
  els.drop.classList.remove('has-image');
  els.drop.querySelector('.drop-idle').hidden = false;
  els.clear.hidden = true;
  els.fileName.textContent = '';
  els.run.disabled = true;
  els.file.value = els.camera.value = '';

  hideResults();
  setStatus('');
}

function hideResults() {
  els.results.hidden = true;
  els.camCard.hidden = true;
  els.shapCard.hidden = true;
}

function setStatus(text, isError) {
  els.status.textContent = text;
  els.status.style.color = isError ? 'var(--stop)' : '';
}

/* ---------------------------------------------------------------- requests */

/* FastAPI puts its message in `detail`; surfacing it beats a bare status code,
 * since the useful ones (503 no model, 413 too large) explain themselves. */
async function errorText(response) {
  try {
    const body = await response.json();
    if (body && body.detail) return body.detail;
  } catch { /* not JSON — fall through */ }
  return `Request failed (HTTP ${response.status}).`;
}

/* One call, not three. /analyze returns the prediction, the heatmap and the
 * attributions together because they are meant to be read together -- fetching
 * them separately is what lets an interface show the picture and quietly drop
 * the numbers. /predict and /explain still exist for callers that want one. */
async function analyse() {
  if (!selected) return;

  els.run.disabled = true;
  setStatus('Analysing…');

  try {
    const data = await requestAnalysis(els.camClass.value);
    renderPrediction(data);
    renderExplanation(data);
    setStatus('');
  } catch (error) {
    setStatus(describeNetworkError(error), true);
  } finally {
    els.run.disabled = false;
  }
}

async function requestAnalysis(className) {
  const form = new FormData();
  form.append('file', selected, selected.name);

  const url = `${apiBase()}/analyze`
    + (className ? `?class_name=${encodeURIComponent(className)}` : '');

  const response = await fetch(url, { method: 'POST', body: form });
  if (!response.ok) throw new Error(await errorText(response));
  return response.json();
}

/* A cross-origin fetch to a host that is not listening fails as an opaque
 * TypeError with no status. On a phone that almost always means the API address
 * is wrong or the server is bound to localhost only, so say that rather than
 * printing "Failed to fetch". */
function describeNetworkError(error) {
  if (error instanceof TypeError) {
    return `Could not reach the API at ${apiBase()}. Check the address in ` +
           `settings, and that the server was started with --host 0.0.0.0.`;
  }
  return error.message;
}

function renderPrediction(data) {
  els.results.hidden = false;
  els.verdict.textContent = data.prediction;

  /* Three distinct states, and the null one is not the negative one.
   * `out_of_distribution === null` means the service had no fitted statistics
   * and never ran the check; showing nothing there would let an untested image
   * look like one that passed. Compared with === true and === false so that a
   * null never falls through into either branch. */
  els.warnOod.hidden = data.out_of_distribution !== true;
  els.warnNoCheck.hidden = data.out_of_distribution !== null
    && data.out_of_distribution !== undefined;

  /* Only worth showing when the stronger warning is not already up, otherwise
   * two boxes say overlapping things about the same image. */
  els.warnLowConf.hidden = !data.low_confidence || data.out_of_distribution === true;

  const entries = CLASSES
    .map((name) => [name, data.probabilities[name] ?? 0])
    .sort((a, b) => b[1] - a[1]);

  els.bars.replaceChildren(...entries.map(([name, value], index) => {
    const row = document.createElement('div');
    row.className = index === 0 ? 'bar-row top' : 'bar-row';
    row.innerHTML = `
      <div class="bar-head">
        <span class="bar-name"></span>
        <span class="bar-pct"></span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${(value * 100).toFixed(1)}%"></div>
      </div>`;
    row.querySelector('.bar-name').textContent = name;
    row.querySelector('.bar-pct').textContent = `${(value * 100).toFixed(1)}%`;
    return row;
  }));
}

/* Re-run for a different class. Both explanations are class-specific, so the
 * map and the bars have to move together -- leaving the old attributions
 * beside a new heatmap would caption one class's reasoning with another's. */
async function reExplain() {
  if (!selected) return;

  els.camCaption.textContent = 'Generating heatmap…';

  try {
    renderExplanation(await requestAnalysis(els.camClass.value));
  } catch (error) {
    els.camCaption.textContent = describeNetworkError(error);
  }
}

function renderExplanation(data) {
  els.camCard.hidden = false;

  /* A data URL rather than an object URL: the PNG arrives inside the JSON, so
   * there is no Blob to revoke and nothing to leak. */
  els.cam.src = `data:image/png;base64,${data.gradcam_png_base64}`;

  /* Reported separately because they come apart whenever a class is requested:
   * the map then answers "why not pneumonia?" while the model's actual call is
   * still something else. Collapsing them would put two different classes'
   * numbers under one label. */
  const predicted = data.prediction;
  const explained = data.explained_class;

  els.camCaption.textContent = (predicted && explained && predicted !== explained)
    ? `Model predicted ${predicted}. This map answers for ${explained}.`
    : `Heatmap for ${explained || predicted || 'the predicted class'}.`;

  renderShap(data);
}

function renderShap(data) {
  const shap = data.shap;

  /* Null means the service had no background loaded and never computed the
   * attribution. Rendering an empty chart would say "nothing contributed",
   * which is a far stronger claim than "this was not measured" -- the same
   * distinction the out-of-distribution notice exists to preserve. */
  if (!shap || !Array.isArray(shap.features) || !shap.features.length) {
    els.shapCard.hidden = true;
    return;
  }

  els.shapCard.hidden = false;
  els.shapSubject.textContent =
    `Toward or against ${data.explained_class}, per learned feature. `
    + `The bars below sum with a base of ${shap.base_value.toFixed(2)} to the `
    + `model's score of ${shap.logit.toFixed(2)}.`;

  /* Scaled to the largest bar shown, so the longest one fills half the track.
   * The absolute logit scale means nothing to a reader; the relative sizes do. */
  const peak = Math.max(...shap.features.map((row) => Math.abs(row.value))) || 1;

  els.shapBars.replaceChildren(...shap.features.map((row) => {
    const element = document.createElement('div');
    element.className = 'shap-row';
    element.innerHTML = `
      <span class="shap-name"></span>
      <span class="shap-track">
        <span class="shap-fill ${row.value >= 0 ? 'pos' : 'neg'}"
              style="width:${(Math.abs(row.value) / peak * 50).toFixed(1)}%"></span>
      </span>
      <span class="shap-value"></span>`;
    element.querySelector('.shap-name').textContent = `feature ${row.feature}`;
    element.querySelector('.shap-value').textContent = row.value >= 0
      ? `+${row.value.toFixed(3)}`
      : row.value.toFixed(3);
    return element;
  }));

  /* The number that decides how the chart should be read. Fifteen bars out of
   * 512 carrying 15% of the movement is a sample of the reasoning, not a
   * summary of it, and the chart cannot show that about itself. */
  const percent = Math.round((shap.coverage ?? 0) * 100);
  els.shapCoverage.textContent =
    `These ${shap.features.length} of ${shap.feature_count} features account for `
    + `${percent}% of what moved the score. The rest is spread across the others — `
    + `this is a sample of the model's reasoning, not the whole of it.`;
}

/* Rebuild the "explain class" dropdown from whatever the server said its
 * classes are, keeping the current selection if it still exists. */
function syncExplainOptions() {
  const previous = els.camClass.value;

  els.camClass.replaceChildren(...[
    ['', 'Predicted class'],
    ...CLASSES.map((name) => [name, name]),
  ].map(([value, label]) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    return option;
  }));

  els.camClass.value = CLASSES.includes(previous) ? previous : '';
}

/* ------------------------------------------------------------------ health */

async function checkHealth() {
  try {
    const response = await fetch(`${apiBase()}/health`);
    const data = await response.json();

    /* Take the server's class list rather than trusting the constant above.
     * The page and the API can be different versions -- a phone with the old
     * page cached talking to a rebuilt laptop is the normal case here -- and a
     * stale list would put the wrong name on every probability bar. */
    if (Array.isArray(data.classes) && data.classes.length) {
      CLASSES = data.classes;
      syncExplainOptions();
    }

    if (data.model_loaded) {
      els.health.textContent = `API ready · ${data.backbone} on ${data.device}`;
      els.health.className = 'pill pill-ok';
      const f1 = data.val_metrics && data.val_metrics.macro_f1;
      const notes = [];
      if (f1) {
        notes.push(`validation macro F1 ${f1.toFixed(4)} — on data whose ` +
                   `classes are separable by source`);
      }
      /* Worth saying before an upload rather than after. Without these the
       * service answers every image, including the ones it should refuse. */
      if (!data.ood_stats_loaded) {
        notes.push('no out-of-distribution statistics loaded — nothing is ' +
                   'checked for being a chest X-ray');
      }
      /* Same reasoning: without a background the attribution panel simply does
       * not appear, and its absence is otherwise indistinguishable from a
       * model whose features happened to contribute nothing. */
      if (!data.shap_background_loaded) {
        notes.push('no SHAP background loaded — per-feature attributions are ' +
                   'not computed');
      }
      els.healthDetail.textContent = notes.join(' · ');
    } else {
      els.health.textContent = 'API up, no model loaded';
      els.health.className = 'pill pill-bad';
      els.healthDetail.textContent = 'Train a checkpoint, or set CXR_CHECKPOINT.';
    }
  } catch {
    els.health.textContent = 'API unreachable';
    els.health.className = 'pill pill-bad';
    els.healthDetail.textContent = apiBase();
  }
}

/* ------------------------------------------------------------------ wiring */

els.pick.addEventListener('click', () => els.file.click());
els.drop.addEventListener('click', () => { if (!selected) els.file.click(); });
els.drop.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); els.file.click(); }
});
els.shoot.addEventListener('click', () => els.camera.click());
els.clear.addEventListener('click', clearFile);

els.file.addEventListener('change', (event) => selectFile(event.target.files[0]));
els.camera.addEventListener('change', (event) => selectFile(event.target.files[0]));

for (const type of ['dragenter', 'dragover']) {
  els.drop.addEventListener(type, (event) => {
    event.preventDefault();
    els.drop.classList.add('over');
  });
}
for (const type of ['dragleave', 'drop']) {
  els.drop.addEventListener(type, (event) => {
    event.preventDefault();
    els.drop.classList.remove('over');
  });
}
els.drop.addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));

els.run.addEventListener('click', analyse);
els.camRun.addEventListener('click', reExplain);

els.settingsOpen.addEventListener('click', () => {
  els.api.value = apiBase();
  els.settings.showModal();
});
els.settings.addEventListener('close', () => {
  if (els.settings.returnValue !== 'save') return;
  const value = els.api.value.trim();
  if (value) localStorage.setItem('cxr-api', value.replace(/\/+$/, ''));
  else localStorage.removeItem('cxr-api');
  checkHealth();
});

/* Only offer the camera button where a rear camera plausibly exists. On a
 * desktop the capture attribute is ignored and the button is a duplicate. */
if (/Android|iPhone|iPad|iPod/i.test(navigator.userAgent)) els.shoot.hidden = false;

/* Service workers need a secure context. Over plain HTTP on a LAN address this
 * silently does nothing, which is fine — the page still works, and both phones
 * can still add it to the home screen manually. */
if ('serviceWorker' in navigator && window.isSecureContext) {
  navigator.serviceWorker.register('sw.js').catch(() => { /* not fatal */ });
}

checkHealth();
