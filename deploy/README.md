# Deploying to a Hugging Face Space

One container, Docker SDK. `uvicorn` answers `/predict` and `/explain`, and the
same process serves the page — locally those are two servers on two ports, which
is better while editing, but a Space has nowhere to put a second one.

## Before you start

**This publishes the model weights.** `checkpoints/` is gitignored in this
repository on purpose; a Space needs them, so `build_space.py` copies them in
and tracks them with LFS. Pushing makes `best.pt` downloadable by anyone who can
see the Space. If that is not what you want, make the Space private.

**It also puts a chest X-ray classifier in front of the public.** The frontend's
disclaimers are not dismissible and `/explain` returns one in an `X-Disclaimer`
header, which is the reason both were built that way. Read
[What this is not](../README.md#what-this-is-not) before pointing anyone at the
URL.

## Steps

**1. Create the Space.** At https://huggingface.co/new-space — pick a name,
choose **Docker** → **Blank** as the SDK, and public or private. Leave the
hardware on the free CPU tier; resnet18 inference on one image takes tens of
milliseconds and needs no GPU.

**2. Assemble the directory.**

```bash
~/venvs/smt/Scripts/python.exe -m deploy.build_space --out ~/cxr-space
```

Copies `app/`, `src/`, `frontend/` and both checkpoints, and writes the
`Dockerfile`, `requirements.txt`, `README.md`, `.gitattributes` and
`.dockerignore` that a Space expects at its root. About 45 MB, almost all of it
`best.pt`.

**3. Push it.**

```bash
cd ~/cxr-space
git init
git lfs install
git remote add origin https://huggingface.co/spaces/<your-user>/<your-space>
git add -A
git commit -m "Deploy chest X-ray classifier"
git push -u origin main
```

Hugging Face asks for a username and an **access token** — not your password.
Make one at https://huggingface.co/settings/tokens with **write** scope.

`git lfs install` is not optional. Without it the two `.pt` files push as
ordinary blobs, and the Hub rejects anything over 10 MB that is not LFS.

**4. Wait for the build.** Five to ten minutes the first time, mostly installing
torch. The Space's **Logs** tab shows progress; **App** shows the running page.

## Checking it worked

```
https://<your-user>-<your-space>.hf.space/health
```

```json
{"model_loaded": true, "device": "cpu", "ood_stats_loaded": true}
```

`ood_stats_loaded: false` means `ood.pt` did not make it, and the check that
refuses non-radiographs is silently not running. `model_loaded: false` means the
same for `best.pt` — the API answers, and 503s every prediction.

## If the build fails

| symptom | cause |
|---|---|
| `libgomp.so.1: cannot open shared object file` | The `apt-get install libgomp1` line was dropped. torch needs it and `python:slim` has not got it. |
| Image too large, or a very long build | The CPU index URL was lost from the `pip install`. The default one resolves the CUDA build: several GB of kernels for a machine with no GPU. |
| `no_model` on `/health` | The checkpoints were not committed, usually because `git lfs install` was skipped and the push silently dropped them. |
| Page loads, every prediction fails | The API is not reachable. Check `CXR_SERVE_FRONTEND=1` survived in the Dockerfile — without it nothing serves the page, so this normally means the opposite is broken. |

## What is deliberately not in the image

`tests/`, `notebooks/`, and the training and analysis modules — they would build
fine, they are just weight for code no request path reaches.

`scikit-learn` is also absent. Only `src.ood`'s *fitting* path uses it, through a
function-level import of `LedoitWolf`. Scoring against already-fitted statistics,
which is all the API does, never reaches it. `matplotlib` **is** needed despite
appearing in no import at the top of any serving module: `src/gradcam_utils.py`
imports `colormaps` inside the function that colours the heatmap, so without it
`/explain` fails at request time rather than at startup.
