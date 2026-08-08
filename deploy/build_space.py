"""Assemble a directory ready to push to a Hugging Face Space.

A Space is a git repo with a particular shape -- Dockerfile and a README
carrying YAML frontmatter, both at the root -- which is not the shape of this
project. Rather than document a dozen copy commands that drift, this builds the
directory.

    python -m deploy.build_space --out ../cxr-space

What it will not do is push. That needs your Hugging Face credentials, and the
weights are yours to publish or not.

**The checkpoints are the reason this needs a script at all.** They are
gitignored here, deliberately -- a 45 MB binary does not belong in a source
repo, and neither does anything trained on medical images by reflex. A Space
does need them, so they are copied in and the generated .gitattributes tracks
them with LFS. Pushing this directory publishes those weights.
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY = PROJECT_ROOT / "deploy"

# Everything the container needs to serve, and nothing else. Notably absent:
# tests/, notebooks/, and the training and analysis modules that only run
# offline -- they would build fine, they are just image weight for code no
# request path reaches.
TREES = ["app", "src", "frontend"]

CHECKPOINTS = ["best.pt", "ood.pt"]

# Only the two that Spaces read from the root. requirements.txt is renamed on
# the way in because the project's own is a different, longer file.
FILES = {
    "Dockerfile": "Dockerfile",
    "requirements.txt": "requirements.txt",
    "space_readme.md": "README.md",
}

GITATTRIBUTES = "*.pt filter=lfs diff=lfs merge=lfs -text\n"

# src/ imports assume the project root is importable, which it is inside the
# container because WORKDIR is the repo root. Nothing to do but say so.
DOCKERIGNORE = """\
__pycache__/
*.pyc
.git/
"""


def build(out, checkpoints_dir):
    out = Path(out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        # Clearing the tracked trees rather than the directory, so a .git left
        # by `git clone` of the Space survives being rebuilt into.
        for name in TREES + ["checkpoints"]:
            shutil.rmtree(out / name, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    for name in TREES:
        shutil.copytree(
            PROJECT_ROOT / name,
            out / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        print(f"copied {name}/")

    for source, target in FILES.items():
        shutil.copy2(DEPLOY / source, out / target)
        print(f"copied deploy/{source} -> {target}")

    (out / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    (out / ".dockerignore").write_text(DOCKERIGNORE, encoding="utf-8")

    checkpoints_dir = Path(checkpoints_dir).expanduser()
    (out / "checkpoints").mkdir(exist_ok=True)
    missing = []
    for name in CHECKPOINTS:
        source = checkpoints_dir / name
        if source.is_file():
            shutil.copy2(source, out / "checkpoints" / name)
            print(f"copied checkpoints/{name} ({source.stat().st_size / 1e6:.0f} MB)")
        else:
            missing.append(str(source))

    if missing:
        raise SystemExit(
            "missing checkpoints: " + ", ".join(missing) + "\n"
            "The Space starts without them, reports no_model on /health and 503s "
            "every prediction, which looks like a broken deployment rather than "
            "an absent file. Train first, or pass --checkpoints."
        )

    print(f"\nready: {out}")
    print("\nnext, from that directory:")
    print("  git init && git lfs install")
    print("  git remote add origin https://huggingface.co/spaces/<user>/<space>")
    print("  git add -A && git commit -m 'Deploy chest X-ray classifier'")
    print("  git push -u origin main")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="Directory to assemble into.")
    parser.add_argument(
        "--checkpoints",
        default=str(PROJECT_ROOT / "checkpoints"),
        help="Where best.pt and ood.pt live.",
    )
    args = parser.parse_args()
    build(args.out, args.checkpoints)


if __name__ == "__main__":
    main()
