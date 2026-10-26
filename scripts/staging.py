"""Shared by the deploy scripts: copy exactly what a hosted build needs into a folder,
so nothing else in the repo (CSVs, raw models, .env, notebooks) ever gets uploaded."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_REPO = "bigboub/clause-and-effect-models"


def stage_app(dest, model_repo=DEFAULT_MODEL_REPO):
    dest = Path(dest)
    shutil.copytree(ROOT / "tos_summarizer", dest / "tos_summarizer",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for d in ["webapp", "examples"]:
        shutil.copytree(ROOT / d, dest / d, ignore=shutil.ignore_patterns("__pycache__"))
    for f in ["fb_tos_testing.txt", "requirements-app.txt"]:
        shutil.copy(ROOT / f, dest / f)
    dockerfile = (ROOT / "deploy" / "Dockerfile.hosted").read_text()
    (dest / "Dockerfile").write_text(dockerfile.replace(DEFAULT_MODEL_REPO, model_repo))
    return dest
