"""Deploy to Hugging Face: upload the compressed models and push the app to the Space.

Needs `hf auth login` first. Re-run any time to push updates.

    python scripts/deploy_hf.py              # models + app
    python scripts/deploy_hf.py --app-only   # just code changes (no 1GB upload)
    python scripts/deploy_hf.py --models-only
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tos_summarizer.config import ROOT
from scripts.staging import stage_app

DEPLOY_DIR = ROOT / "deploy" / "huggingface"


def upload_models(api, repo_id):
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=DEPLOY_DIR / "MODEL_CARD.md", path_in_repo="README.md", repo_id=repo_id)
    api.upload_folder(folder_path=ROOT / "models", repo_id=repo_id, allow_patterns=["bert/*", "bart/*"],
                      commit_message="Upload fp16 BERT + BART models")
    print(f"Models: https://huggingface.co/{repo_id}")


def upload_app(api, space_id, model_repo):
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)

    # Stage exactly what the Space needs, so nothing else in the repo gets uploaded
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stage_app(tmp, model_repo)
        shutil.copy(DEPLOY_DIR / "SPACE_README.md", tmp / "README.md")

        api.upload_folder(folder_path=tmp, repo_id=space_id, repo_type="space",
                          delete_patterns=["*.py", "tos_summarizer/**"],
                          commit_message="Deploy app")

    print(f"Space: https://huggingface.co/spaces/{space_id}")
    print(f"App:   https://{space_id.replace('/', '-').replace('_', '-').lower()}.hf.space")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="bigboub/clause-and-effect")
    parser.add_argument("--models", default="bigboub/clause-and-effect-models")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--app-only", action="store_true")
    group.add_argument("--models-only", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    print(f"Logged in as {api.whoami()['name']}")

    if not args.app_only:
        if not (ROOT / "models" / "bert").exists():
            sys.exit("models/ not found — run scripts/export_models.py first")
        upload_models(api, args.models)
    if not args.models_only:
        upload_app(api, args.space, args.models)


if __name__ == "__main__":
    main()
