import os
from pathlib import Path

import torch

# Project root — everything else is resolved relative to this so paths work
# the same locally, in Docker, and from the training scripts
ROOT = Path(__file__).resolve().parent.parent

# Where the app loads its models from. These default to the compressed (fp16)
# copies made by scripts/export_models.py, but can be pointed anywhere with env vars
# — e.g. BERT_MODEL_PATH=bert_model_v2 to try out a retrained classifier
BERT_MODEL_PATH = Path(os.environ.get("BERT_MODEL_PATH", ROOT / "models" / "bert"))
BART_MODEL_PATH = Path(os.environ.get("BART_MODEL_PATH", ROOT / "models" / "bart"))

# Raw training outputs (what the notebooks/training scripts save to)
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT))

# Base checkpoints we fine-tune from
BERT_BASE_MODEL = "nlpaueb/legal-bert-base-uncased"
BART_BASE_MODEL = "facebook/bart-large-cnn"

# Label mappings shared by every stage
id2label = {0: 'clearly_fair', 1: 'potentially_unfair', 2: 'clearly_unfair'}
label2id = {'clearly_fair': 0, 'potentially_unfair': 1, 'clearly_unfair': 2}

# Same max lengths the models were trained with
BERT_MAX_LENGTH = 256
BART_MAX_INPUT_LENGTH = 256
BART_MAX_TARGET_LENGTH = 128

# Default for flagging a clause: if P(potentially_unfair) + P(clearly_unfair) is at
# least this, we flag it even when 'clearly_fair' is BERT's top pick. Checked on the
# val/test splits with scripts/tune_threshold.py — BERT's misses are mostly confident
# ones, so going lower buys ~1% recall per ~5% more false flags. 0.5 is a mild bump
DEFAULT_UNFAIR_THRESHOLD = float(os.environ.get("UNFAIR_THRESHOLD", 0.5))


# App only: minutes without user activity before the page navigates to a static
# "paused" page, closing its websocket so a pay-per-use host (Cloud Run) can scale
# to zero even if someone leaves the tab open. 0 disables it
IDLE_TIMEOUT_MINUTES = float(os.environ.get("IDLE_TIMEOUT_MINUTES", 15))

# App only: longest document accepted, so one request can't tie up the server for
# ages. The Facebook ToS sample is ~32k characters
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", 100_000))


def get_device():
    # Same GPU check as the notebooks — CUDA, then Apple Silicon, then CPU
    return torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
