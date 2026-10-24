"""Compress the trained models for the app/Docker image.

Saves fp16 copies of the fine-tuned BERT and BART checkpoints (roughly half the size).
The app loads them back as fp32, so inference runs at full precision.

    python scripts/export_models.py
    python scripts/export_models.py --bert bert_model_v2 --bart bart_model_v2
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BartForConditionalGeneration, BartTokenizer)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tos_summarizer.config import ROOT


def export(model_cls, tokenizer_cls, src, dst):
    print(f"{src} -> {dst}")
    model = model_cls.from_pretrained(src, dtype=torch.float16)
    model.save_pretrained(dst)
    tokenizer_cls.from_pretrained(src).save_pretrained(dst)

    # safetensors writes weights as owner-only (600) — make them readable so the
    # non-root user in the container can load them (also needed for the compose mount)
    for f in Path(dst).iterdir():
        f.chmod(0o644)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert", default=ROOT / "bert_model")
    parser.add_argument("--bart", default=ROOT / "bart_model_v2")
    parser.add_argument("--out", default=ROOT / "models")
    args = parser.parse_args()

    out = Path(args.out)
    export(AutoModelForSequenceClassification, AutoTokenizer, args.bert, out / "bert")
    export(BartForConditionalGeneration, BartTokenizer, args.bart, out / "bart")


if __name__ == "__main__":
    main()
