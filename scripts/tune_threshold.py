"""Pick the unfair-probability threshold for flagging clauses.

Runs BERT on the validation split and shows, for each threshold, how many truly
unfair clauses get flagged (recall) vs. how many fair clauses get flagged by mistake.

    python scripts/tune_threshold.py
    BERT_MODEL_PATH=bert_model_v2 python scripts/tune_threshold.py --split test
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tos_summarizer.config import BERT_MODEL_PATH, DATA_DIR, label2id, get_device
from tos_summarizer.classifier import load_bert, classify_clauses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_bert(BERT_MODEL_PATH, device)
    df = pd.read_csv(DATA_DIR / f"df_{args.split}.csv")

    probs = classify_clauses(df['sentence'].astype(str).tolist(), model, tokenizer, device)
    unfair_prob = 1.0 - probs[:, label2id['clearly_fair']].numpy()
    argmax_unfair = probs.argmax(dim=1).numpy() != label2id['clearly_fair']
    is_unfair = (df['label'] != label2id['clearly_fair']).to_numpy()

    print(f"{'threshold':>9} | {'unfair caught':>13} | {'fair flagged':>12} | flagged")
    for t in [None, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]:
        flagged = argmax_unfair if t is None else (argmax_unfair | (unfair_prob >= t))
        recall = (flagged & is_unfair).sum() / is_unfair.sum()
        fpr = (flagged & ~is_unfair).sum() / (~is_unfair).sum()
        name = "argmax" if t is None else f"{t:.2f}"
        print(f"{name:>9} | {recall:>13.1%} | {fpr:>12.1%} | {flagged.sum()}")


if __name__ == "__main__":
    main()
