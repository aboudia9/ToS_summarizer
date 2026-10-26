"""Precompute the analysis of the bundled example documents (examples/manifest.json),
so the app's example buttons show results instantly instead of running the models.

    python scripts/precompute_examples.py              # all examples
    python scripts/precompute_examples.py tiktok x     # just these

Rerun after retraining or re-exporting either model. Runs on CPU so the cached
output matches what the CPU-only deployments would produce.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tos_summarizer.config import BERT_MODEL_PATH, BART_MODEL_PATH
from tos_summarizer.classifier import load_bert
from tos_summarizer.summarizer import load_bart, summarize_clauses
from tos_summarizer.preprocessing import clean_tos_text, ensure_nltk_data, split_clauses
from tos_summarizer.pipeline import score_clauses
from tos_summarizer.examples import MIN_THRESHOLD, load_examples, save_cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slugs", nargs="*", help="Only these examples (default: all)")
    args = parser.parse_args()

    ensure_nltk_data()
    bert_model, bert_tokenizer = load_bert(BERT_MODEL_PATH, "cpu")
    bart_model, bart_tokenizer = load_bart(BART_MODEL_PATH, "cpu")

    for ex in load_examples():
        if args.slugs and ex.slug not in args.slugs:
            continue
        start = time.time()
        text = ex.read_text()
        clauses = split_clauses(clean_tos_text(text))
        scored = [s for s in score_clauses(clauses, bert_model, bert_tokenizer, "cpu")
                  if s.is_flagged(MIN_THRESHOLD)]
        summaries = summarize_clauses([(s.clause, s.label) for s in scored],
                                      bart_model, bart_tokenizer, "cpu")
        save_cached(ex.slug, text, scored, summaries)
        print(f"{ex.name}: {len(clauses)} clauses, {len(scored)} cached ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
