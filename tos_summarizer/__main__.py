"""Command-line version of the pipeline: python -m tos_summarizer path/to/tos.{txt,pdf,docx,html}"""
import argparse

from .config import BERT_MODEL_PATH, BART_MODEL_PATH, DEFAULT_UNFAIR_THRESHOLD, get_device
from .classifier import load_bert
from .summarizer import load_bart
from .preprocessing import load_tos_document
from .pipeline import process_tos_document, print_results


def main():
    parser = argparse.ArgumentParser(description="Summarize the risky clauses in a ToS document")
    parser.add_argument("file", help="Path to a ToS document (.txt, .pdf, .docx or .html)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_UNFAIR_THRESHOLD,
                        help="Flag clauses whose combined unfair probability is at least this")
    args = parser.parse_args()

    device = get_device()
    bert_model, bert_tokenizer = load_bert(BERT_MODEL_PATH, device)
    bart_model, bart_tokenizer = load_bart(BART_MODEL_PATH, device)

    document = load_tos_document(args.file)
    results = process_tos_document(
        document, bert_model, bart_model, bert_tokenizer, bart_tokenizer, device,
        unfair_threshold=args.threshold
    )
    print_results(results)


if __name__ == "__main__":
    main()
