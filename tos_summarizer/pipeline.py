from dataclasses import dataclass

from .config import id2label, label2id, DEFAULT_UNFAIR_THRESHOLD
from .classifier import classify_clauses
from .preprocessing import split_clauses
from .summarizer import summarize_clauses


@dataclass
class ClauseResult:
    clause: str
    label: str
    unfair_prob: float   # P(potentially_unfair) + P(clearly_unfair)
    summary: str


def process_tos_document(document, bert_model, bart_model, bert_tokenizer, bart_tokenizer, device,
                         unfair_threshold=DEFAULT_UNFAIR_THRESHOLD, progress_callback=None):
    # Step 1 — Split document into individual clauses by sentence
    clauses = split_clauses(document)

    print(f"Found {len(clauses)} clauses in document")

    # Step 2 — Run BERT classification on each clause
    probs = classify_clauses(clauses, bert_model, bert_tokenizer, device)
    fair_id = label2id['clearly_fair']

    # Step 3 — Filter out clearly fair clauses, keep the rest.
    # A clause is flagged if BERT's top pick is unfair (same as before), OR if the
    # combined unfair probability clears the threshold — this catches the borderline
    # clauses BERT would otherwise quietly call fair
    flagged_clauses = []
    flagged_probs = []

    for clause, p in zip(clauses, probs):
        unfair_prob = 1.0 - p[fair_id].item()
        pred = p.argmax().item()

        if pred == fair_id and unfair_prob < unfair_threshold:
            continue

        if pred == fair_id:
            # Use whichever unfair class is more likely
            pred = max((i for i in id2label if i != fair_id), key=lambda i: p[i].item())

        flagged_clauses.append((clause, id2label[pred]))
        flagged_probs.append(unfair_prob)

    print(f"Flagged {len(flagged_clauses)} potentially risky clauses")

    # Step 4 — Run BART summarization on flagged clauses
    summaries = summarize_clauses(
        flagged_clauses, bart_model, bart_tokenizer, device, progress_callback=progress_callback
    )

    return [
        ClauseResult(clause=clause, label=label, unfair_prob=prob, summary=summary)
        for (clause, label), prob, summary in zip(flagged_clauses, flagged_probs, summaries)
    ]


def print_results(results):
    # Step 5 — Format and print plain English output
    print("\n" + "="*50)
    print("TERMS OF SERVICE ANALYSIS")
    print("="*50)

    for r in results:
        if r.label == 'clearly_unfair':
            emoji = '🚨'
        else:
            emoji = '⚠️'

        print(f"\n{emoji} {r.label.upper().replace('_', ' ')}")
        print(f"• {r.summary}")

    print("\n" + "="*50)
