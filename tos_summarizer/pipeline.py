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


@dataclass
class ScoredClause:
    clause: str
    label: str             # BERT's pick, or the likelier unfair class if BERT picked fair
    unfair_prob: float     # P(potentially_unfair) + P(clearly_unfair)
    top_pick_unfair: bool  # BERT's top pick was one of the unfair classes

    def is_flagged(self, unfair_threshold):
        # A clause is flagged if BERT's top pick is unfair (same as before), OR if the
        # combined unfair probability clears the threshold — this catches the borderline
        # clauses BERT would otherwise quietly call fair
        return self.top_pick_unfair or self.unfair_prob >= unfair_threshold


def score_clauses(clauses, bert_model, bert_tokenizer, device):
    # Run BERT classification on each clause. Nothing here depends on the threshold,
    # so one scoring pass can be re-filtered at any threshold (used to cache examples)
    probs = classify_clauses(clauses, bert_model, bert_tokenizer, device)
    fair_id = label2id['clearly_fair']
    unfair_ids = [i for i in id2label if i != fair_id]

    scored = []
    for clause, p in zip(clauses, probs):
        pred = p.argmax().item()
        top_pick_unfair = pred != fair_id
        if not top_pick_unfair:
            # Use whichever unfair class is more likely
            pred = max(unfair_ids, key=lambda i: p[i].item())
        scored.append(ScoredClause(clause, id2label[pred], 1.0 - p[fair_id].item(), top_pick_unfair))
    return scored


def process_tos_document(document, bert_model, bart_model, bert_tokenizer, bart_tokenizer, device,
                         unfair_threshold=DEFAULT_UNFAIR_THRESHOLD, progress_callback=None):
    # Step 1 — Split document into individual clauses by sentence
    clauses = split_clauses(document)

    print(f"Found {len(clauses)} clauses in document")

    # Step 2 & 3 — Classify every clause, keep the ones that aren't clearly fair
    flagged = [s for s in score_clauses(clauses, bert_model, bert_tokenizer, device)
               if s.is_flagged(unfair_threshold)]

    print(f"Flagged {len(flagged)} potentially risky clauses")

    # Step 4 — Run BART summarization on flagged clauses
    summaries = summarize_clauses(
        [(s.clause, s.label) for s in flagged], bart_model, bart_tokenizer, device,
        progress_callback=progress_callback
    )

    return [
        ClauseResult(clause=s.clause, label=s.label, unfair_prob=s.unfair_prob, summary=summary)
        for s, summary in zip(flagged, summaries)
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
