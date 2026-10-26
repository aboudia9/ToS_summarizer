"""Bundled example ToS documents, with their analysis precomputed by
scripts/precompute_examples.py so the demo buttons give instant results.

A cached run summarizes every clause that would be flagged at MIN_THRESHOLD (the
lowest the app's slider goes), so it can be re-filtered at any threshold without
running the models. Labels don't depend on the threshold, so this matches a live run.
"""
import hashlib
import json
from dataclasses import dataclass

from .config import ROOT
from .pipeline import ClauseResult, ScoredClause
from .preprocessing import clean_tos_text

EXAMPLES_DIR = ROOT / "examples"
CACHE_DIR = EXAMPLES_DIR / "cache"
MIN_THRESHOLD = 0.10


@dataclass
class Example:
    slug: str
    name: str
    file: str      # relative to ROOT
    source: str
    retrieved: str
    tip: str       # what's worth looking at in the results

    @property
    def path(self):
        return ROOT / self.file

    def read_text(self):
        return self.path.read_text(encoding="utf-8")


def load_examples():
    manifest = json.loads((EXAMPLES_DIR / "manifest.json").read_text(encoding="utf-8"))
    return [Example(**e) for e in manifest if (ROOT / e["file"]).exists()]


def text_key(text):
    # Keyed on the cleaned text, so pasting an example in by hand also hits the cache
    return hashlib.sha256(clean_tos_text(text).encode("utf-8")).hexdigest()


def save_cached(slug, text, scored, summaries):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clauses = [dict(s.__dict__, summary=summary) for s, summary in zip(scored, summaries)]
    payload = {"text_sha256": text_key(text), "min_threshold": MIN_THRESHOLD, "clauses": clauses}
    (CACHE_DIR / f"{slug}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                                            encoding="utf-8")


def load_cache_index():
    # {text_sha256: [(ScoredClause, summary), ...]} for every cached example
    index = {}
    for f in sorted(CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else []:
        payload = json.loads(f.read_text(encoding="utf-8"))
        index[payload["text_sha256"]] = [
            (ScoredClause(c["clause"], c["label"], c["unfair_prob"], c["top_pick_unfair"]), c["summary"])
            for c in payload["clauses"]
        ]
    return index


def cached_results(cached, unfair_threshold):
    # Same filter process_tos_document applies, minus the model calls. Thresholds
    # below MIN_THRESHOLD can't be served from the cache (the slider doesn't go there)
    if unfair_threshold < MIN_THRESHOLD:
        return None
    return [
        ClauseResult(clause=s.clause, label=s.label, unfair_prob=s.unfair_prob, summary=summary)
        for s, summary in cached if s.is_flagged(unfair_threshold)
    ]
