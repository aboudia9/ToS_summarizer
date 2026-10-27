"""Background analysis jobs for the web UI.

Each analysis runs as a job on a single worker thread (the Pi can only afford one model
run at a time), independent of any browser connection. A job keeps an append-only log
of events; the browser streams that log over Server-Sent Events and, if it drops (a
phone locking its screen, switching apps), reconnects and resumes from the last event
it saw. The job itself never notices the disconnect.

Event types, in order:
    queued     {ahead}                      jobs in front of this one
    status     {stage, sentences?}          loading_models / splitting / classifying / summarizing
    clauses    {items, sentences, min_threshold, cached}
                                            every flagged clause (no summaries yet)
    summaries  {items: [{id, summary}]}     one per BART batch, most severe clauses first
    done       {cached}
    failed     {message}   (not "error", which EventSource uses for connection errors)
"""
import logging
import queue
import secrets
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from tos_summarizer.config import BERT_MODEL_PATH, BART_MODEL_PATH, get_device
from tos_summarizer.classifier import load_bert
from tos_summarizer.summarizer import load_bart, summarize_clauses
from tos_summarizer.preprocessing import clean_tos_text, ensure_nltk_data, split_clauses
from tos_summarizer.pipeline import score_clauses
from tos_summarizer.examples import MIN_THRESHOLD, load_cache_index, text_key

log = logging.getLogger(__name__)

JOB_TTL_SECONDS = 2 * 60 * 60   # finished jobs (and their results) are kept this long
MAX_JOBS_KEPT = 200
MAX_WAITING = 3                 # live jobs allowed in the queue before new ones are refused


class QueueFull(Exception):
    pass


@dataclass
class Job:
    id: str
    key: str
    threshold: float
    text: str = field(repr=False)
    created: float = field(default_factory=time.time)
    events: list = field(default_factory=list)
    finished: bool = False
    failed: bool = False

    def emit(self, type_, **data):
        # Appending to a list is atomic in CPython, so readers can slice job.events
        # from the server's event loop without a lock
        self.events.append({"id": len(self.events) + 1, "type": type_, "data": data})
        if type_ in ("done", "failed"):
            self.failed = type_ == "failed"
            self.finished = True
            self.text = ""  # no need to hold on to the document once it's analyzed


def clause_item(i, s):
    return {"id": i, "clause": s.clause, "label": s.label,
            "unfair_prob": round(s.unfair_prob, 4), "top_pick_unfair": s.top_pick_unfair}


class JobManager:
    def __init__(self):
        self.jobs = OrderedDict()
        self.by_input = {}          # (text key, threshold) -> job id, so resubmits reuse a job
        self.waiting = deque()
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.models = None
        self.models_ready = threading.Event()
        self.example_cache = load_cache_index()
        self.offload = None         # set to an offload.Offload to use a GPU worker when present

    def start(self):
        threading.Thread(target=self._load_models, daemon=True, name="model-loader").start()
        threading.Thread(target=self._worker, daemon=True, name="job-worker").start()

    def _load_models(self):
        # Loaded once per process and shared by every job. The server is already up
        # while this runs, so example (cached) results work immediately
        ensure_nltk_data()
        device = get_device()
        bert_model, bert_tokenizer = load_bert(BERT_MODEL_PATH, device)
        bart_model, bart_tokenizer = load_bart(BART_MODEL_PATH, device)
        self.models = (bert_model, bert_tokenizer, bart_model, bart_tokenizer, device)
        self.models_ready.set()
        log.info("Models loaded on %s", device)

    # ---------------- Submitting ----------------

    def submit(self, text, threshold):
        key = text_key(text)
        with self.lock:
            self._prune()
            cached = self.example_cache.get(key)
            # Cached examples hold every clause flagged at MIN_THRESHOLD, so one job serves
            # any threshold; the browser filters them itself
            dedupe = (key, None if cached else threshold)
            existing = self.jobs.get(self.by_input.get(dedupe))
            if existing and not existing.failed:
                return existing

            if not cached and len(self.waiting) >= MAX_WAITING:
                raise QueueFull()

            job = Job(id=secrets.token_urlsafe(12), key=key, threshold=threshold, text=text)
            self.jobs[job.id] = job
            self.by_input[dedupe] = job.id

            if cached:
                self._replay_cached(job, cached)
            else:
                self.waiting.append(job)
                ahead = len(self.waiting) - 1 + (1 if self._running else 0)
                if ahead:
                    job.emit("queued", ahead=ahead)
                self.queue.put(job)
            return job

    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def _replay_cached(self, job, cached):
        job.emit("clauses", sentences=None, min_threshold=MIN_THRESHOLD, cached=True,
                 items=[clause_item(i, s) for i, (s, _) in enumerate(cached)])
        job.emit("summaries", items=[{"id": i, "summary": summary}
                                     for i, (_, summary) in enumerate(cached)])
        job.emit("done", cached=True)

    def _prune(self):
        cutoff = time.time() - JOB_TTL_SECONDS
        for job_id in list(self.jobs):
            job = self.jobs[job_id]
            expired = job.finished and job.created < cutoff
            if expired or (job.finished and len(self.jobs) > MAX_JOBS_KEPT):
                del self.jobs[job_id]
        self.by_input = {k: v for k, v in self.by_input.items() if v in self.jobs}

    # ---------------- Running ----------------

    _running = None

    def _worker(self):
        while True:
            job = self.queue.get()
            with self.lock:
                self.waiting.remove(job)
                self._running = job
                for ahead, waiting_job in enumerate(self.waiting, start=1):
                    waiting_job.emit("queued", ahead=ahead)
            try:
                self._run(job)
            except Exception:
                log.exception("Job %s failed", job.id)
                job.emit("failed", message="Something went wrong analyzing this document.")
            finally:
                with self.lock:
                    self._running = None

    def _run(self, job):
        if not self.models_ready.is_set():
            job.emit("status", stage="loading_models")
            self.models_ready.wait()
        bert_model, bert_tokenizer, bart_model, bart_tokenizer, device = self.models

        job.emit("status", stage="splitting")
        clauses = split_clauses(clean_tos_text(job.text))

        job.emit("status", stage="classifying", sentences=len(clauses))
        flagged = [s for s in score_clauses(clauses, bert_model, bert_tokenizer, device)
                   if s.is_flagged(job.threshold)]
        job.emit("clauses", sentences=len(clauses), min_threshold=job.threshold, cached=False,
                 items=[clause_item(i, s) for i, s in enumerate(flagged)])

        job.emit("status", stage="summarizing")
        # Most severe first, so the clauses people care about fill in first
        for label in ("clearly_unfair", "potentially_unfair"):
            ids = [i for i, s in enumerate(flagged) if s.label == label]
            if not ids:
                continue

            def on_batch(indices, summaries, ids=ids):
                job.emit("summaries", items=[{"id": ids[j], "summary": summary}
                                             for j, summary in zip(indices, summaries)])

            summarize_clauses([(flagged[i].clause, flagged[i].label) for i in ids],
                              bart_model, bart_tokenizer, device, batch_callback=on_batch,
                              offload=self.offload)
        job.emit("done", cached=False)
