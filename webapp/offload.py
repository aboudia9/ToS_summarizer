"""Hands BART batches to the GPU worker (gpu_worker/) when one is around.

The worker runs on another machine on the tailnet and *calls in* here: it long-polls
GET /next for a batch, then POSTs the summaries to /result. This app never opens a
connection itself, so its container keeps no route to anything. Whenever no worker has
checked in recently, or the one that took a batch doesn't answer in time, the batch just
runs on the Pi's CPU as before.

Served on its own port (WORKER_PORT), which Tailscale exposes to the tailnet only. The
public Funnel address only ever reaches the web UI on 8501.
"""
import hmac
import json
import logging
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)

WORKER_PORT = 8502
POLL_WAIT_SECONDS = 25     # how long a worker's GET /next is held open waiting for work
SEEN_SECONDS = 10          # a worker that polled this recently counts as available
CLAIM_TIMEOUT = 3          # a batch nobody picks up in this long runs locally
RESULT_TIMEOUT = 60        # long enough for the worker to load BART onto its GPU first
MAX_BODY_BYTES = 1_000_000


@dataclass
class Task:
    items: list
    id: str = field(default_factory=lambda: secrets.token_urlsafe(9))
    claimed: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    result: list = None


class Offload:
    def __init__(self, token):
        self.token = token
        self.cond = threading.Condition()
        self.pending = deque()
        self.tasks = {}
        self.polling = 0
        self.last_seen = 0.0

    # ---------------- Called by the job worker (summarize_clauses' offload hook) ----------------

    def __call__(self, batch):
        with self.cond:
            if not self.polling and time.monotonic() - self.last_seen > SEEN_SECONDS:
                return None
            task = Task(items=[[clause, label] for clause, label in batch])
            self.tasks[task.id] = task
            self.pending.append(task)
            self.cond.notify_all()
        try:
            if not task.claimed.wait(CLAIM_TIMEOUT):
                return None
            if not task.done.wait(RESULT_TIMEOUT):
                log.warning("GPU worker didn't return batch %s in time; running it locally", task.id)
                return None
            return task.result
        finally:
            with self.cond:
                self.tasks.pop(task.id, None)
                if task in self.pending:
                    self.pending.remove(task)

    # ---------------- Called by the worker, over HTTP ----------------

    def next_task(self):
        deadline = time.monotonic() + POLL_WAIT_SECONDS
        with self.cond:
            self.polling += 1
            self.last_seen = time.monotonic()
            try:
                while not self.pending:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return None
                    self.cond.wait(left)
                task = self.pending.popleft()
                task.claimed.set()
                return task
            finally:
                self.polling -= 1
                self.last_seen = time.monotonic()

    def finish_task(self, task_id, summaries):
        with self.cond:
            self.last_seen = time.monotonic()
            task = self.tasks.get(task_id)
        if not task or task.done.is_set():
            return  # already given up on and run locally
        ok = (isinstance(summaries, list) and len(summaries) == len(task.items)
              and all(isinstance(s, str) and s for s in summaries))
        task.result = summaries if ok else None
        task.done.set()

    def serve(self):
        server = ThreadingHTTPServer(("0.0.0.0", WORKER_PORT), _Handler)
        server.daemon_threads = True
        server.offload = self
        threading.Thread(target=server.serve_forever, daemon=True, name="offload-server").start()
        log.info("Waiting for a GPU worker on port %d", WORKER_PORT)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _reply(self, status, body=None):
        data = json.dumps(body).encode() if body is not None else b""
        self.send_response(status)
        if data:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self):
        expected = f"Bearer {self.server.offload.token}".encode()
        return hmac.compare_digest(self.headers.get("Authorization", "").encode(), expected)

    def do_GET(self):
        if self.path != "/next":
            return self._reply(404)
        if not self._authorized():
            return self._reply(401)
        task = self.server.offload.next_task()
        if task:
            self._reply(200, {"id": task.id, "items": task.items})
        else:
            self._reply(204)

    def do_POST(self):
        if self.path != "/result":
            return self._reply(404)
        if not self._authorized():
            return self._reply(401)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            return self._reply(413)
        try:
            body = json.loads(self.rfile.read(length))
            self.server.offload.finish_task(str(body["id"]), body.get("summaries"))
        except (ValueError, KeyError, TypeError):
            return self._reply(400)
        self._reply(204)
