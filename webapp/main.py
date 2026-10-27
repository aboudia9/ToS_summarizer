"""Web UI + API for the ToS summarizer.

    uvicorn webapp.main:app --port 8501

Run a single worker process: the models live in memory once and jobs are queued
in-process (see jobs.py).
"""
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tos_summarizer.config import DEFAULT_UNFAIR_THRESHOLD, MAX_INPUT_CHARS
from tos_summarizer.examples import MIN_THRESHOLD, load_examples
from .jobs import JobManager, QueueFull
from .offload import Offload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).parent / "static"
GITHUB_URL = "https://github.com/aboudia9/ToS_summarizer"
MAX_BODY_BYTES = 1_000_000
HEARTBEAT_SECONDS = 15

# Pages allowed to embed the app in an iframe (the portfolio, its Vercel previews, local dev)
FRAME_ANCESTORS = ("'self' https://boudia.dev https://*.boudia.dev https://*.vercel.app "
                   "http://localhost:*")
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        f"connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors {FRAME_ANCESTORS}"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}

jobs = JobManager()
# Shared secret for the optional GPU worker (see offload.py); without it the Pi does everything
if os.environ.get("WORKER_TOKEN"):
    jobs.offload = Offload(os.environ["WORKER_TOKEN"])
examples = {ex.slug: ex for ex in load_examples()}


@asynccontextmanager
async def lifespan(app):
    jobs.start()
    if jobs.offload:
        jobs.offload.serve()
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def limits_and_headers(request: Request, call_next):
    # Refuse oversized bodies up front instead of reading them into memory
    if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "Request too large."}, status_code=413)
    response = await call_next(request)
    response.headers.update(SECURITY_HEADERS)
    return response


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/healthz")
def healthz():
    return {"ok": True, "models_ready": jobs.models_ready.is_set()}


@app.get("/api/config")
def config():
    return {
        "examples": [{"slug": ex.slug, "name": ex.name, "tip": ex.tip, "source": ex.source,
                      "retrieved": ex.retrieved} for ex in examples.values()],
        "default_threshold": DEFAULT_UNFAIR_THRESHOLD,
        "min_threshold": MIN_THRESHOLD,
        "max_chars": MAX_INPUT_CHARS,
        "github": GITHUB_URL,
        "models_ready": jobs.models_ready.is_set(),
    }


@app.get("/api/examples/{slug}")
def example_text(slug: str):
    ex = examples.get(slug)
    if not ex:
        raise HTTPException(404, "No such example.")
    return {"slug": ex.slug, "text": ex.read_text()}


class JobRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    threshold: float = Field(default=DEFAULT_UNFAIR_THRESHOLD, ge=MIN_THRESHOLD, le=0.9)


@app.post("/api/jobs")
def create_job(req: JobRequest):
    if not req.text.strip():
        raise HTTPException(422, "Paste some Terms of Service text first.")
    try:
        job = jobs.submit(req.text, round(req.threshold, 2))
    except QueueFull:
        raise HTTPException(429, "The server is busy with other documents right now. "
                                 "Try one of the examples, or try again in a few minutes.")
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "This analysis has expired. Run it again.")
    return {"finished": job.finished, "failed": job.failed}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, after: int = 0):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "This analysis has expired. Run it again.")

    # Resume after the last event the browser saw: EventSource sends Last-Event-ID on its
    # own reconnects, and the page passes ?after= when it reopens the stream itself
    last_seen = request.headers.get("last-event-id")
    sent = max(after, int(last_seen) if last_seen and last_seen.isdigit() else 0)

    async def stream():
        nonlocal sent
        last_write = time.monotonic()
        yield "retry: 3000\n\n"
        while True:
            for event in job.events[sent:]:
                yield f"id: {event['id']}\nevent: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                sent = event["id"]
                last_write = time.monotonic()
            if job.finished and sent >= len(job.events):
                return
            if await request.is_disconnected():
                return
            if time.monotonic() - last_write > HEARTBEAT_SECONDS:
                yield ": ping\n\n"  # keeps proxies from closing a quiet connection
                last_write = time.monotonic()
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
