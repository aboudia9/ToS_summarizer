"""GPU worker: runs the BART rewriting step for the Pi on this machine's GPU.

    python -m gpu_worker.worker

It only ever connects *out*, to the Pi's worker port (webapp/offload.py): it long-polls
for a batch of clauses, summarizes it, and posts the summaries back. Nothing connects in.

The GPU is only held while there's work. BART is loaded in a child process on the first
batch, and that process exits after IDLE_SECONDS without work, which frees every byte of
VRAM (CUDA's own overhead included). While BART isn't loaded the worker checks free VRAM
first and stays away while something else is using the GPU (a game, an editor), so the
Pi just does the work itself.

Settings (environment variables):
    PI_URL            the Pi's worker address, e.g. https://clause-and-effect.<tailnet>.ts.net:8601
    WORKER_TOKEN      shared secret, same value as on the Pi
    DEVICE            cuda (default); cpu or mps for testing on a machine without one
    IDLE_SECONDS      unload BART after this long without work (default 300)
    MIN_FREE_VRAM_MB  only load BART when at least this much VRAM is free (default 2300)
"""
import json
import logging
import multiprocessing as mp
import os
import time
import urllib.error
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gpu_worker")

PI_URL = os.environ.get("PI_URL", "").rstrip("/")
TOKEN = os.environ.get("WORKER_TOKEN", "")
DEVICE = os.environ.get("DEVICE", "cuda")
IDLE_SECONDS = int(os.environ.get("IDLE_SECONDS", 300))
MIN_FREE_VRAM_MB = int(os.environ.get("MIN_FREE_VRAM_MB", 2300))

POLL_TIMEOUT = 40          # the Pi holds a poll open for 25s
LOAD_TIMEOUT = 180
BATCH_TIMEOUT = 55         # the Pi gives up on a batch after 60s
BUSY_RECHECK_SECONDS = 15
ERROR_BACKOFF_SECONDS = 10


# ---------------- Model process (the only part that touches the GPU) ----------------

def _serve_model(conn, device_name):
    import torch
    from tos_summarizer.config import BART_MODEL_PATH
    from tos_summarizer.summarizer import generate_summaries, load_bart

    device = torch.device(device_name)
    model, tokenizer = load_bart(BART_MODEL_PATH, device)
    conn.send("ready")
    while True:
        batch = conn.recv()
        if batch is None:
            return
        try:
            summaries = generate_summaries(batch, model, tokenizer, device)
            # Peak GPU memory this process has held (PyTorch's share; CUDA adds ~300MB)
            peak_mb = torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else None
            conn.send(("ok", summaries, peak_mb))
        except Exception as e:  # e.g. CUDA out of memory: report it, the Pi runs the batch
            conn.send(("error", repr(e), None))
            if device.type == "cuda":
                torch.cuda.empty_cache()


class ModelProcess:
    def __init__(self):
        ctx = mp.get_context("spawn")
        self.conn, child_conn = ctx.Pipe()
        self.proc = ctx.Process(target=_serve_model, args=(child_conn, DEVICE), daemon=True)
        started = time.monotonic()
        self.proc.start()
        if not self.conn.poll(LOAD_TIMEOUT) or self.conn.recv() != "ready":
            self.stop()
            raise RuntimeError("BART didn't load")
        log.info("BART loaded on %s in %.1fs", DEVICE, time.monotonic() - started)

    def summarize(self, batch):
        self.conn.send(batch)
        if not self.conn.poll(BATCH_TIMEOUT):
            raise RuntimeError("batch timed out")
        status, value, peak_mb = self.conn.recv()
        if status != "ok":
            raise RuntimeError(value)
        return value, peak_mb

    def stop(self):
        try:
            self.conn.send(None)
        except OSError:
            pass
        self.proc.join(10)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join()


# ---------------- GPU check (NVML reads memory without creating a CUDA context) ----------------

def free_vram_mb():
    if DEVICE != "cuda":
        return None
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
            return info.free / 2**20
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        if not getattr(free_vram_mb, "warned", False):
            log.warning("Couldn't read free VRAM (%s); loading BART without checking", e)
            free_vram_mb.warned = True
        return None


# ---------------- Talking to the Pi ----------------

def call_pi(method, path, body=None, timeout=POLL_TIMEOUT):
    req = urllib.request.Request(
        PI_URL + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp) if resp.status == 200 else None


def main():
    if not PI_URL or not TOKEN:
        raise SystemExit("Set PI_URL and WORKER_TOKEN")
    model = None
    last_work = time.monotonic()
    busy_logged = False
    log.info("Polling %s for work", PI_URL)

    while True:
        if model and time.monotonic() - last_work > IDLE_SECONDS:
            model.stop()
            model = None
            log.info("Idle for %ds: unloaded BART, GPU memory released", IDLE_SECONDS)

        if not model:
            free = free_vram_mb()
            if free is not None and free < MIN_FREE_VRAM_MB:
                if not busy_logged:
                    log.info("GPU busy (%.0f MB free); leaving the work to the Pi", free)
                    busy_logged = True
                time.sleep(BUSY_RECHECK_SECONDS)
                continue
            busy_logged = False

        try:
            task = call_pi("GET", "/next")
        except (urllib.error.URLError, OSError) as e:
            log.warning("Can't reach the Pi (%s); retrying in %ds", e, ERROR_BACKOFF_SECONDS)
            time.sleep(ERROR_BACKOFF_SECONDS)
            continue
        if not task:
            continue

        summaries = None
        try:
            if not model:
                model = ModelProcess()
            started = time.monotonic()
            summaries, peak_mb = model.summarize([tuple(item) for item in task["items"]])
            log.info("Summarized %d clauses in %.1fs%s", len(summaries), time.monotonic() - started,
                     f" (GPU peak {peak_mb:.0f} MB)" if peak_mb else "")
        except Exception as e:
            log.warning("Batch failed (%s); the Pi will run it", e)
            if model:
                model.stop()
                model = None
        last_work = time.monotonic()

        try:
            call_pi("POST", "/result", {"id": task["id"], "summaries": summaries})
        except (urllib.error.URLError, OSError) as e:
            log.warning("Couldn't send results back (%s)", e)


if __name__ == "__main__":
    main()
