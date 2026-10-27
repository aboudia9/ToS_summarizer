# GPU worker image for the desktop (gpu_worker/worker.py). Build context: the repo root.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CUDA 12.6 build of torch: runs on the GTX 1650 (Turing) with any recent Game Ready driver
COPY requirements-app.txt .
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126 \
 && pip install -r requirements-app.txt nvidia-ml-py==13.615.71

RUN useradd --create-home --uid 1000 appuser && chown appuser /app
USER appuser

# Only BART: the Pi still does the (quick) BERT step itself
ARG MODEL_REPO=bigboub/clause-and-effect-models
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_REPO}', local_dir='models', allow_patterns=['bart/*'])"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --chown=appuser tos_summarizer/ ./tos_summarizer/
COPY --chown=appuser gpu_worker/ ./gpu_worker/

CMD ["python", "-m", "gpu_worker.worker"]
