FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/share/nltk_data \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# CPU-only torch first (the default wheel bundles ~2GB of CUDA libs we can't use here)
COPY requirements-app.txt .
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-app.txt

# Bake the NLTK data in so nothing gets downloaded at runtime
RUN python -m nltk.downloader -d $NLTK_DATA punkt_tab averaged_perceptron_tagger_eng

# Compressed models (run scripts/export_models.py first). Copied before the code
# so code changes don't invalidate this ~1GB layer
COPY --chmod=755 models/ ./models/

COPY tos_summarizer/ ./tos_summarizer/
COPY app.py fb_tos_testing.txt ./

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
