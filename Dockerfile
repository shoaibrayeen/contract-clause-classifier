# Serving image for the clause classifier API (CPU).
# Docker on macOS cannot use the Apple GPU, so this image runs on CPU everywhere.
# For GPU speed on a Mac, run natively: `uv run python scripts/05_serve.py`.
#
#   docker build -t contract-clause-classifier .
#   docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" -v "$PWD/data:/app/data" contract-clause-classifier

FROM python:3.12.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    MODEL_DIR=/app/artifacts/best \
    DATA_DIR=/app/data \
    DEVICE=cpu \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only torch wheel first (~200 MB instead of multi-GB CUDA wheels); same pinned version as pyproject.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1

# Runtime deps only (training extras peft/accelerate/scikit-learn are not needed to serve a merged model).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . && \
    useradd --create-home --uid 1000 app && \
    mkdir -p /app/artifacts /app/data /app/.cache && chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "classification.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
