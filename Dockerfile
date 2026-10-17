# Images for the clause classifier. Two targets share one base:
#
#   api    (default, last stage)  CPU serving image: API, playground, model pages. Runtime dependencies only.
#   tools                         every `make` step (data, training, evaluation, GIFs, pages, tests) with all
#                                 dependency groups from uv.lock. The project folder is mounted at /work/<folder>, so code,
#                                 configs, data and artifacts come from the host and outputs land on the host.
#
#   docker build -t contract-clause-classifier .                       # api
#   docker build --target tools -t contract-clause-classifier-tools .  # tools (normally via `make docker-build`)
#
# Docker on macOS cannot use the Apple GPU, so both images run on CPU there. For an NVIDIA GPU on Linux, build the
# tools image with TORCH_INDEX=https://download.pytorch.org/whl/cu130 (the `tools-gpu` service in docker-compose.yml).
# The api image serves every trained version under /app/artifacts/runs (default = latest transformer) and the
# classical models under /app/artifacts/classical/runs.

ARG PYTHON_IMAGE=python:3.12.15-slim-bookworm

# ---------------------------------------------------------------- base: Python + torch (same pinned version as uv.lock)
FROM ${PYTHON_IMAGE} AS base

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false

# CPU wheel by default (~200 MB instead of multi-GB CUDA wheels).
RUN pip install --index-url ${TORCH_INDEX} torch==2.14.1

# ---------------------------------------------------------------- tools: every pipeline step, run by `make`
FROM base AS tools

# DejaVu fonts for the terminal replay GIF (Menlo exists only on macOS); make for the Makefile tests.
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core make && rm -rf /var/lib/apt/lists/*

# All dependency groups (train, viz, dev) at the exact versions in uv.lock. torch is already installed above, and the
# CUDA packages the lock lists for Linux are skipped so the CPU image stays small.
COPY pyproject.toml uv.lock /tmp/project/
RUN pip install uv==0.11.21 && \
    cd /tmp/project && \
    uv export --frozen --all-groups --no-emit-project --no-hashes \
      | grep -viE '^(torch|triton|nvidia-|cuda-)' > /tmp/requirements.txt && \
    uv pip install --system --no-deps -r /tmp/requirements.txt && \
    rm -rf /tmp/project /tmp/requirements.txt /root/.cache

# The project itself is mounted at /work/<folder name> (not installed; docker-compose.yml sets PYTHONPATH and the working
# directory), so edits on the host apply without a rebuild.
# The container runs as the host user (see docker-compose.yml), so HOME and caches must be writable by any uid;
# the Hugging Face cache is a named volume initialised from /cache.
ENV HOME=/tmp \
    HF_HOME=/cache/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    TERM=xterm-256color
RUN mkdir -p /cache/huggingface && chmod -R 1777 /cache
COPY docker/tools-entrypoint.sh /usr/local/bin/tools-entrypoint
RUN chmod 0755 /usr/local/bin/tools-entrypoint
WORKDIR /work
ENTRYPOINT ["tools-entrypoint"]
CMD ["python", "scripts/07_models.py", "list"]

# ---------------------------------------------------------------- api: CPU serving image (default target)
FROM base AS api

ENV HF_HOME=/app/.cache/huggingface \
    RUNS_DIR=/app/artifacts/runs \
    CLASSICAL_RUNS_DIR=/app/artifacts/classical/runs \
    PAGES_DIR=/app/docs/models \
    LABEL_GROUPS=/app/configs/ledgar_label_groups.yaml \
    DATA_DIR=/app/data \
    DEVICE=cpu

WORKDIR /app

# Runtime deps only (training extras peft/accelerate are not needed to serve a merged model).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs/ledgar_label_groups.yaml ./configs/ledgar_label_groups.yaml
# Raw LEDGAR data (28 MB) so /v1/dataset works without a volume mount.
COPY data/raw/ledgar ./data/raw/ledgar
# Generated model pages (served at /models/*) and the changelog page.
COPY docs/models ./docs/models
COPY changelog.html ./changelog.html
RUN pip install . && \
    useradd --create-home --uid 1000 app && \
    mkdir -p /app/artifacts /app/data /app/.cache && chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "classification.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
