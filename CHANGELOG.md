# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-10-11

First complete version: data, training, telemetry, evaluation, inference, API and Docker.

### Added

- **Data**: LEDGAR download from the LexGLUE parquet export with a datasets-server fallback, label map extraction, tokenisation cache, length-grouped batch sampler (`scripts/01_prepare_data.py`).
- **Model**: Qwen3-0.6B-Base sequence classifier with LoRA adapters and a trainable 100-way head; merge-and-export to a standalone serving model.
- **Training**: device-agnostic PyTorch loop (CUDA, MPS, CPU) with AdamW, warm-up plus cosine schedule, gradient accumulation and clipping, periodic evaluation, best and last checkpoints, early stopping and `--resume` (`scripts/02_train.py`).
- **Live terminal dashboard** during training: progress, loss trend, learning rate, gradient norm, throughput, memory, CPU and RAM, current batch clause types, validation history.
- **Training telemetry** in `artifacts/runs/<run>/telemetry/`: per-step metrics with batch contents and per-example losses, per-layer gradient norms, per-eval metrics with per-class F1, confusion matrices and calibration bins, probe-set snapshots of predictions, embeddings and per-layer vectors, lifecycle events, and system monitoring. Schema in `docs/TELEMETRY.md`.
- **GIF renderer** for telemetry (`scripts/06_render_gif.py`); sanity-run GIF in `demo/`.
- **Evaluation**: full test metrics, per-class report, top confusions, temperature-scaling calibration and expected calibration error (`scripts/03_evaluate.py`).
- **Inference**: contract segmentation into provisions, `.txt` / `.pdf` / `.docx` reading, per-provision top-k and document-level aggregation (`scripts/04_predict.py`).
- **API** (FastAPI, OpenAI-compatible): `GET /health`, `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/chat/completions`, `GET /v1/dataset` (10 rows per page by default), `GET /v1/labels`; OpenAI error envelope; optional bearer-token auth; CORS (`scripts/05_serve.py`).
- **Docker**: CPU serving image on `python:3.12.15-slim-bookworm` with a non-root user and `HEALTHCHECK`; `docker-compose.yml`.
- **Makefile** with `setup`, `data`, `train`, `resume`, `sanity`, `evaluate`, `gif`, `predict`, `serve`, `test`, `docker-*` and `all`.
- **Tests**: 28 pytest tests (segmentation, data, telemetry metadata, prediction format, API routes and errors, auth, OpenAPI, OpenAI SDK compatibility) using a stub model.
- **Docs and agent files**: README, `docs/TELEMETRY.md`, `plans/PLAN.md`, `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/`, `MEMORY.md`, this changelog.

### Pinned

- Python 3.12; torch 2.14.1, transformers 5.19.0, peft 0.21.2, accelerate 1.15.0, huggingface_hub 2.2.0, safetensors 0.8.0, numpy 2.5.3, pyarrow 26.0.0, scikit-learn 1.9.1, fastapi 0.143.0, uvicorn 0.54.0, pydantic 2.14.0, pymupdf 1.28.2, python-docx 1.2.0, rich 15.0.0, typer 0.27.3, pyyaml 6.0.3, psutil 7.2.2, matplotlib 3.11.2, pillow 12.3.0, pytest 9.1.1, httpx 0.28.1, openai 3.28.0.

### Defaults chosen from measurements

- Micro-batch 4 with 8 accumulation steps (effective 32): keeps MPS memory near 20 GB instead of 35 GB on a 48 GB Mac.
- `precision: auto` uses fp32 on MPS (bf16 autocast measured no faster).
- Probe snapshots use length-sorted batches (5x faster).
