# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/). A styled copy of this file is at [changelog.html](changelog.html); the two are kept in sync (see the Changelog rules in AGENTS.md).

## [Unreleased]

Model versioning, accuracy-focused training defaults, a browser playground, and rules for every AI assistant.

### Added

- **Model versions.** Every training run creates `<train.run_name>-v<model_version>` (first: `ledgar-qwen3-0.6b-v1`) and no model is ever overwritten. `model_version` in `configs/ledgar_qwen3_0.6b.yaml` starts at 1 and `make all` raises it after the run, evaluation and GIF succeed. Unfinished versions are never replaced silently: `make resume` continues one, `make restart` discards and retrains it.
- **Multi-version serving.** The API serves every trained version. Requests without `model` (or with `latest`) use the newest; a version id such as `ledgar-qwen3-0.6b-v1` selects that version; a base name selects its latest version. The response's `model` field names the version that answered. New versions appear without restarting the server; up to `MAX_LOADED_MODELS` (default 2) stay in memory.
- **Playground usage GIF** (`demo/playground_usage.gif`), recorded from a real session with `ledgar-qwen3-0.6b-v1` and shown in the README. `make playground-gif` (`scripts/08_record_playground.py`) re-records it: it drives headless Chrome through the playground, captions each step, and starts the API if needed.
- **Playground** at `http://localhost:8000/playground` (`/` redirects there): paste text or load a sample, pick a version, adjust top-k and threshold, and see clause types with confidence bars and per-clause predictions.
- **Label grouping** (`configs/ledgar_label_groups.yaml`, `data.label_groups`): 41 LEDGAR labels that denote the same legal concept merged into 19 clause types, so 100 labels become 78. Every merge passed a legal test on the actual provisions and a statistical test on an independent TF-IDF baseline over the full validation split; five candidate merges were rejected (Miscellaneous/General, Remedies/Specific Performance/Enforcements, Taxes vs Withholdings, Fees vs Expenses, Venue vs Jurisdiction). On the first run's checkpoint this moves validation accuracy from 83.0% to 88.9%; the TF-IDF baseline from 87.8% to 91.2%.
- `scripts/07_models.py` and `make models`, `make next`, `make bump`, `make adopt RUN=...`, `make restart`, `make finish`, `make rules`.
- `merged/model_info.json` per version (id, version, base model, method, label count, label groups, best validation and test scores), shown by `make models` and `/v1/models`.
- `label_space.json` saved with every checkpoint and merged model (label names, the dataset's original labels and their mapping, groups used). `make evaluate` scores each model against its own label space, so older versions stay evaluable after the groups file changes; a model without the file is accepted only when its labels match the current groups or the original 100.
- **The LEDGAR dataset is committed** (`data/raw/ledgar/`, 28 MB, CC BY 4.0) so every clone trains and evaluates on identical rows. `data/raw/ledgar/README.md` credits the sources with the exact download URL, date and SHA-256 checksum of every file plus citations; `data/raw/README.md` sets the policy for adding data. The Docker image includes the data, so `/v1/dataset` works without a volume.
- Top-3 accuracy in training evaluations, the test report, `make models` and `/v1/models`.
- `/v1/dataset` rows include `original_label` (LEDGAR's heading) next to `label` (the clause type the model predicts).
- `MODEL=` on `make evaluate`, `make gif` and `make predict`; `--model` on `03_evaluate.py` and `04_predict.py`; `--run latest|<version>` and `--copy-to` on `06_render_gif.py`.
- Training settings `train.label_smoothing` and `train.best_metric`.
- Rules for all AI assistants: core-rules summary plus Model, Changelog, Security and privacy, Git, Data and Documentation rules in `AGENTS.md`; generated rule files for GitHub Copilot, Gemini, Windsurf, Cline, Roo Code, Continue, Amazon Q, JetBrains Junie, Zed and Aider (`make rules`), kept in sync by a test.
- Tests: 74 in total (versioning, multi-version API, label grouping, dashboard fit, playground, rule-file sync).

### Changed

- **Breaking: default training is full fine-tuning** (all 596M weights, lr 2e-5, 6% warm-up, 3 epochs, label smoothing 0.1, best checkpoint by accuracy) on the 78 merged clause types. Verified end to end on the real model: about 17% slower than LoRA on an M4 Pro, 25 GB peak memory, about 9 hours for 3 epochs, about 10 GB of disk per version. Set `full_finetune: false` for LoRA (now r=64).
- **Breaking: models predict 78 clause types instead of 100.** Merged names such as `Representations and Warranties` replace their parts. Set `data.label_groups: null` for the original labels. Versions trained before and after this change are not comparable.
- **Breaking: run base name is now `ledgar-qwen3-0.6b`** (was `ledgar-qwen3-0.6b-lora`); runs are saved as `ledgar-qwen3-0.6b-v<N>`.
- **Breaking: API settings.** `RUNS_DIR` (default `artifacts/runs`) replaces `MODEL_DIR` as the model source; `MODEL_DIR` is now an optional extra folder. `MODEL_ID` (`clause-classifier-qwen3-0.6b`) is an alias for the latest version. `/health` adds `latest_model`, `latest_version`, `available_models`, `loaded_models` and renames `model_dir` to `models_dir`.
- Input is split into provisions by default in the API, CLI and playground (`segment: false` treats it as one clause). Previously only inputs over 350 words were split.
- `artifacts/best` always points at the latest version rather than the last run.
- Live dashboard: 10 rows of current-batch clause types and validation history, a compact stats block, top-3 accuracy column, and project-relative paths (`contract-clause-classifier/artifacts/runs/...`).
- `build_model` re-initialises a mismatched classification head only when the base model already has one; for a causal-LM base any weight mismatch still raises.
- Config comments now carry the measured evidence for every training setting (loss trend, clipping, hard-example lengths, evaluation time, memory, disk).
- `.gitignore` allows `data/raw/ledgar/` and keeps `data/cache/` ignored.
- Console and API output show project-relative paths instead of `/Users/...`.
- Docker image serves all versions (`RUNS_DIR=/app/artifacts/runs`) and includes the label-group file.

### Measured

- **First fully trained model, `ledgar-qwen3-0.6b-v1`:** 92.46% accuracy, 97.69% top-3 accuracy and 89.7% macro-F1 on the full 10,000-row LEDGAR test split (78 clause types); calibration error 2.2% after temperature scaling (7.0% before). Validation accuracy plateaued at 92.2% from step 4,000 of 5,625.
- README sample responses now come from v1, and the README documents the commercial clause types LEDGAR does not cover (Limitation of Liability, Force Majeure, Non-Compete and others), which the model can only map to the nearest known type.

### Fixed

- The live dashboard re-printed itself in short terminals (the panel was taller than the window). It now sizes itself to the terminal.
- Short multi-section contracts were classified as a single clause.
- Starting training from a model that already had a classification head of another size failed.
- `make resume` on a run that died before its first checkpoint would have restarted it in place and appended duplicate telemetry; it now stops and points to `make restart`.
- Evaluating a model trained with a different label grouping than the current config crashed (`Target 73 is out of bounds`); it now uses the model's saved label space.

## [0.1.0] - 2026-10-11

First complete version: data, training with telemetry, evaluation, inference, OpenAI-compatible API, Docker, documentation, licensing and agent guidance. Built on an Apple M4 Pro (48 GB); every step was verified end to end except the multi-hour full training run, which is left to the owner (`make all`).

### Added

#### Data (`src/classification/data.py`, `scripts/01_prepare_data.py`)

- Download of the LEDGAR split of LexGLUE from the Hugging Face parquet export (`coastalcph/lex_glue`, config `ledgar`, revision `refs/convert/parquet`), with an automatic fallback to the `lighteval/lexglue` mirror through the datasets-server API.
- Extraction of the 100 clause-type names from the parquet schema metadata into `data/raw/ledgar/labels.json`.
- Tokenisation cache per tokenizer and `max_len` (parquet, `list<int32>`), so repeated runs skip tokenisation.
- Length-grouped batch sampler (shuffle, sort within mega-batches, cut) that cuts padding to about 2%, deterministic per seed and epoch so `--resume` sees the same order.
- Collate function that carries the source row id of every example, so each training batch can be traced back to the dataset.
- `DatasetStore`, a lazy memory-mapped reader used by the paginated `/v1/dataset` endpoint.
- Token-length statistics printed after preparation (mean 138, median 105, p90 292, 1.9% truncated at 512).

#### Model (`src/classification/model.py`)

- Qwen3-0.6B-Base loaded as `AutoModelForSequenceClassification` with a 100-way head and the pad token set.
- LoRA via PEFT on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (r=16, alpha=32, dropout 0.05), head fully trainable; 10.2M of 606M parameters train. `full_finetune: true` trains everything.
- Checkpoint save and load for both adapter and full-weight checkpoints; adapter loading rebuilds the base with the right label count first (PEFT adapter directories carry no base config).
- Merge-and-export (`merge_and_unload`) to a standalone bf16 safetensors model with `labels.json` and `calibration.json`, and a permissions fix so exported files are world-readable (safetensors are written 0600, which broke Docker's non-root user).
- Temperature calibration file helpers.

#### Training (`src/classification/train.py`, `scripts/02_train.py`)

- Plain PyTorch loop, no `Trainer`: AdamW with decoupled weight decay groups, 3% linear warm-up then cosine decay, gradient accumulation, gradient clipping, mixed precision through autocast.
- Device auto-detection `cuda > mps > cpu`; precision `auto` is bf16 on CUDA (fp16 when bf16 is unsupported) and fp32 on MPS and CPU.
- Evaluation every 500 optimizer steps on a fixed 3,000-example validation subset; `checkpoint-best/` kept by macro-F1; `checkpoint-last/` with optimizer and scheduler state; early stopping after 4 evaluations without improvement.
- `--resume` continues from `checkpoint-last/`, skipping already-consumed micro-batches.
- Periodic accelerator cache release (`train.empty_cache_steps`, default 25).
- At the end: best checkpoint merged to `merged/` and the `artifacts/best` symlink updated.
- CLI overrides for run name, epochs, limit, eval steps and limit, batch size, accumulation, learning rate, device, precision and backbone.
- Noisy HTTP loggers silenced.

#### Live dashboard (`src/classification/dashboard.py`)

- Rich live panel refreshed twice a second: progress bar with ETA, loss and smoothed loss with a sparkline, learning rate, gradient norm and clipping flag, batch accuracy, tokens per second, step time, padding ratio, accelerator memory, CPU and RAM, the clause types in the current batch, and the last five validation results.
- Plain one-line log every 10 steps when output is not a terminal.

#### Telemetry (`src/classification/telemetry.py`, `docs/TELEMETRY.md`)

- `RunRecorder` writes append-only, line-buffered files under `artifacts/runs/<run>/telemetry/`, safe across crashes and resumes.
- `meta.json`: config, environment (Python, torch, transformers, peft, platform, chip, memory, GPU), model shape, dataset statistics (label counts, length histograms), probe-set ids, sessions, final summary.
- `steps.jsonl`: per optimizer step: loss, learning rate, gradient norm, clipping, step time, tokens, padding ratio, throughput, memory, plus the batch's example ids, labels, lengths, per-example losses and correctness.
- `layers.jsonl`: gradient norm per transformer layer and head every 10 steps.
- `evals.jsonl`: per evaluation: accuracy, micro and macro-F1, loss, duration, per-class F1 for all 100 types, trainable weight norm per layer.
- `evals/step_*.npz`: confusion matrix and calibration bins per evaluation.
- `snapshots/step_*.npz`: for a fixed stratified probe set (5 per class, about 500) at step 0 and every evaluation: probabilities, final-layer embeddings, activation norm at each of the 29 layers, and full per-layer vectors for a 1-per-class subset.
- `events.jsonl`: start, snapshot, eval, checkpoint, early stop, done.
- `system.jsonl`: CPU, RAM, process memory, accelerator memory, phase, sampled every 5 seconds by a background thread.
- Probe forward pass uses length-sorted batches (24 s instead of 120 s for the probe set).

#### Evaluation (`src/classification/evaluate.py`, `scripts/03_evaluate.py`)

- Full test-split report: accuracy, micro, macro and weighted F1, negative log-likelihood, expected calibration error before and after calibration, per-class precision, recall and F1, top confusions.
- Temperature scaling fitted on validation with L-BFGS and saved next to the model; the API applies it automatically.

#### Inference (`segment.py`, `io_utils.py`, `predict.py`, `scripts/04_predict.py`)

- Contract segmentation: paragraph splitting, heading gluing, numbered-section detection (`1.`, `1.1`, `Article IV`, `Section 5.2`), sub-item grouping, long-chunk splitting on sentence boundaries, merging of tiny unnumbered fragments.
- Readers for `.txt`, `.md`, `.pdf` (PyMuPDF, text layer required) and `.docx` (python-docx, including tables).
- `Predictor` with a pluggable backend: per-provision top-k with threshold (top-1 always kept) and document-level aggregation by maximum confidence.
- CLI for files or `--text`, with `--top-k`, `--threshold`, `--no-segment`, `--out`.

#### API (`src/classification/api/`, `scripts/05_serve.py`)

- FastAPI app factory with lifespan model loading; the app keeps serving `/health` and the dataset routes if the model fails to load.
- `GET /health` (200 or 503 with a reason), `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/chat/completions`, `GET /v1/dataset` (10 rows per page by default, up to 100), `GET /v1/labels`.
- Chat completions accept the OpenAI body (string or content-part messages), return a standard `chat.completion` object with real token usage, and reject `stream=true` and `n>1` with 400. Extra knobs `top_k`, `threshold`, `segment`.
- OpenAI-style error envelope for all errors, including validation errors.
- Optional bearer-token authentication (`API_KEY`), CORS, input size limit, settings from environment variables (`.env.example`).
- Verified with the official `openai` Python SDK.

#### Docker

- `Dockerfile` on `python:3.12.15-slim-bookworm` with the CPU-only torch wheel, runtime dependencies only, non-root user, `HEALTHCHECK` on `/health`. Image about 1.9 GB.
- `docker-compose.yml` mounting `./artifacts` and `./data`.
- `.dockerignore`.

#### Visualisation (`scripts/06_render_gif.py`, `demo/`)

- Renders a run's telemetry into an animated GIF with four panels: training loss, validation accuracy and macro-F1, per-layer gradient heatmap, and PCA of probe embeddings moving between snapshots. Works mid-training.
- `demo/sanity_training.gif` and `demo/sanity_training_final_frame.png` from the 5-minute sanity run; `demo/sample_contract.txt`, a 12-section master services agreement.

#### Tooling

- `Makefile`: `setup`, `data`, `train`, `resume`, `sanity`, `evaluate`, `gif`, `predict`, `serve`, `test`, `docker-build`, `docker-up`, `docker-down`, `all`, `clean-runs`, `help`.
- `pyproject.toml` with exact pins, dependency groups `train`, `viz`, `dev`, and `uv.lock`; `.python-version` = 3.12.
- `.gitignore` for data, artifacts, model files, environments and editor state.

#### Tests (`tests/`)

- 28 pytest tests using a keyword stub model: segmentation (5), data paging and batching (4), telemetry sessions (1), prediction format (4), API routes, errors, auth, OpenAPI schema and OpenAI SDK compatibility (14). No downloads or GPU needed; about 3 seconds.

#### Documentation

- `README.md`: how it works, training data, requirements, setup, pipeline, training, telemetry, evaluation, CLI, API with sample requests and responses for every endpoint, Docker, layout, configuration, testing, troubleshooting, pinned versions, license, author.
- `docs/TELEMETRY.md`: schema of every telemetry file with shapes and intended uses.
- `plans/PLAN.md`: approved design plan plus an implementation-status table of deviations.
- `CHANGELOG.md` and `changelog.html`.

#### Agent guidance

- `AGENTS.md` as the single source of truth (commands, layout, 12 rules, style, gotchas) with a map of how Codex, Claude Code and Cursor read it.
- `CLAUDE.md` importing `AGENTS.md` with Claude-specific notes; `.cursor/rules/project.mdc` (always on) and `.cursor/rules/python-style.mdc`; `.codex/README.md`.
- Rule 11: always update `MEMORY.md` after completing a task. Rule 12: a task is complete only when tests pass and docs, changelog and memory are updated.
- `MEMORY.md`: owner preferences, design decisions, measurements and gotchas, dated.

#### Licensing

- Apache License 2.0 (`LICENSE`, copyright 2026 Shoaib Rayeen), chosen to match Qwen3-0.6B-Base and the Hugging Face stack and to include a patent grant. License, author, keywords and URLs added to `pyproject.toml`.

### Changed during the build

- Package renamed from `clauseclf` to `classification`; project folder renamed from `domain_models` to `contract-clause-classifier`.
- Framework switched from MLX (first plan) to PyTorch so training is machine-independent.
- Micro-batch reduced from 8 to 4 with accumulation raised from 4 to 8 (effective batch 32 unchanged) after batch 8 drove MPS memory to about 35 GB and swapping on a 48 GB Mac.
- Epochs reduced from 3 to 2 (about 2.3 hours per epoch on the M4 Pro).
- In-training evaluation limited to a fixed 3,000-example validation subset to keep each evaluation near a minute.
- `precision: auto` uses fp32 on MPS after bf16 autocast measured no faster.

### Fixed during the build

- Progress bar crashed on evaluation (`KeyError: 'status'`).
- Adapter checkpoints failed to load through `AutoPeftModelForSequenceClassification` (2-label head); fixed by rebuilding the base with the right label count.
- Short numbered sections were merged into neighbouring provisions by the segmenter.
- Exported safetensors were unreadable by Docker's non-root user (0600).
- `system_fingerprint` showed the folder name `merged` instead of the run name.
- Duplicate session entries in `meta.json` on every metadata update.

### Pinned

- Python 3.12; torch 2.14.1, transformers 5.19.0, peft 0.21.2, accelerate 1.15.0, huggingface_hub 2.2.0, safetensors 0.8.0, numpy 2.5.3, pyarrow 26.0.0, scikit-learn 1.9.1, fastapi 0.143.0, uvicorn 0.54.0, pydantic 2.14.0, pymupdf 1.28.2, python-docx 1.2.0, rich 15.0.0, typer 0.27.3, pyyaml 6.0.3, psutil 7.2.2, matplotlib 3.11.2, pillow 12.3.0, pytest 9.1.1, httpx 0.28.1, openai 3.28.0; Docker base `python:3.12.15-slim-bookworm`.

### Measured (Apple M4 Pro, 48 GB, MPS)

- About 990 tokens per second, about 4.4 seconds per optimizer step of 32 examples; 2 epochs about 5 hours.
- Micro-batch 4: about 19.5 GB accelerator memory. Micro-batch 8: about 35 GB and swap exhaustion.
- Sanity run (256 examples, 128 steps): validation accuracy 0.8% to 20%, proving the loop learns.

[Unreleased]: https://github.com/shoaibrayeen/contract-clause-classifier/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shoaibrayeen/contract-clause-classifier/releases/tag/v0.1.0
