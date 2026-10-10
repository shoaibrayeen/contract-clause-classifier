# Plan: Contract Clause Classifier (0.6B) — portable training, FastAPI + Docker serving

## Context

Goal: fine-tune a ~0.5–0.8B model so that, given a contract, it returns the clause types detected as `[{name, confidence}, ...]`. Requirements gathered:

- **Training must be machine-independent** → PyTorch + Hugging Face `transformers` + `peft`, device auto-detect (`cuda` > `mps` > `cpu`). First training runs happen on this M4 Pro (48 GB) via MPS; the same code runs unchanged on a CUDA box.
- **The laptop is the inference target** → the trained model is served locally with FastAPI.
- **API must be OpenAI-compatible** → `POST /v1/chat/completions` (+ `GET /v1/models`), assistant content is a JSON string with the predictions.
- **Health check** → `GET /health`.
- **Dataset API, paginated** → `GET /v1/dataset?split=train&offset=0&limit=10`.
- **Docker** for the API.
- **Not CUAD.** Dataset: **LEDGAR** (LexGLUE): 80k real SEC-contract provisions, 100 clause-type labels (Governing Laws, Confidentiality, Indemnifications, Non-Competes, Terminations, Arbitration, …). Plain parquet on HF (`coastalcph/lex_glue`, config `ledgar`, revision `refs/convert/parquet`; mirror `lighteval/lexglue`), ~28 MB. One label per provision; multi-label output per contract comes from classifying each provision and aggregating. Verified: files, 100 label names, sample rows.
- **Model: `Qwen/Qwen3-0.6B-Base`** loaded with `AutoModelForSequenceClassification` (transformers ships `Qwen3ForSequenceClassification`, last-token pooling built in). LoRA via PEFT (`task_type=SEQ_CLS`, `modules_to_save=["score"]`). Config swap to `Qwen/Qwen2.5-0.5B` or `HuggingFaceTB/SmolLM2-360M` is one line.

MLX (chosen earlier) is dropped because it is Mac-only and would make training machine-dependent, which conflicts with the later requirement. Optional later: export to MLX/ONNX for faster laptop inference.

### Requirements checklist
- [x] Fine-tune a 0.5–0.8B model → Qwen3-0.6B-Base + LoRA
- [x] Dataset other than CUAD, found via web search → LEDGAR (100 clause types)
- [x] Input a contract, output `[{name, confidence}]` → per-clause + document-level
- [x] Runs on this M4 Pro → native MPS serving; CPU Docker
- [x] Training not tied to a machine → PyTorch, device auto-detect
- [x] OpenAI-compatible API → `/v1/chat/completions`, `/v1/models`
- [x] Health check → `/health`
- [x] Dataset API, 10 per page → `/v1/dataset` (default `limit=10`)
- [x] Docker → Dockerfile + docker-compose
- [x] FastAPI
- [x] All code in `/Users/shoaib.rayeen/my_space/contract-clause-classifier`

## Architecture

```
contract (.txt/.pdf/.docx or chat message text)
   │  segment.py: split into provisions (headings / numbering / blank lines)
   ▼
[provision_1 ... provision_n]
   │  Qwen3-0.6B + LoRA + 100-way score head (softmax, temperature-scaled)
   ▼
per-provision top-k [{name, confidence}]
   │  aggregate: max confidence per label across provisions, threshold
   ▼
{"document":[{"name","confidence"}...], "clauses":[{"index","text","predictions":[...]}...]}
```

## Project layout (`/Users/shoaib.rayeen/my_space/contract-clause-classifier`)

```
contract-clause-classifier/
├── pyproject.toml              # uv, Python 3.12; deps: torch, transformers, peft, accelerate, datasets, pyarrow,
│                               # scikit-learn, numpy, pyyaml, typer, rich, fastapi, uvicorn, pydantic, pymupdf, python-docx, pytest, httpx
├── README.md
├── .env.example                # MODEL_DIR, DATA_DIR, DEVICE, THRESHOLD, TOP_K
├── .gitignore / .dockerignore  # data/, artifacts/
├── Dockerfile                  # python:3.12-slim, CPU torch wheel, copies src + artifacts/best
├── docker-compose.yml          # api service, port 8000, volumes for artifacts/ and data/
├── configs/
│   └── ledgar_qwen3_0.6b.yaml  # model, max_len, lora (r/alpha/dropout/targets), lr, batch, accum, epochs, eval_steps, device: auto
├── data/                       # downloaded parquet + tokenized cache (gitignored)
├── artifacts/                  # runs/<name>/ (adapter, score head, metrics.jsonl, calibration.json); best/ symlink
├── src/classification/
│   ├── __init__.py
│   ├── config.py               # pydantic/yaml config loading, device resolution (cuda>mps>cpu), dtype choice
│   ├── labels.py               # 100-label map id↔name, saved to artifacts/labels.json
│   ├── data.py                 # download LEDGAR parquet, HF Dataset, tokenize (max_len 512), length-grouped sampler, class weights
│   ├── model.py                # build AutoModelForSequenceClassification + PEFT LoRA; load_for_inference(merge adapters)
│   ├── train.py                # training loop (plain PyTorch, no Trainer): AdamW, warmup+cosine, grad-accum, clip, eval, early stop, resume
│   ├── evaluate.py             # accuracy, micro/macro F1, per-class report, top confusions, ECE + temperature scaling
│   ├── segment.py              # contract text → provisions
│   ├── io_utils.py             # read .txt/.pdf (PyMuPDF)/.docx (python-docx)
│   ├── predict.py              # Predictor class: provisions → [{name, confidence}], document aggregation
│   └── api/
│       ├── app.py              # FastAPI factory, lifespan loads Predictor once, CORS
│       ├── schemas.py          # OpenAI chat request/response models, dataset page model, health model
│       └── routes/
│           ├── health.py       # GET /health
│           ├── openai.py       # GET /v1/models, POST /v1/chat/completions
│           └── dataset.py      # GET /v1/dataset
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_train.py             # --config, --limit, --epochs, --device overrides
│   ├── 03_evaluate.py
│   ├── 04_predict.py           # CLI: file or --text → JSON
│   └── 05_serve.py             # uvicorn classification.api.app:create_app --port 8000
└── tests/
    ├── test_segment.py
    ├── test_data.py            # label map has 100 names, tokenization shapes
    ├── test_predict_format.py  # output schema (uses a tiny stub model)
    └── test_api.py             # TestClient: /health, /v1/models, /v1/chat/completions, /v1/dataset paging
```

## Steps

### Step 0 — Environment
- `uv init`, Python 3.12 (already installed via uv; system Python 3.9 is too old).
- Smoke test prints the resolved device (`mps` on this laptop) and runs a 1-step forward on Qwen3-0.6B-Base.
- MPS notes baked into `config.py`: fp32 master weights on MPS (bf16 optional flag), `PYTORCH_ENABLE_MPS_FALLBACK=1`, `pin_memory=False`.

### Step 1 — Data (`data.py`, `01_prepare_data.py`)
- Download the three LEDGAR parquet files; save `artifacts/labels.json` (100 names).
- Tokenize with the Qwen3 tokenizer, `max_len=512` (log token-length histogram; most provisions < 300).
- Cache to `data/cache/ledgar_qwen3/` with `datasets.save_to_disk`.
- Class frequencies → optional weighted cross-entropy (config flag, default off).

### Step 2 — Model (`model.py`)
- `AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen3-0.6B-Base", num_labels=100)`; set `pad_token_id` (Qwen uses `<|endoftext|>`).
- PEFT LoRA: r=16, alpha=32, dropout=0.05, targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, `modules_to_save=["score"]`. ~1.5% trainable params.
- `full_finetune: true` flag unfreezes everything (fits on 48 GB; ~0.6B × 16 bytes fp32+Adam ≈ 10 GB + activations).
- `load_for_inference()` merges the adapter (`merge_and_unload`) and saves a plain HF checkpoint to `artifacts/best/` so the API and Docker image need no PEFT at runtime.

### Step 3 — Training (`train.py`, `02_train.py`)
- Plain PyTorch loop (readable, no `Trainer` magic): AdamW lr 2e-4 (LoRA) / 2e-5 (full), 3% warmup + cosine, wd 0.01, clip 1.0, batch 8 × accum 4 (eff. 32), length-grouped batching.
- Eval on 10k val every 500 steps: macro-F1 early stopping, best checkpoint kept, `metrics.jsonl` + Rich progress; `--resume`.
- Expected on M4 Pro MPS, LoRA, max_len 512: roughly 3–4 h/epoch; plan 2–3 epochs (run overnight). On a CUDA GPU: ~20–30 min/epoch.
- Sanity gate first: `--limit 512 --epochs 3` must drive loss near zero.

### Step 4 — Evaluation (`evaluate.py`, `03_evaluate.py`)
- Test split: accuracy, micro-F1, macro-F1, per-class P/R, top-20 confusions → `artifacts/runs/<name>/eval.json`.
- Reference (LexGLUE paper, BERT-class encoders): micro-F1 ≈ 87–88, macro ≈ 81–83. Target: micro-F1 ≥ 85.
- Calibration: fit temperature on val, store `calibration.json`; applied in `predict.py` so `confidence` is meaningful.

### Step 5 — Inference (`segment.py`, `predict.py`, `04_predict.py`)
- Segmentation: headings/numbering (`1.`, `1.1`, `ARTICLE IV`, `Section 5.2`, ALL-CAPS lines) + blank lines; merge fragments < 25 words; split > 400 tokens on sentences.
- Per provision: top-k (default 3) with temperature-scaled softmax; keep ≥ threshold (0.30), always keep top-1.
- Document: max per label across provisions, sorted, thresholded.
- Batched inference (batch 16) on the resolved device; `torch.inference_mode()`.

### Step 6 — FastAPI (`api/`, `05_serve.py`)
- `GET /health` → `{"status":"ok","model":"<id>","device":"mps","labels":100,"uptime_s":…}`; returns 503 until the model is loaded.
- `GET /v1/models` → OpenAI list with one model id `clause-classifier-qwen3-0.6b`.
- `POST /v1/chat/completions` → accepts the OpenAI body (`model`, `messages`, optional `temperature` ignored, `stream` → 400 not supported). Contract text = last user message content (string or text parts). Response is a standard `chat.completion` object; `choices[0].message.content` is the JSON string `{"document":[...],"clauses":[...]}`; `usage` filled with real token counts. Optional extra: if the body has `response_format: {"type":"json_object"}` behaviour is identical. Works with the official `openai` Python/JS SDK by setting `base_url`.
- `GET /v1/dataset?split=train|validation|test&offset=0&limit=10` → `{"split","offset","limit","total","next_offset","items":[{"id","text","label_id","label"}]}`; **`limit` defaults to 10**, capped at 100; `offset` defaults to 0; 404 for bad split; 422 for bad ints. Reads the cached parquet via pyarrow (memory-mapped). If the data has not been prepared yet, the endpoint downloads the parquet on first call.
- Lifespan loads `Predictor` once from `MODEL_DIR` (default `artifacts/best`). Config via env vars (`.env.example`).

### Step 7 — Docker
- `Dockerfile`: `python:3.12-slim`, install CPU-only torch wheel (keeps image ~1.5 GB), copy `src/`, `artifacts/best/` (or mount), `uvicorn` entrypoint, `HEALTHCHECK` hitting `/health`.
- `docker-compose.yml`: `api` on `8000`, volumes `./artifacts:/app/artifacts`, `./data:/app/data`, env `DEVICE=cpu`.
- Caveat stated in README: Docker on macOS cannot use the Apple GPU, so the container runs on CPU (fine for classification, ~0.3–0.6 s per provision). For GPU-speed on the laptop run `uv run scripts/05_serve.py` natively (MPS). On Linux with NVIDIA, swap base image to a CUDA torch wheel.

## Decisions / assumptions
- Base (not instruct) Qwen3-0.6B; single-label CE on LEDGAR; multi-label at inference via top-k + threshold + doc aggregation. Switching to sigmoid/BCE later is a config change.
- LoRA by default; full fine-tune via flag.
- Inputs: txt, pdf (text layer), docx, raw text. Scanned PDFs out of scope.
- Streaming chat completions not supported (400). Only the last user message is classified.
- No git init unless asked; data/ and artifacts/ ignored if initialised.

## Verification
1. `uv run pytest` — segmentation, label map, output schema, API routes (TestClient with stub model).
2. `uv run scripts/02_train.py --limit 512 --epochs 3` → loss < 0.1 (loop learns on MPS).
3. Full run `uv run scripts/02_train.py --config configs/ledgar_qwen3_0.6b.yaml`; monitor `metrics.jsonl`; memory well under 48 GB.
4. `uv run scripts/03_evaluate.py` → test micro-F1 reported vs ≥ 85 target.
5. `uv run scripts/04_predict.py --text "This Agreement shall be governed by the laws of the State of New York."` → top-1 `Governing Laws`.
6. `uv run scripts/05_serve.py` then: `curl localhost:8000/health`; `curl "localhost:8000/v1/dataset?split=train&offset=0&limit=10"`; `curl -X POST localhost:8000/v1/chat/completions -d '{"model":"clause-classifier-qwen3-0.6b","messages":[{"role":"user","content":"<contract text>"}]}'`; and the same via the `openai` Python SDK with `base_url="http://localhost:8000/v1"`.
7. `docker compose up --build` → repeat step 6 against the container (CPU).

## Sources
- LexGLUE / LEDGAR: https://arxiv.org/pdf/2110.00976 , https://github.com/coastalcph/lex-glue , https://huggingface.co/datasets/coastalcph/lex_glue , https://aclanthology.org/2020.lrec-1.155
- Model: https://huggingface.co/Qwen/Qwen3-0.6B-Base
- Alternatives reviewed and rejected: https://huggingface.co/datasets/dvgodoy/CUAD_v1_Contract_Understanding_clause_classification , https://huggingface.co/datasets/mteb/UnfairTOSLegalBenchClassification , https://huggingface.co/litillabs/litil-clause-classifier

---

## Implementation status (2026-10-11)

Implemented as planned, with these changes made during the build (owner requests or measurements):

| Area | Plan | Built | Why |
|---|---|---|---|
| Package | `src/clauseclf` | `src/classification` | owner request |
| Project folder | `domain_models` | `contract-clause-classifier` | owner request |
| Data caching | `datasets.save_to_disk` | pyarrow parquet cache (no `datasets` dependency) | lighter, same function |
| Batch | 8 × accum 4 | 4 × accum 8 (effective 32 unchanged) | batch 8 drove MPS to ~35 GB and swapped on 48 GB |
| Epochs | 3 | 2 | ~2.3 h/epoch on M4 Pro; LEDGAR converges in 2-3 |
| In-training eval | full 10k validation | fixed 3,000 subset | keeps each eval near a minute; final eval uses full splits |
| Precision on MPS | bf16 option | fp32 by default | bf16 autocast measured no faster on MPS |
| Telemetry | not in plan | full per-step / per-layer / per-eval / probe-snapshot / system recording + live dashboard | owner request: watch training live and visualise later |
| Visualisation | not in plan | `scripts/06_render_gif.py`, GIFs in `demo/` | owner request |
| Commands | separate scripts | plus a `Makefile` (`make all`) | one-command pipeline for the owner |
| Agent docs | not in plan | `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/`, `MEMORY.md`, `CHANGELOG.md` | owner request |
| Dependency pins | `>=` | exact `==` pins, Python 3.12 | owner request: stable versions |
| `/v1/labels` | not in plan | added | lists the 100 clause types |

Verified: 28 tests pass; sanity training learns (val acc 0.8% → 17.6% in 64 steps); resume works; merged export, evaluation/calibration, CLI on txt/pdf/docx, native API and the OpenAI SDK, and the Docker image (health, chat, HEALTHCHECK healthy) all work. Full training is left for the owner to run (`make all`).
