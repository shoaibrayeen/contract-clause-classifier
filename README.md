# Contract Clause Classifier

Fine-tune a ~0.6B language model (Qwen3-0.6B) to recognise **clause types in contracts**, then serve it through an **OpenAI-compatible FastAPI**.

Give it a contract (text, PDF or DOCX). It splits the contract into provisions, classifies each one into one of **100 clause types**, and returns calibrated confidences:

```json
{
  "document": [
    {"name": "Governing Laws", "confidence": 0.97},
    {"name": "Confidentiality", "confidence": 0.94},
    {"name": "Terminations", "confidence": 0.88}
  ],
  "clauses": [
    {"index": 4, "text": "4. Confidentiality. Each party shall hold ...",
     "predictions": [{"name": "Confidentiality", "confidence": 0.94}]}
  ],
  "num_clauses": 13
}
```

Training runs on **any machine** (Apple Silicon MPS, NVIDIA CUDA or CPU, auto-detected). While it trains, a **live dashboard** prints in the terminal and **every step is recorded** to disk so the run can be replayed as charts, a GIF or a video later.

![Sanity-run telemetry rendered as a GIF](demo/sanity_training.gif)

---

## Contents

1. [How it works](#how-it-works)
2. [Training data](#training-data)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Run the full pipeline](#run-the-full-pipeline)
6. [Training in detail](#training-in-detail)
7. [Training telemetry and visualisation](#training-telemetry-and-visualisation)
8. [Evaluation](#evaluation)
9. [Predict from the command line](#predict-from-the-command-line)
10. [API (FastAPI, OpenAI-compatible)](#api-fastapi-openai-compatible)
11. [Docker](#docker)
12. [Project layout](#project-layout)
13. [Configuration reference](#configuration-reference)
14. [Testing](#testing)
15. [Troubleshooting](#troubleshooting)
16. [Pinned versions](#pinned-versions)
17. [License](#license)
18. [Author](#author)

---

## How it works

```
contract (.txt / .pdf / .docx, or the user message of a chat request)
   │  segment.py   split into provisions (numbered sections, headings, paragraphs)
   ▼
[provision 1 … provision n]
   │  Qwen3-0.6B-Base + LoRA adapters + 100-way classification head
   │  (last-token pooling, temperature-scaled softmax)
   ▼
per-provision top-k [{name, confidence}]
   │  aggregate: max confidence per clause type across provisions
   ▼
{"document": [...], "clauses": [...], "num_clauses": n}
```

| Choice | What | Why |
|---|---|---|
| Dataset | [LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue) (LexGLUE) | 80,000 real contract provisions from SEC filings, 100 clause types, chronological train/validation/test split (60k / 10k / 10k). Not CUAD, by design. |
| Model | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 0.6B parameters, Apache-2.0, strong for its size. Swap to `Qwen/Qwen2.5-0.5B` or `HuggingFaceTB/SmolLM2-360M` with one config line. |
| Method | LoRA (r=16) on all attention and MLP projections + trainable head | 10.2M trainable of 606M parameters. Fits comfortably on a 48 GB Mac. Full fine-tuning is one flag. |
| Framework | PyTorch + Hugging Face `transformers` + `peft` | Same code on MPS, CUDA and CPU. |
| Confidence | Softmax with temperature scaling fitted on validation | Confidences are calibrated, so 0.9 means right about 90% of the time. |

LEDGAR labels each provision with a single clause type. Multiple clause types per contract come from classifying each provision separately; multiple labels per provision come from the top-k list above the threshold.

The full design rationale is in [plans/PLAN.md](plans/PLAN.md).

---

## Training data

The model is trained on **LEDGAR** (Labeled EDGAR), as packaged in the [LexGLUE benchmark](https://huggingface.co/datasets/coastalcph/lex_glue) (`coastalcph/lex_glue`, config `ledgar`).

| | |
|---|---|
| Source | Provisions (paragraphs) of commercial contracts filed with the US Securities and Exchange Commission and published on EDGAR, which is public information. Collected by Tuggener et al., [LREC 2020](https://aclanthology.org/2020.lrec-1.155). |
| Size | 80,000 provisions from 12,000+ contracts: **60,000 train** (filed 2016 to 2017), **10,000 validation** (2018), **10,000 test** (2019). The chronological split means the test set is newer than the training set. |
| Labels | **100 clause types**, one per provision, chosen by LexGLUE as the 100 most frequent of the original 12,000+ section headings. |
| Text | English. Average provision is 114 words (median 86); with the Qwen3 tokenizer, mean 138 tokens and 1.9% longer than the 512-token limit. |
| Format | Two columns, `text` and `label` (an integer id into the label list). Downloaded as three parquet files (28 MB). |
| License | LexGLUE is published under **CC BY 4.0**; the underlying filings are public SEC records. See [License](#license). |

The 100 clause types:

> Adjustments, Agreements, Amendments, Anti-Corruption Laws, Applicable Laws, Approvals, Arbitration, Assignments, Assigns, Authority, Authorizations, Base Salary, Benefits, Binding Effects, Books, Brokers, Capitalization, Change In Control, Closings, Compliance With Laws, Confidentiality, Consent To Jurisdiction, Consents, Construction, Cooperation, Costs, Counterparts, Death, Defined Terms, Definitions, Disability, Disclosures, Duties, Effective Dates, Effectiveness, Employment, Enforceability, Enforcements, Entire Agreements, Erisa, Existence, Expenses, Fees, Financial Statements, Forfeitures, Further Assurances, General, Governing Laws, Headings, Indemnifications, Indemnity, Insurances, Integration, Intellectual Property, Interests, Interpretations, Jurisdictions, Liens, Litigations, Miscellaneous, Modifications, No Conflicts, No Defaults, No Waivers, Non-Disparagement, Notices, Organizations, Participations, Payments, Positions, Powers, Publicity, Qualifications, Records, Releases, Remedies, Representations, Sales, Sanctions, Severability, Solvency, Specific Performance, Submission To Jurisdiction, Subsidiaries, Successors, Survival, Tax Withholdings, Taxes, Terminations, Terms, Titles, Transactions With Affiliates, Use Of Proceeds, Vacations, Venues, Vesting, Waiver Of Jury Trials, Waivers, Warranties, Withholdings

The classes are imbalanced. In the training split the most frequent types are Governing Laws (3,167), Notices (2,493), Counterparts (2,427), Entire Agreements (2,340) and Severability (1,808); the rarest are Books (23), Assigns (31), Qualifications (47), Anti-Corruption Laws (106) and Sanctions (118). Macro-F1 is therefore reported alongside accuracy, and `train.class_weighted_loss` can up-weight rare types.

Example rows:

| text | label |
|---|---|
| This Lease shall in all respects be governed by the laws of the State of New York. | Governing Laws |
| The Committee may terminate the Plan at any time and grant no further Restricted Fund Shares. | Terminations |
| Except as otherwise set forth in this Debenture, the Company ... expressly waives presentment, protest, demand, notice of dishonor ... | Waivers |

Browse the data yourself with `make data` and then `GET /v1/dataset?split=train&offset=0&limit=10` on the API, or read `data/raw/ledgar/*.parquet` directly.

**What the model does not learn from this data.** LEDGAR labels one clause type per provision and contains no annotation of risk, parties, dates or obligations. Multi-label output for a whole contract comes from splitting it into provisions. Provisions written in styles far from SEC-filed commercial contracts (consumer terms, non-English text) will be classified less reliably.

---

## Requirements

| | Minimum | Tested |
|---|---|---|
| OS | macOS on Apple Silicon, or Linux | macOS 26 on Apple M4 Pro, 48 GB |
| Memory | 24 GB unified memory / RAM for training | about 20 GB of accelerator memory with the default config |
| Disk | 10 GB free | base model 1.2 GB, data 0.1 GB, each run about 2 GB |
| Python | 3.12 (installed by uv automatically) | 3.12.13 |
| Tools | [uv](https://docs.astral.sh/uv/), `make`; Docker optional | uv 0.11, Docker 29 |

Training time for the default config (2 epochs, 3,750 optimizer steps):

| Hardware | Time |
|---|---|
| Apple M4 Pro (MPS) | about 5 hours |
| Modern NVIDIA GPU (CUDA, bf16) | about 30 to 50 minutes |

---

## Setup

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. From the project folder: install Python 3.12 and all pinned dependencies into .venv
cd contract-clause-classifier
make setup            # same as: uv sync

# 3. Check everything works (no model or download needed)
make test
```

`uv sync` reads `.python-version` (3.12) and `uv.lock`, so everyone gets the identical, tested set of packages. Dependency groups:

| Group | Contents | Installed by |
|---|---|---|
| runtime | torch, transformers, FastAPI, pyarrow, PyMuPDF, python-docx, rich, psutil … | always, and in Docker |
| `train` | peft, accelerate, scikit-learn | `uv sync` (default) |
| `viz` | matplotlib, pillow | `uv sync` (default) |
| `dev` | pytest, httpx, openai | `uv sync` (default) |

Optional: copy `.env.example` to `.env` to change API settings.

---

## Run the full pipeline

One command runs everything: download data, train with the live dashboard, evaluate and calibrate, and render the training GIF.

```bash
make all
```

Or step by step:

```bash
make data       # 1. download LEDGAR (28 MB) and tokenize all splits
make train      # 2. fine-tune; live dashboard in the terminal, telemetry saved to disk
make evaluate   # 3. fit confidence calibration on validation, report test metrics
make gif        # 4. render the run's telemetry to a GIF (also copied to demo/)
make predict    # 5. classify demo/sample_contract.txt
make serve      # 6. start the API on http://localhost:8000  (docs at /docs)
```

Each `make` target is a thin wrapper; the underlying commands are in the [Makefile](Makefile) (`make help` lists them).

**Where the trained model goes.** Every run writes to `artifacts/runs/<run-name>/`. When training ends, the best checkpoint is merged into a standalone model at `artifacts/runs/<run-name>/merged/`, and the link `artifacts/best` is pointed at it. The CLI, the API and Docker all load `artifacts/best`.

---

## Training in detail

```bash
make train                                   # default run name: ledgar-qwen3-0.6b-lora
make train RUN=my-experiment                 # different run name
make resume                                  # continue after Ctrl-C, a crash or a reboot
make sanity                                  # 5-minute smoke test on 256 examples
```

Direct invocation with overrides (any config value can also be changed in the YAML):

```bash
uv run python scripts/02_train.py --run-name exp2 --epochs 3 --lr 1e-4
uv run python scripts/02_train.py --model-name Qwen/Qwen2.5-0.5B --run-name qwen25
uv run python scripts/02_train.py --device cpu --limit 2000          # force CPU, subset
```

### What you see in the terminal

A live panel refreshes twice a second: progress and ETA, loss and smoothed loss with a trend sparkline, learning rate, gradient norm (and whether it was clipped), tokens per second, padding ratio, accelerator memory, CPU and RAM, the clause types in the current batch, and the latest validation results. If output is redirected to a file, it prints one plain log line every 10 steps instead.

### What happens during a run

1. The Qwen3 backbone is loaded and LoRA adapters plus a 100-way head are attached.
2. Batches are grouped by length to cut padding (about 2% padding instead of about 50%).
3. Every 500 optimizer steps it evaluates on a fixed 3,000-example validation subset, saves `checkpoint-last/` (with optimizer state for resume), and keeps `checkpoint-best/` by macro-F1. Training stops early after 4 evaluations without improvement.
4. At the end, `checkpoint-best/` is merged into `merged/` and `artifacts/best` is updated.

### Run directory

```
artifacts/runs/<run-name>/
├── config.yaml          resolved config used for the run
├── metrics.jsonl        compact log (every 10 steps + each eval)
├── checkpoint-best/     best LoRA adapter + head + tokenizer
├── checkpoint-last/     latest adapter + optimizer/scheduler state (for --resume)
├── merged/              standalone serving model (bf16 safetensors, labels.json, calibration.json, eval.json)
├── telemetry/           everything needed to replay the run (next section)
└── visuals/             training.gif / training.png (after `make gif`)
```

### Memory notes for Apple Silicon

The default micro-batch is 4 with 8 accumulation steps (effective batch 32). A micro-batch of 8 pushes the MPS allocator past 35 GB and starts swapping on a 48 GB machine. The trainer also releases cached accelerator memory every 25 steps. Closing heavy apps (browsers, Docker VMs) during training helps.

---

## Training telemetry and visualisation

Every run records its full history to `artifacts/runs/<run-name>/telemetry/`. Files are append-only and line-buffered, so they survive crashes and `--resume`, and can be read while training is still going.

| File | What it captures |
|---|---|
| `meta.json` | config, hardware and software versions, dataset statistics (label counts, length histograms), probe-set ids, label names |
| `steps.jsonl` | one line per optimizer step: loss, learning rate, gradient norm, clipping, step time, tokens/s, padding, memory, and the batch itself: example ids, labels, lengths, per-example loss and correctness |
| `layers.jsonl` | every 10 steps: gradient norm of each of the 28 transformer layers and the head |
| `evals.jsonl` | each evaluation: accuracy, macro-F1, loss, per-class F1 for all 100 types, LoRA weight norm per layer |
| `evals/step_*.npz` | each evaluation: 100x100 confusion matrix and calibration bins |
| `snapshots/step_*.npz` | at step 0 and each evaluation, for a fixed probe set of about 500 validation examples: predicted probabilities, final-layer embeddings, per-layer activation norms, and full per-layer vectors for about 100 of them |
| `events.jsonl` | start, snapshot, eval, checkpoint, early stop, done |
| `system.jsonl` | every 5 seconds: CPU %, RAM, process memory, accelerator memory, training phase |

Together these show how the loss fell, which clause types the model learned first, how each layer's gradients evolved, how individual examples flowed through training, and how each clause type's representation moved through the network's layers and separated into clusters. The exact schema is in [docs/TELEMETRY.md](docs/TELEMETRY.md).

Render a GIF at any time, including mid-training:

```bash
make gif                                          # default run, copies to demo/
uv run python scripts/06_render_gif.py --run artifacts/runs/sanity --frames 80 --fps 10
```

---

## Evaluation

```bash
make evaluate
uv run python scripts/03_evaluate.py --model-dir artifacts/runs/<run>/merged --limit 1000   # quick
```

It fits a temperature on the validation split, writes `calibration.json` next to the model (the API applies it automatically), and writes `eval.json` with test accuracy, micro, macro and weighted F1, log loss, calibration error before and after scaling, per-class precision and recall, and the most frequent confusions.

Reference points on the LEDGAR test set from the [LexGLUE paper](https://arxiv.org/abs/2110.00976): BERT-class encoders reach about 87 to 88 micro-F1 and 81 to 83 macro-F1. This project targets micro-F1 of 85 or better.

---

## Predict from the command line

```bash
uv run python scripts/04_predict.py demo/sample_contract.txt
uv run python scripts/04_predict.py path/to/contract.pdf --top-k 5 --threshold 0.2 --out result.json
uv run python scripts/04_predict.py --text "This Agreement shall be governed by the laws of New York."
```

Inputs: `.txt`, `.md`, `.pdf` with a text layer, `.docx`. Scanned PDFs need OCR first.

Sample output for the `--text` call:

```json
{
  "document": [
    {"name": "Governing Laws", "confidence": 0.8069}
  ],
  "clauses": [
    {
      "index": 0,
      "text": "This Agreement shall be governed by the laws of New York.",
      "predictions": [
        {"name": "Governing Laws", "confidence": 0.8069}
      ]
    }
  ],
  "num_clauses": 1
}
```

`document` lists every clause type found anywhere in the input with its best confidence, highest first. `clauses` gives the top-k predictions per provision (the top-1 is always kept, the rest only above the threshold). A provision the model is unsure about can therefore appear in `clauses` with a low-confidence top-1 while adding nothing to `document`.

> All sample outputs in this README come from the 5-minute `make sanity` model, so the confidences are modest. A full `make train` run produces much sharper predictions.

---

## API (FastAPI, OpenAI-compatible)

```bash
make serve                       # http://localhost:8000, Swagger UI at /docs, schema at /openapi.json
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | 200 with model and device info when the model is loaded, 503 otherwise |
| GET | `/v1/models` | OpenAI model list (one model: `clause-classifier-qwen3-0.6b`) |
| GET | `/v1/models/{id}` | one model |
| POST | `/v1/chat/completions` | classify the last user message; reply content is the predictions JSON |
| GET | `/v1/dataset` | LEDGAR rows, 10 per page by default: `?split=train&offset=0&limit=10` (limit max 100) |
| GET | `/v1/labels` | the 100 clause types |

### Endpoints with sample requests and responses

#### `GET /health`

```bash
curl -s localhost:8000/health
```

`200` when the model is loaded:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "clause-classifier-qwen3-0.6b",
  "model_dir": "artifacts/best",
  "device": "mps",
  "num_labels": 100,
  "uptime_s": 3.304,
  "version": "0.1.0",
  "detail": null
}
```

`503` while the model is missing or still loading (the dataset endpoints keep working):

```json
{
  "status": "unavailable",
  "model_loaded": false,
  "model": "clause-classifier-qwen3-0.6b",
  "model_dir": "artifacts/best",
  "device": null,
  "num_labels": null,
  "uptime_s": 2.146,
  "version": "0.1.0",
  "detail": "No trained model at artifacts/best (run scripts/02_train.py first)"
}
```

#### `GET /v1/models` and `GET /v1/models/{id}`

```bash
curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/models/clause-classifier-qwen3-0.6b
```

```json
{
  "object": "list",
  "data": [
    {"id": "clause-classifier-qwen3-0.6b", "object": "model", "created": 1791666222, "owned_by": "local"}
  ]
}
```

An unknown id returns `404` in the OpenAI error format:

```json
{
  "error": {
    "message": "The model 'gpt-4o' does not exist",
    "type": "not_found_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

#### `POST /v1/chat/completions` (classify a clause or a whole contract)

```bash
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "clause-classifier-qwen3-0.6b",
  "messages": [{"role": "user", "content": "This Agreement shall be governed by the laws of New York."}]
}'
```

A standard `chat.completion` object. The predictions are the JSON string in `choices[0].message.content`:

```json
{
  "id": "chatcmpl-5f0b4683344349f48f47d5b4",
  "object": "chat.completion",
  "created": 1791666511,
  "model": "clause-classifier-qwen3-0.6b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"document\": [{\"name\": \"Governing Laws\", \"confidence\": 0.8069}], \"clauses\": [{\"index\": 0, \"text\": \"This Agreement shall be governed by the laws of New York.\", \"predictions\": [{\"name\": \"Governing Laws\", \"confidence\": 0.8069}]}], \"num_clauses\": 1}",
        "refusal": null
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {"prompt_tokens": 12, "completion_tokens": 82, "total_tokens": 94},
  "system_fingerprint": "fp_ledgar-qwen3-0.6b-lora"
}
```

Send a whole contract the same way (here the 13-section `demo/sample_contract.txt`, with the optional `top_k` and `threshold` extras). The parsed `content` looks like this, abbreviated:

```bash
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "$(python3 -c '
import json; print(json.dumps({"model": "clause-classifier-qwen3-0.6b", "top_k": 2, "threshold": 0.2,
  "messages": [{"role": "user", "content": open("demo/sample_contract.txt").read()}]}))')" \
  | python3 -c "import sys, json; print(json.loads(sys.stdin.read())['choices'][0]['message']['content'])"
```

```json
{
  "document": [
    {"name": "Governing Laws", "confidence": 0.5642},
    {"name": "Counterparts", "confidence": 0.2466},
    {"name": "Entire Agreements", "confidence": 0.1002}
  ],
  "clauses": [
    {
      "index": 0,
      "text": "MASTER SERVICES AGREEMENT This Master Services Agreement (the \"Agreement\") is entered into as of March 1, 2025 ...",
      "predictions": [{"name": "Survival", "confidence": 0.0358}]
    },
    {
      "index": 1,
      "text": "1. Definitions. Capitalized terms used but not defined elsewhere in this Agreement have the meanings set forth ...",
      "predictions": [{"name": "Headings", "confidence": 0.0468}]
    },
    "... 11 more provisions ..."
  ],
  "num_clauses": 13
}
```

Unsupported options return `400`:

```json
{
  "error": {
    "message": "stream=true is not supported by this classifier",
    "type": "invalid_request_error",
    "param": "stream",
    "code": "unsupported_parameter"
  }
}
```

And without a loaded model the endpoint returns `503` with code `model_not_loaded`.

#### `GET /v1/dataset` (browse the training data, 10 rows per page)

```bash
curl -s "localhost:8000/v1/dataset?split=train&offset=0&limit=2"
```

```json
{
  "split": "train",
  "offset": 0,
  "limit": 2,
  "total": 60000,
  "next_offset": 2,
  "items": [
    {
      "id": "train-0",
      "text": "Except as otherwise set forth in this Debenture, the Company, for itself and its legal representatives, successors and assigns, expressly waives presentment, protest, demand, notice of dishonor, notice of nonpayment, notice of maturity, notice of protest, presentment for the purpose of accelerating maturity, and diligence in collection.",
      "label_id": 97,
      "label": "Waivers"
    },
    {
      "id": "train-1",
      "text": "No ERISA Event has occurred or is reasonably expected to occur that, when taken together with all other such ERISA Events ... nor otherwise has any liability under Title IV of ERISA.",
      "label_id": 39,
      "label": "Erisa"
    }
  ]
}
```

`split` is `train`, `validation` or `test`; `offset` defaults to 0 and `limit` to 10 (maximum 100). Page through with `next_offset` until it is `null`. A bad split returns `422`:

```json
{
  "error": {
    "message": "Input should be 'train', 'validation' or 'test'",
    "type": "invalid_request_error",
    "param": "split",
    "code": "invalid_value"
  }
}
```

#### `GET /v1/labels`

```bash
curl -s localhost:8000/v1/labels
```

```json
{
  "object": "list",
  "total": 100,
  "data": ["Adjustments", "Agreements", "Amendments", "Anti-Corruption Laws", "Applicable Laws", "Approvals", "..."]
}
```

### OpenAI Python SDK

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")   # or your API_KEY
resp = client.chat.completions.create(
    model="clause-classifier-qwen3-0.6b",
    messages=[{"role": "user", "content": open("demo/sample_contract.txt").read()}],
    extra_body={"top_k": 3, "threshold": 0.3},       # optional, non-OpenAI knobs
)
result = json.loads(resp.choices[0].message.content)
print(resp.usage)                # CompletionUsage(prompt_tokens=658, completion_tokens=1011, total_tokens=1669)
print(result["document"])        # [{"name": "Governing Laws", "confidence": 0.5642}, ...]
```

Request details:

* The contract is the **last `user` message** (string or `[{"type": "text", "text": ...}]` parts). System messages are ignored.
* Optional extras: `top_k` (1 to 100), `threshold` (0 to 1), `segment` (force segmentation on or off; by default inputs over 350 words are segmented).
* `stream: true` and `n > 1` return a 400 error. Unknown OpenAI parameters are accepted and ignored.
* `usage` reports real token counts.
* Errors use the OpenAI envelope: `{"error": {"message", "type", "param", "code"}}`.
* Set `API_KEY` to require `Authorization: Bearer <key>` on `/v1/*`. `/health` stays open.

Environment variables (see `.env.example`): `MODEL_DIR`, `MODEL_ID`, `DATA_DIR`, `DEVICE`, `TOP_K`, `THRESHOLD`, `BATCH_SIZE`, `API_KEY`, `CORS_ORIGINS`, `MAX_INPUT_CHARS`.

---

## Docker

The image serves a trained model on CPU. Training is done outside Docker.

```bash
make docker-build                 # docker build -t contract-clause-classifier:0.1.0 .
make docker-up                    # docker compose up: http://localhost:8000, mounts ./artifacts and ./data
make docker-down
```

* Docker on macOS cannot use the Apple GPU, so the container runs on CPU (fine for classification). For GPU speed on a Mac, use `make serve` natively.
* The image is about 1.9 GB (CPU-only torch). It runs as a non-root user and has a `HEALTHCHECK` on `/health`.
* `artifacts/best` is a relative symlink, which is why the whole `artifacts/` folder is mounted.
* **Colima users:** Colima only shares folders listed under `mounts:` in `~/.colima/default/colima.yaml`. If the project is not under a shared folder, the mounts arrive empty and `/health` reports "No trained model". Add the project folder to `mounts:` and run `colima restart`, or copy the model into a container:

  ```bash
  docker create --name ccc -p 8000:8000 contract-clause-classifier:0.1.0
  docker cp -L artifacts/best ccc:/app/artifacts/best
  docker cp data/raw ccc:/app/data/raw
  docker start ccc
  ```

---

## Project layout

```
contract-clause-classifier/
├── Makefile                    one-word commands for every step (make help)
├── pyproject.toml / uv.lock    pinned dependencies (Python 3.12)
├── configs/ledgar_qwen3_0.6b.yaml   all training settings
├── src/classification/
│   ├── config.py               config dataclasses, device + precision resolution
│   ├── data.py                 LEDGAR download, tokenisation cache, length-grouped batching, dataset API store
│   ├── labels.py               label map I/O
│   ├── model.py                build (LoRA) / load / merge-export the classifier
│   ├── train.py                device-agnostic training loop, checkpoints, resume
│   ├── telemetry.py            run recorder, probe snapshots, system monitor
│   ├── dashboard.py            live terminal dashboard
│   ├── evaluate.py             metrics, temperature scaling, calibration error
│   ├── segment.py              contract -> provisions
│   ├── io_utils.py             read txt / pdf / docx
│   ├── predict.py              Predictor: provisions -> [{name, confidence}]
│   └── api/                    FastAPI app, OpenAI schemas, routes (health, openai, dataset)
├── scripts/                    01_prepare_data … 06_render_gif (numbered pipeline steps)
├── tests/                      pytest suite (stub model; no download or GPU needed)
├── demo/                       sample contract, telemetry GIF and final frame
├── docs/TELEMETRY.md           telemetry file schema
├── plans/PLAN.md               approved design plan
├── Dockerfile, docker-compose.yml, .env.example
├── AGENTS.md, CLAUDE.md, .cursor/rules/, .codex/   guidance for coding agents (AGENTS.md is canonical)
├── MEMORY.md                   project decisions and gotchas (agent memory)
├── CHANGELOG.md, changelog.html   release notes (Markdown and a standalone HTML page)
├── LICENSE                     Apache-2.0
├── data/                       (git-ignored) raw parquet + tokenised cache
└── artifacts/                  (git-ignored) runs/, best -> runs/<run>/merged
```

---

## Configuration reference

Everything lives in [configs/ledgar_qwen3_0.6b.yaml](configs/ledgar_qwen3_0.6b.yaml). The most useful settings:

| Key | Default | Meaning |
|---|---|---|
| `model_name` | `Qwen/Qwen3-0.6B-Base` | any HF causal LM with a sequence-classification head |
| `full_finetune` | `false` | train all weights instead of LoRA (use `train.lr: 2e-5`) |
| `device` | `auto` | `auto` picks cuda, then mps, then cpu |
| `precision` | `auto` | bf16 autocast on CUDA, fp32 on MPS and CPU |
| `data.max_len` | `512` | tokens per provision (about 2% of LEDGAR is longer) |
| `lora.r` / `lora.alpha` | `16` / `32` | LoRA rank and scale |
| `train.epochs` | `2` | passes over the 60k training provisions |
| `train.batch_size` × `train.grad_accum` | `4 × 8` | effective batch 32 |
| `train.lr` | `2e-4` | peak learning rate, 3% warm-up then cosine decay |
| `train.eval_steps` | `500` | optimizer steps between evaluations |
| `train.eval_limit` | `3000` | validation examples used during training (`null` for all 10k) |
| `train.early_stopping_patience` | `4` | evaluations without macro-F1 improvement before stopping |
| `train.class_weighted_loss` | `false` | up-weight rare clause types |
| `train.empty_cache_steps` | `25` | release cached accelerator memory |
| `telemetry.*` | all on | what to record (see [docs/TELEMETRY.md](docs/TELEMETRY.md)) |

---

## Testing

```bash
make test
```

28 tests cover segmentation, data paging and batching, the prediction output format, every API route and error case, API-key auth, the OpenAPI schema, and compatibility with the official OpenAI Python SDK. They use a keyword-based stub model, so they run in seconds with no download or GPU.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` returns 503 "No trained model" | Train first (`make train`), or set `MODEL_DIR` to a `merged/` folder. |
| Training is slow and the Mac feels sluggish | Memory pressure. Keep `batch_size: 4`, close heavy apps, check swap with `sysctl vm.swapusage`. |
| Training was interrupted | `make resume` continues from `checkpoint-last/`. |
| `MPS backend out of memory` | Lower `train.batch_size` to 2 and raise `grad_accum` to 16, or set `gradient_checkpointing: true`. |
| Docker `/health` says model missing on Colima | See the Colima note in [Docker](#docker). |
| Hugging Face rate-limit warnings | Optional: `export HF_TOKEN=...` (a free read token). |
| A PDF returns an error about a text layer | The PDF is scanned; run OCR first. |

---

## Pinned versions

Exact versions are pinned in `pyproject.toml` and locked in `uv.lock`; all are stable releases tested together.

| Package | Version | Package | Version |
|---|---|---|---|
| Python | 3.12 | fastapi | 0.143.0 |
| torch | 2.14.1 | uvicorn | 0.54.0 |
| transformers | 5.19.0 | pydantic | 2.14.0 |
| peft | 0.21.2 | pyarrow | 26.0.0 |
| accelerate | 1.15.0 | pymupdf | 1.28.2 |
| scikit-learn | 1.9.1 | python-docx | 1.2.0 |
| huggingface_hub | 2.2.0 | rich | 15.0.0 |
| safetensors | 0.8.0 | psutil | 7.2.2 |
| numpy | 2.5.3 | matplotlib | 3.11.2 |
| pillow | 12.3.0 | Docker base | python:3.12.15-slim-bookworm |

## License

This project is released under the **Apache License 2.0**. See [LICENSE](LICENSE).

Apache 2.0 was chosen because it is permissive (commercial use, modification and redistribution are allowed), it includes an explicit patent grant, and it matches the licenses of the main components this project builds on, so the whole stack is license-compatible:

| Component | License |
|---|---|
| This code | [Apache-2.0](LICENSE) |
| [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) (base model) | Apache-2.0 |
| PyTorch | BSD-3-Clause |
| transformers, peft, accelerate, huggingface_hub | Apache-2.0 |
| FastAPI, uvicorn, pydantic, rich, typer | MIT |
| [LexGLUE / LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue) (training data) | CC BY 4.0 (attribution required); derived from public SEC EDGAR filings |

A model you train with this code is a derivative of Qwen3-0.6B-Base (Apache-2.0) and of LEDGAR (CC BY 4.0). If you redistribute trained weights, keep the attribution to both. Cite the data as:

> Tuggener, D., von Däniken, P., Peetz, T., and Cieliebak, M. (2020). LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts. LREC 2020.
> Chalkidis, I. et al. (2022). LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. ACL 2022.

## Author

**Shoaib Rayeen**
Email: [shoaibrayeen.me@gmail.com](mailto:shoaibrayeen.me@gmail.com)
Portfolio: [https://shoaibrayeen.github.io/](https://shoaibrayeen.github.io/)
Repository: [github.com/shoaibrayeen/contract-clause-classifier](https://github.com/shoaibrayeen/contract-clause-classifier)

Questions, bug reports and pull requests are welcome through the repository's issues page.
