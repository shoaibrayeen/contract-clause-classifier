# Plan: classical model, side-by-side playground, and model pages

(Approved 2026-10-11. **Status: implemented 2026-10-11** — `ledgar-tfidf-lr-v1` trained and evaluated (test 90.88%), `/v1/compare`, playground compare mode and `docs/models/` pages done; transformer v1 files verified unchanged by checksum. The earlier transformer plan is `plans/PLAN.md`.)

## Context

`ledgar-qwen3-0.6b-v1` (the fine-tuned transformer) is trained, versioned and served. The owner now wants a
**traditional machine-learning classifier** trained on the same data, kept in its own directories, so that:

1. the playground calls **both** models and shows each one's prediction and confidence side by side;
2. there is a **comparison page** for the two models and a **detail page per model** (`models/transformer`,
   `models/classical`) with description, training curves, accuracy and every other measured detail, in the
   same standalone style as `changelog.html`;
3. **the existing transformer model and its flow are not changed.**

Evidence this will work: during the label-group validation a TF-IDF + logistic-regression baseline reached
87.8% on the 100 original labels and 91.2% on the 78 merged clause types (full 10k validation split), in about
3 minutes of CPU time. The transformer is at 92.2% validation / 92.5% test.

Owner decisions (asked and answered):

| Question | Decision |
|---|---|
| Where the pages live | Static HTML generated into `docs/models/` (viewable on GitHub), also served by the API at `/models/*` |
| Both models in the API | New `POST /v1/compare`; `/v1/chat/completions` stays single-model and unchanged |
| Classical algorithms | TF-IDF + logistic regression is the served model; linear SVM and Complement Naive Bayes are trained and reported on the classical page |
| Playground default | Compare both side by side when a classical model exists; a selector still allows one model |

## Requirements checklist

- [x] Classical model trained on the same dataset (same 78 clause types, same splits) → step 1
- [x] Saved in classical directories: `src/classification/classical/`, `configs/classical/`, `scripts/classical/`, `artifacts/classical/runs/` → layout, step 1
- [x] Playground calls both models and shows confidence for each → `POST /v1/compare` (step 4) + compare view (step 5)
- [x] Comparison page for the two models → `docs/models/compare.html`, served at `/models/compare` (step 6)
- [x] `models/transformer` and `models/classical` detail pages with description, loss, accuracy and all details → step 6; the classical page shows training and validation log loss plus a learning curve, since logistic regression has no per-step loss curve
- [x] No change to the existing transformer model, its training flow, its artifacts or the API default → "What does not change"
- [x] Questions asked before planning → pages location, compare endpoint, algorithms, playground default

## What does not change

* `src/classification/train.py`, `model.py`, `evaluate.py`, `scripts/02_train.py`, `03_evaluate.py`,
  `configs/ledgar_qwen3_0.6b.yaml`, `make all`: untouched.
* `artifacts/runs/ledgar-qwen3-0.6b-v1/`: read-only (the pages generator reads its `merged/eval.json`,
  `merged/model_info.json`, `config.yaml` and `telemetry/`). `artifacts/before_v2.sha256` verifies this afterwards.
* API defaults: a request without `model` (or `latest`) still resolves to the latest **transformer**;
  `artifacts/best` keeps pointing at it. Registry changes below are additive.

## Directory layout ("classical directories")

```
src/classification/classical/
├── __init__.py
├── features.py        build_features(cfg): TF-IDF word 1-2 grams + char_wb 3-5 grams (FeatureUnion), sublinear tf, min_df, max_features
├── train.py           train candidates on train split, pick by validation, learning curve, telemetry, save run dir
├── backend.py         SklearnBackend (implements predict.Backend): joblib pipeline, decision_function -> softmax / temperature
└── explain.py         top weighted n-grams per clause type (served model only)
configs/classical/ledgar_tfidf_lr.yaml      model_version: 1, run_name: ledgar-tfidf-lr, features, candidates, learning-curve sizes
scripts/classical/01_train.py              train next classical version (versioned like the transformer: never overwrite, resume n/a)
scripts/classical/02_evaluate.py           calibrate (temperature on validation) + full test report into the run's merged/
scripts/10_model_pages.py                  generate docs/models/{compare,transformer,classical}.html + assets
scripts/11_compare.py                      CLI: both models on a file/--text, side-by-side table
artifacts/classical/runs/ledgar-tfidf-lr-v<N>/
├── config.yaml
├── merged/            model.joblib, labels.json, label_space.json, model_info.json (family: classical, backend: sklearn), calibration.json, eval.json, top_features.json, config.json (copy of the YAML as JSON; satisfies "complete" check)
└── telemetry/         candidates.json (each candidate: params, validation accuracy/macro-F1, fit seconds, feature count), learning_curve.json (accuracy vs 10/25/50/100% of train), events.jsonl
docs/models/
├── compare.html, transformer.html, classical.html
├── assets/*.png       matplotlib charts (committed; small)
└── data/*.json        the numbers behind the pages (for reuse / re-rendering)
```

## Steps

### 1. Classical training (`src/classification/classical/`, `scripts/classical/01_train.py`)

* Data: reuse `data.load_split(cfg.data, split)` (texts + ids already mapped through the 78-type label space
  `configs/ledgar_label_groups.yaml`) and `data.get_label_space` for `label_space.json`. Same splits as v1.
* Candidates (configurable list in the YAML), all scikit-learn 1.9.1 already pinned:
  1. `LogisticRegression` on word 1-2 grams + char_wb 3-5 grams, C grid `[3, 10, 30]`, `max_iter 2000` (served).
  2. `LinearSVC` (C grid) wrapped in `CalibratedClassifierCV` (reported; also usable as a backend since it exposes `predict_proba`).
  3. `ComplementNB` (reported).
  Model selection: validation accuracy among LR configs picks the served pipeline; SVM/NB results are recorded for the page only.
* Learning curve for the served config: fits on 10%, 25%, 50%, 100% of train, validation accuracy each (`telemetry/learning_curve.json`).
* Save: `joblib.dump(pipeline, merged/model.joblib)`, `labels.json`, `label_space.json` (from `LabelSpace.save`),
  `model_info.json` via `registry.write_model_info(..., family="classical", backend="sklearn", method="tfidf+lr",
  features=..., n_features=..., train_seconds=..., best_val_accuracy=...)`, `config.json`, `top_features.json`
  (top 15 positive n-grams per class from the LR coefficients).
* Versioning: `registry.plan_training_run(artifacts/classical/runs, "ledgar-tfidf-lr", model_version)` and
  `registry.bump_config_version` reused unchanged; the classical config has its own `model_version`.
* Expected time: about 20 to 30 minutes of CPU, 2 to 3 GB of RAM.

### 2. Evaluation (`scripts/classical/02_evaluate.py`)

Reuse `evaluate.fit_temperature`, `evaluate.full_report`, `evaluate.expected_calibration_error`, `softmax` with the
pipeline's `decision_function` output as logits (multinomial LR logits are exactly that). Writes `calibration.json`,
`eval.json` (accuracy, top-3, micro/macro/weighted F1, NLL, ECE, per-class, top confusions, same schema as v1) and
updates `model_info.json["test"]` via `registry.update_model_info`, so `make models` and `/v1/models` show it.
Also writes `telemetry/test_predictions.npz` (argmax + probs on the test split) for the agreement analysis.

### 3. Registry and model store (additive, `src/classification/registry.py`, `api/models.py`, `api/settings.py`)

* `ModelInfo.family` ("transformer" default when absent in `model_info.json`, "classical" when written by step 1).
* `is_complete`: `merged/config.json` exists (classical runs write one too, so no change in logic).
* `latest_model(models, base=None, family="transformer")`: default behaviour unchanged. New alias `latest-classical`
  resolves to the newest classical version; base names (`ledgar-tfidf-lr`) and exact ids work as today.
* `ModelStore` takes `runs_dirs: list[Path]` (`RUNS_DIR` + new `CLASSICAL_RUNS_DIR=artifacts/classical/runs`), scans
  both, and the loader dispatches on `family`: `load_predictor` (HF) or `SklearnBackend`. `Predictor` is reused as
  is (segmentation, top-k, aggregation). `count_tokens` in the sklearn backend uses the Qwen tokenizer when the
  transformer tokenizer is available (same `usage` semantics), else whitespace tokens.
* `/v1/models` and `/v1/models/{id}`: `ModelCard.family` added; `/health` adds `latest_classical`.
* Dependencies: `scikit-learn` and `joblib` move from the `train` group to runtime deps so the API and Docker image
  can load `model.joblib` (image grows about 40 MB). `peft`/`accelerate` stay training-only. `uv.lock` updated.

### 4. `POST /v1/compare` (`api/routes/compare.py`, `api/schemas.py`)

Request: `{"messages": [...]} ` (OpenAI-style, last user message) or `{"text": "..."}`, optional
`"models": ["<transformer id>", "<classical id>"]` (default: latest of each family), `top_k`, `threshold`, `segment`.
Segmentation runs **once** (`segment.segment_contract`), then each model's `backend.predict_proba` scores the same
provisions, so clause indexes align. Response:

```json
{"input": {"num_clauses": 13},
 "models": [{"id": "ledgar-qwen3-0.6b-v1", "family": "transformer", "latency_ms": 410, "document": [...], "clauses": [...]},
            {"id": "ledgar-tfidf-lr-v1", "family": "classical", "latency_ms": 12, "document": [...], "clauses": [...]}],
 "agreement": {"clauses_agree": 11, "rate": 0.846, "per_clause": [true, ...]}}
```
Errors use the existing OpenAI envelope; 503 when either family has no trained model (with which one is missing).

### 5. Playground (`api/static/playground.html`)

* Model selector gains **"Compare: transformer + classical"**, selected by default when `/v1/models` lists both
  families; otherwise the current single-model behaviour.
* Compare view: per clause, two rows of chips (one per model, labelled) with an agree/disagree marker; document-level
  grouped bars (two colours, legend with model ids); latency per model in the meta line; raw response.
* Header links to `/models/compare`, `/models/transformer`, `/models/classical`.
* `scripts/08_record_playground.py` records the compare flow (new frames: "Both models side by side"); the GIF is
  re-recorded after training.

### 6. Model pages (`scripts/10_model_pages.py`, `docs/models/`, served at `/models/*`)

Generated from artifacts only (no network), standalone HTML in the `changelog.html` style with light/dark CSS, charts
as PNGs from matplotlib (already pinned) in `docs/models/assets/`, and all numbers also written to `docs/models/data/*.json`.

* `transformer.html` (latest transformer version; a table lists every version): description and architecture
  (Qwen3-0.6B-Base config: 28 layers, 1,024 hidden, GQA 16/8, 78-way head), training settings from `config.yaml`,
  compute and time from `telemetry/meta.json` and `events.jsonl`, **loss curve** (steps.jsonl), **validation accuracy /
  top-3 / macro-F1 per evaluation** (evals.jsonl), test metrics (`eval.json`), per-class F1 bar chart, top-20 confusions,
  calibration before/after, dataset and label-grouping summary, links to the three GIFs.
* `classical.html`: pipeline description (features, n-grams, feature count, classifier, C), candidate results table
  (LR grid, SVM, NB: validation accuracy, macro-F1, fit time), **loss**: training and validation log loss of the
  served model (and per candidate), **learning curve** chart (validation accuracy and log loss vs training-set
  size), test metrics, per-class F1, top-20 confusions, calibration, **top phrases per clause type**
  (explainability), model size and CPU latency.
* Navigation: `changelog.html`, the README and the playground link to the three pages; the pages link to each other.
* `compare.html`: side-by-side metrics table (accuracy, top-3, macro-F1, ECE, training time and hardware, inference
  latency, model size, parameters, explainability, dependencies), per-class F1 difference chart (where each wins),
  **agreement analysis on the 10k test set** (agree rate, accuracy when they agree vs disagree, oracle "either right"),
  top confusions each model makes that the other doesn't, and a recommendation paragraph.
* `api/routes/pages.py`: `GET /models/{compare|transformer|classical}` serves the generated files (404 with "run
  `make model-pages`" if absent); `/models/assets/*` static.

### 7. Make targets and CLI

* `make classical` = `scripts/classical/01_train.py` + `02_evaluate.py` + `make model-pages` + bump of the classical
  config version. `make classical-evaluate`, `make model-pages`, `make compare FILE=...` (`scripts/11_compare.py`).
* `make models` lists both families (new `family` column); `make serve` serves both.
* Docker: `CLASSICAL_RUNS_DIR` env, compose mounts already cover `./artifacts`; `docs/models` copied into the image.

### 8. Tests (all without a trained model, GPU or network)

* Registry: family default, `latest` unchanged with a classical model present, `latest-classical`, base-name resolution.
* `SklearnBackend`: tiny pipeline fitted in-test on `conftest` data; `predict_proba` shape/ordering, temperature.
* `/v1/compare`: `StaticModelStore` with two stub predictors of different families; alignment of clause indexes,
  agreement counts, 503 when a family is missing; `/v1/models` family field.
* Pages generator: synthetic run dirs (transformer-like and classical-like) → three HTML files and charts exist, numbers in `data/*.json`.
* Version/changelog/rules checks extended to the classical config. Existing 77 tests must keep passing.

### 9. Documentation, rules, memory

* README: "Classical model" section (how it is built, how to train, measured numbers), the comparison table with the
  measured results, `/v1/compare` with a real sample response, playground compare description, pages links,
  layout and config reference updates.
* `docs/TELEMETRY.md`: classical telemetry files. `AGENTS.md` model rules: two model families, which is the API
  default, classical versioning in `configs/classical/`; `make rules` regenerates the assistant files.
* `CHANGELOG.md` (+ `make changelog-html`), `MEMORY.md` (decisions, measured numbers, gotchas).

## Verification

1. `make test` — all existing tests plus the new ones pass.
2. `make classical` — creates `artifacts/classical/runs/ledgar-tfidf-lr-v1/`, prints validation results for all
   candidates, writes `eval.json`; `make models` shows both families; classical config `model_version` becomes 2.
3. `shasum -a 256 -c artifacts/before_v2.sha256` — all 78 transformer/demo/data files unchanged.
4. `make serve` → `curl /health` (both latest ids), `curl /v1/models` (family field), `curl -X POST /v1/compare`
   with `demo/sample_contract.txt` (two models, aligned clauses, agreement), `/v1/chat/completions` without `model`
   still answers with `ledgar-qwen3-0.6b-v1`.
5. Playground in the browser: compare mode default, both columns, agreement markers; single-model selection still works.
6. `make model-pages` → open `docs/models/compare.html`, `transformer.html`, `classical.html` in the browser; served
   copies at `/models/*` match.
7. Docker: build, run with `./artifacts` mounted, `/v1/compare` works on CPU with both models.
8. `scripts/08_record_playground.py` re-records the playground GIF including the compare step.

## Expected outcome

| | Classical (TF-IDF + LR) | Transformer v1 (measured) |
|---|---|---|
| Test accuracy | about 91% (to be measured) | 92.5% |
| Training | 20-30 min CPU | 9 h 57 min GPU |
| Inference per clause | about 1 ms CPU | about 50 ms CPU / faster on GPU |
| Model size | about 100 MB | 1.2 GB |
| Explainable | top phrases per clause type | no |
