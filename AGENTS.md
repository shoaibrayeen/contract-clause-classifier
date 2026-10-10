# AGENTS.md

Guidance for coding agents working in this repository. Humans: start with [README.md](README.md).

This file is the **single source of truth** for agent rules. Each tool reads it through its own entry point:

| Agent | Reads | Notes |
|---|---|---|
| OpenAI Codex (CLI and cloud) | `AGENTS.md` (this file) | Codex loads it automatically from the repo root. |
| Claude Code | [`CLAUDE.md`](CLAUDE.md) | Imports this file with `@AGENTS.md` and adds Claude-specific notes. |
| Cursor | [`.cursor/rules/project.mdc`](.cursor/rules/project.mdc) (always on) and [`.cursor/rules/python-style.mdc`](.cursor/rules/python-style.mdc) (Python files) | Both point back here. |
| GitHub Copilot | [`.github/copilot-instructions.md`](.github/copilot-instructions.md) | Generated. |
| Gemini CLI / Gemini Code Assist | [`GEMINI.md`](GEMINI.md) | Generated. |
| Windsurf | [`.windsurf/rules/project.md`](.windsurf/rules/project.md) | Generated (always on). |
| Cline | [`.clinerules/project.md`](.clinerules/project.md) | Generated. |
| Roo Code | [`.roo/rules/project.md`](.roo/rules/project.md) | Generated. |
| Continue | [`.continue/rules/project.md`](.continue/rules/project.md) | Generated (always applied). |
| Amazon Q Developer | [`.amazonq/rules/project.md`](.amazonq/rules/project.md) | Generated. |
| JetBrains Junie | [`.junie/guidelines.md`](.junie/guidelines.md) | Generated. |
| Zed | [`.rules`](.rules) | Generated. |
| Aider | [`CONVENTIONS.md`](CONVENTIONS.md), loaded by [`.aider.conf.yml`](.aider.conf.yml) | Generated. |
| Any other agent | `AGENTS.md` | Follow the rules below. |

The generated files contain the core rules summary below and point back here. After editing the summary, run `make rules` to regenerate them; `tests/test_agent_rules.py` fails if any of them is out of sync.

Change rules **here** only. `CLAUDE.md` and the Cursor `.mdc` files may add tool-specific notes; everything else is generated.

## Core rules (summary)

<!-- core-rules:start -->
These are the rules every AI assistant must follow in this repository. The full rules, with reasons, are in `AGENTS.md`.

1. **Read first:** `AGENTS.md` (rules), `MEMORY.md` (decisions, measurements, gotchas), `plans/PLAN.md` (design), `docs/TELEMETRY.md` (telemetry schema).
2. **Run Python with `uv run`** (Python 3.12, exact pins from `uv.lock`). Add dependencies only as `uv add pkg==X.Y.Z` (stable releases).
3. **Never start full training** (`make all`, `make train`) unless the owner asks; give the command instead. Verify training changes with `make sanity`.
4. **Model versions are immutable:** each run creates `<train.run_name>-v<model_version>`; never overwrite, rename or delete a version; `model_version` is bumped only by `make bump` after success.
5. **The API stays OpenAI-compatible** and predictions keep the shape `[{"name", "confidence"}]`; requests without `model` use the latest version.
6. **Keep training device-agnostic** (CUDA, Apple MPS, CPU via `config.resolve_device`).
7. **Tests must pass** (`make test`) and must not download models or need a GPU; add tests for every behaviour change.
8. **Accuracy claims need evidence** from `make evaluate` or telemetry, naming the split, the label set (72 merged or 100 original) and top-1 vs top-3.
9. **Security and privacy:** never commit secrets or `.env`; never log or persist contract text; keep `/v1/*` behind `API_KEY` when set; never commit `artifacts/` or `data/`.
10. **Do not commit or push** unless the owner asks.
11. **Docs move with code:** update README (with sample responses), `docs/`, and the agent files in the same change.
12. **Finish properly:** update `CHANGELOG.md` and `changelog.html` (kept in sync) and append dated decisions, measurements and gotchas to `MEMORY.md` before reporting a task done.
<!-- core-rules:end -->

## What this project is

A contract clause-type classifier. Qwen3-0.6B-Base is fully fine-tuned with a classification head on LEDGAR (80k SEC contract provisions); LEDGAR's 100 labels are merged into 72 clause types (`configs/ledgar_label_groups.yaml`). Every training run produces a new model version (`<run_name>-v<N>`) and the API serves all versions, defaulting to the latest. A contract is split into provisions, each is classified, and the API returns `[{name, confidence}]` per provision and per document. Serving is an OpenAI-compatible FastAPI. Training records detailed telemetry for later visualisation.

Read before larger changes: [plans/PLAN.md](plans/PLAN.md) (design), [MEMORY.md](MEMORY.md) (decisions and gotchas), [docs/TELEMETRY.md](docs/TELEMETRY.md) (telemetry schema).

## Commands

```bash
uv sync                                   # install (Python 3.12, exact pins from uv.lock)
make test                                 # pytest; stub model, no GPU or download, ~5 s
make data                                 # download + tokenize LEDGAR (idempotent)
make sanity                               # 5-min training smoke test (run name "sanity", unversioned)
make all / train / resume / restart       # full training (hours) - only when the user asks
make models / make next                   # list versions / show the next version name
make evaluate MODEL=latest                # calibrate + test metrics (any version id)
make gif MODEL=latest                     # render a version's telemetry to GIF
make serve                                # API + playground on :8000 (/playground)
uv run python scripts/04_predict.py --text "..."   # one-off prediction
```

Always run Python through `uv run` (system Python is 3.9 and lacks the deps).

## Layout

| Path | Role |
|---|---|
| `src/classification/` | the package (import name `classification`) |
| `src/classification/registry.py` | model versions: naming, discovery, latest, never-overwrite planning, config bump |
| `src/classification/api/` | FastAPI app factory `create_app`, multi-version `ModelStore`, schemas, routes, `static/playground.html` |
| `scripts/0N_*.py` | numbered pipeline steps; thin Typer CLIs over the package |
| `configs/ledgar_qwen3_0.6b.yaml` | single source of training settings incl. `model_version`; mirrors dataclasses in `config.py` |
| `configs/ledgar_label_groups.yaml` | synonym LEDGAR labels merged into 72 clause types |
| `tests/` | pytest; `conftest.py` provides `StubBackend`, `predictor`, `tiny_data` fixtures |
| `artifacts/runs/<run>-v<N>/` | one folder per model version (git-ignored); `merged/` is the served model; `artifacts/best` → latest |
| `data/` | raw parquet + tokenised cache (git-ignored) |
| `demo/` | sample contract and rendered GIFs (committed) |

## Rules

1. **Do not start full training** (`make all`, `make train`, `scripts/02_train.py` without `--limit`) unless the user explicitly asks; give the user the command instead. It runs for hours and uses ~20 GB of accelerator memory. Use `make sanity` or `--limit 256` to verify training changes.
2. **Keep training device-agnostic.** No CUDA-only or MPS-only code paths without a fallback. Device and precision come from `config.resolve_device` / `resolve_autocast_dtype`.
3. **Dependencies are pinned exactly** in `pyproject.toml` (stable releases only). Add or upgrade with `uv add pkg==X.Y.Z`, keep `uv.lock` in sync, and note it in `CHANGELOG.md`. Runtime deps must stay light: training-only packages go in the `train` group, plotting in `viz`, test tools in `dev`.
4. **The API must stay OpenAI-compatible.** `POST /v1/chat/completions` returns a standard `chat.completion` object; predictions are a JSON string in `choices[0].message.content`; errors use `{"error": {message, type, param, code}}` (see `api/errors.py`). `tests/test_api.py::test_openai_sdk_compat` must keep passing.
5. **Output contract:** predictions are lists of `{"name": str, "confidence": float}`, sorted by confidence, top-1 always kept. Changing this shape is a breaking change.
6. **Model versions are immutable.** Never write into a completed run directory (`merged/config.json` exists) except `model_info.json`, `calibration.json` and `eval.json` from evaluation. New training always goes to a new `-v<N>`; `registry.plan_training_run` enforces this. Requests without `model` must keep resolving to the latest version.
7. **Telemetry is append-only** and its schema is documented in `docs/TELEMETRY.md`. If you add or change a field, update that file in the same change.
8. **Config changes:** add the field to the dataclass in `config.py` with a default, add it to the YAML with a comment, and mention it in the README configuration table if user-facing.
9. **Keep runtime imports lazy for training-only packages** (`peft`, `sklearn`): the API and Docker image do not install them.
10. **Tests:** add or update tests for any behaviour change; `make test` must pass. Tests must not download models or need a GPU.
11. **Update the changelog** for every user-visible change, following the [changelog rules](#changelog-rules) below.
12. **Always update memory after completing a task.** Before you report a task as done, append to [`MEMORY.md`](MEMORY.md): any decision you made and why, any measurement you took, and any gotcha you hit (with today's date). If nothing new was learned, say so in your final message. This keeps the next agent, or the next session, from rediscovering the same things.
13. Finish what you start: a task is complete only when `make test` passes, the docs that describe the changed behaviour (README, docs/, CHANGELOG.md) are updated, and MEMORY.md has been updated (rule 12).

## Model rules

How models are trained, named, kept and served. These protect the owner's trained models and keep accuracy claims honest.

1. **One run = one version.** Training without `--run-name` creates `<train.run_name>-v<model_version>`. Never pass a version suffix in `train.run_name`, and never rename or delete a version folder unless the owner asks.
2. **Never overwrite a completed version.** A run directory with `merged/config.json` is final. Only evaluation may add or update `calibration.json`, `eval.json` and `model_info.json` in its `merged/`. Retraining always produces a new version.
3. **Unfinished runs are the owner's decision.** If a version exists but did not finish, do not delete it yourself: tell the owner to run `make resume` or `make restart`.
4. **Bump only after success.** `model_version` is raised by `scripts/07_models.py bump` (the last step of `make all`, `make resume`, `make restart`) and only if the current version completed. Do not edit `model_version` by hand to skip a failed run.
5. **Latest means newest completed versioned model.** API requests without `model` (or with `latest` / the `MODEL_ID` alias) must resolve to it. Unversioned runs such as `sanity` must never become latest while a versioned model exists. `artifacts/best` must always point at latest.
6. **Every served model must be traceable.** `merged/model_info.json` records id, version, base model, method, label count, label groups, best validation scores and test scores. Keep writing it when export or evaluation changes.
7. **Accuracy claims need evidence.** Report accuracy only from `make evaluate` (full test split) or telemetry, and say which split, which label set (72 merged or 100 original) and whether it is top-1 or top-3. Do not promise accuracy the data cannot support; the evidence and ceiling are in README "Accuracy: what to expect" and MEMORY.md.
8. **Changing what the model learns is a new model.** Changes to `data.label_groups`, the label set, `max_len`, the base model or the loss produce models that are not comparable with earlier versions: note them in the changelog and MEMORY.md, and in the README if user-facing.
9. **Label groups need evidence.** Only merge labels that the data shows are synonyms (confusion between them in telemetry, or identical meaning). Record the evidence in the groups file header and MEMORY.md.
10. **Never commit models or data.** `artifacts/` and `data/` stay git-ignored; share models by copying `merged/` folders.

## Changelog rules

1. Every user-visible change goes into `CHANGELOG.md` under `## [Unreleased]`, in the same change, grouped as **Added**, **Changed**, **Fixed**, **Removed**, **Deprecated** or **Security** ([Keep a Changelog](https://keepachangelog.com/en/1.1.0/)).
2. Keep `changelog.html` in sync with `CHANGELOG.md`: same entries, same order. Edit both in the same change.
3. Write entries for the owner, not the code: what changed and why it matters, with the command or file name to use. One idea per bullet.
4. Breaking changes (output shape, API behaviour, config keys, model naming, label set) are marked **Breaking** at the start of the bullet, with what to do about them.
5. New, upgraded or removed dependencies list the exact version.
6. Model-affecting changes (rule 8 above) say how new versions will differ from earlier ones, with measured numbers when available.
7. On release, rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`, bump `version` in `pyproject.toml` and `src/classification/__init__.py`, add a fresh empty `[Unreleased]`, and update the compare links at the bottom. Follow [Semantic Versioning](https://semver.org/): breaking = major, features = minor, fixes = patch.
8. Never rewrite the entries of a released version, except to fix a factual error (and say so).

## Security and privacy rules

1. **No secrets in the repo.** `.env`, API keys, Hugging Face tokens and credentials are never committed or printed. Document new settings in `.env.example` with placeholder values.
2. **Contracts are confidential.** Never log, persist or send request text (API, playground, CLI input) anywhere; log sizes and timings only. Tests use synthetic text.
3. **Keep the API's protections:** `/v1/*` requires `Authorization: Bearer <API_KEY>` when `API_KEY` is set (constant-time comparison in `api/security.py`); `/health` and `/playground` stay open; inputs are capped by `MAX_INPUT_CHARS`.
4. **Docker runs as a non-root user** with a health check; keep both.
5. **Only trusted sources:** models and datasets come from the Hugging Face repos named in the config. Do not add code that downloads or executes anything else without the owner's approval.
6. **Never weaken a check to make a test pass** (auth, validation, never-overwrite guards).

## Git rules

1. Do not commit, push, tag or open pull requests unless the owner asks. Never force-push or rewrite published history.
2. Never commit `artifacts/`, `data/`, `.env`, model files (`*.safetensors`, `*.bin`, `*.pt`) or caches; `.gitignore` covers them, do not weaken it.
3. Commit `uv.lock` together with any `pyproject.toml` change.
4. Commit messages: an imperative summary of 72 characters or less, a blank line, then why the change was made.

## Data rules

1. `data/raw/` holds the downloaded LEDGAR parquet: read-only, never edited. Derived data goes to `data/cache/`.
2. LEDGAR (via LexGLUE) is CC BY 4.0: keep the attribution in the README and in any redistributed model.
3. Label changes go through `configs/ledgar_label_groups.yaml` (see Model rules 8 and 9), never by editing the data.

## Documentation rules

1. The README is the user manual: every command and endpoint is documented with a real sample response; numbers in it name their source.
2. Behaviour changes update the README, `docs/` and the agent files in the same change; telemetry fields are documented in `docs/TELEMETRY.md`.
3. Larger changes start with a plan saved under `plans/`, and the plan gets an implementation-status note when done.
4. Rendered visuals (GIFs, images) go in `demo/`.
5. Console and API output shows project-relative paths (`config.display_path`), never the owner's home directory.

## Style

* Python 3.12, type hints, `from __future__ import annotations`, dataclasses for config, pydantic for API schemas.
* Module docstring at the top of every file explaining its role.
* Paths: resolve relative paths against the project root with `config.resolve_path`, not the CWD.
* Logging via `logging`; user-facing CLI output via `rich`.
* Small, focused functions; no new frameworks without a reason.

## Gotchas (details in MEMORY.md)

* MPS: micro-batch 8 drives the allocator to ~35 GB and swaps on a 48 GB Mac; default is 4 × 8 accumulation. `train.empty_cache_steps` releases cache.
* bf16 autocast is not faster than fp32 on MPS for this model, so `precision: auto` uses fp32 there.
* PEFT adapter dirs have no base config: `model.load_trained` rebuilds the base with the right label count first.
* The tokenisation cache stores LEDGAR's original label ids; `data.get_label_space` maps them on load, so changing label groups never needs re-tokenising.
* The live dashboard must fit the terminal (rich cannot redraw a region taller than the window and leaves copies); `tests/test_dashboard.py` guards this.
* Saved safetensors are created 0600; `model.make_readable` fixes permissions so Docker's non-root user can read them.
* Colima only shares folders listed in `~/.colima/default/colima.yaml`; bind mounts from elsewhere arrive empty.
