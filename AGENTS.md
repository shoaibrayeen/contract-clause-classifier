# AGENTS.md

Guidance for coding agents working in this repository. Humans: start with [README.md](README.md).

This file is the **single source of truth** for agent rules. Each tool reads it through its own entry point:

| Agent | Reads | Notes |
|---|---|---|
| OpenAI Codex (CLI and cloud) | `AGENTS.md` (this file) | Codex loads it automatically from the repo root. |
| Claude Code | [`CLAUDE.md`](CLAUDE.md) | Imports this file with `@AGENTS.md` and adds Claude-specific notes. |
| Cursor | [`.cursor/rules/project.mdc`](.cursor/rules/project.mdc) (always on) and [`.cursor/rules/python-style.mdc`](.cursor/rules/python-style.mdc) (Python files) | Both point back here. |
| Any other agent | `AGENTS.md` | Follow the rules below. |

Keep the three entry points in sync: change the rules **here**, and only add tool-specific notes in the other files.

## What this project is

A contract clause-type classifier. Qwen3-0.6B-Base is fine-tuned with LoRA plus a 100-way classification head on LEDGAR (80k SEC contract provisions, 100 clause types). A contract is split into provisions, each is classified, and the API returns `[{name, confidence}]` per provision and per document. Serving is an OpenAI-compatible FastAPI. Training records detailed telemetry for later visualisation.

Read before larger changes: [plans/PLAN.md](plans/PLAN.md) (design), [MEMORY.md](MEMORY.md) (decisions and gotchas), [docs/TELEMETRY.md](docs/TELEMETRY.md) (telemetry schema).

## Commands

```bash
uv sync                                   # install (Python 3.12, exact pins from uv.lock)
make test                                 # pytest; stub model, no GPU or download, ~5 s
make data                                 # download + tokenize LEDGAR (idempotent)
make sanity                               # 5-min training smoke test (run name "sanity")
make train / make resume                  # full training (hours) - only when the user asks
make evaluate                             # calibrate + test metrics for artifacts/best
make gif                                  # render telemetry to GIF
make serve                                # API on :8000
uv run python scripts/04_predict.py --text "..."   # one-off prediction
```

Always run Python through `uv run` (system Python is 3.9 and lacks the deps).

## Layout

| Path | Role |
|---|---|
| `src/classification/` | the package (import name `classification`) |
| `src/classification/api/` | FastAPI app factory `create_app`, schemas, routes |
| `scripts/0N_*.py` | numbered pipeline steps; thin Typer CLIs over the package |
| `configs/ledgar_qwen3_0.6b.yaml` | single source of training settings; mirrors dataclasses in `config.py` |
| `tests/` | pytest; `conftest.py` provides `StubBackend`, `predictor`, `tiny_data` fixtures |
| `artifacts/runs/<run>/` | training outputs (git-ignored); `artifacts/best` symlink → `runs/<run>/merged` |
| `data/` | raw parquet + tokenised cache (git-ignored) |
| `demo/` | sample contract and rendered GIFs (committed) |

## Rules

1. **Do not start full training** (`make train`, `scripts/02_train.py` without `--limit`) unless the user explicitly asks. It runs for hours and uses ~20 GB of accelerator memory. Use `make sanity` or `--limit 256` to verify training changes.
2. **Keep training device-agnostic.** No CUDA-only or MPS-only code paths without a fallback. Device and precision come from `config.resolve_device` / `resolve_autocast_dtype`.
3. **Dependencies are pinned exactly** in `pyproject.toml` (stable releases only). Add or upgrade with `uv add pkg==X.Y.Z`, keep `uv.lock` in sync, and note it in `CHANGELOG.md`. Runtime deps must stay light: training-only packages go in the `train` group, plotting in `viz`, test tools in `dev`.
4. **The API must stay OpenAI-compatible.** `POST /v1/chat/completions` returns a standard `chat.completion` object; predictions are a JSON string in `choices[0].message.content`; errors use `{"error": {message, type, param, code}}` (see `api/errors.py`). `tests/test_api.py::test_openai_sdk_compat` must keep passing.
5. **Output contract:** predictions are lists of `{"name": str, "confidence": float}`, sorted by confidence, top-1 always kept. Changing this shape is a breaking change.
6. **Telemetry is append-only** and its schema is documented in `docs/TELEMETRY.md`. If you add or change a field, update that file in the same change.
7. **Config changes:** add the field to the dataclass in `config.py` with a default, add it to the YAML with a comment, and mention it in the README configuration table if user-facing.
8. **Keep runtime imports lazy for training-only packages** (`peft`, `sklearn`): the API and Docker image do not install them.
9. **Tests:** add or update tests for any behaviour change; `make test` must pass. Tests must not download models or need a GPU.
10. Update `CHANGELOG.md` (Unreleased section) for user-visible changes.
11. **Always update memory after completing a task.** Before you report a task as done, append to [`MEMORY.md`](MEMORY.md): any decision you made and why, any measurement you took, and any gotcha you hit (with today's date). If nothing new was learned, say so in your final message. This keeps the next agent, or the next session, from rediscovering the same things.
12. Finish what you start: a task is complete only when `make test` passes, the docs that describe the changed behaviour (README, docs/, CHANGELOG.md) are updated, and MEMORY.md has been updated (rule 11).

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
* Saved safetensors are created 0600; `model.make_readable` fixes permissions so Docker's non-root user can read them.
* Colima only shares folders listed in `~/.colima/default/colima.yaml`; bind mounts from elsewhere arrive empty.
