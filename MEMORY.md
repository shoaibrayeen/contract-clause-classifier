# MEMORY.md

Project memory: decisions, measurements and gotchas that are not obvious from the code. Agents and humans should read this before larger changes and append to it when they learn something new. Newest entries at the top of each section. Dates are absolute.

## Owner preferences

- Training is started by the owner, not by agents. Agents provide the exact command (`make train` / `make all`) and verify changes with `make sanity`. (2026-10-11)
- Use stable, exactly pinned library versions and Python 3.12. (2026-10-11)
- Training must print progress live in the terminal and save every step for later visualisation (charts, GIF, video). (2026-10-11)
- The API must be OpenAI-compatible, use FastAPI, have `/health` and a paginated dataset endpoint (10 per page), and ship with Docker. (2026-10-11)
- Training must be machine-independent; the owner's laptop (Apple M4 Pro, 48 GB) must be able to train and serve. (2026-10-11)
- Rendered visuals go in `demo/`. Plans go in `plans/`. (2026-10-11)

## Design decisions

- **Dataset: LEDGAR** via LexGLUE parquet (`coastalcph/lex_glue`, config `ledgar`, revision `refs/convert/parquet`; fallback mirror `lighteval/lexglue` via the datasets-server API). CUAD was explicitly excluded by the owner. LEDGAR is single-label per provision; multi-label output comes from segmentation + top-k. (2026-10-11)
- **Model: Qwen3-0.6B-Base + LoRA** (r=16, alpha=32, all attention and MLP projections) with `modules_to_save=["score"]`. Base rather than instruct for a classification head. (2026-10-11)
- **PyTorch, not MLX.** MLX was the first choice but is Mac-only; the owner then required machine-independent training. (2026-10-11)
- **Classification head, not generation.** A softmax head gives real, calibratable confidences; generating JSON would not. (2026-10-11)
- **Serving uses a merged model** (`merge_and_unload`, bf16 safetensors) so the API and Docker image do not need `peft`. (2026-10-11)
- **Temperature scaling** is fitted on validation by `scripts/03_evaluate.py` and stored as `calibration.json` next to the model; `HFBackend` applies it. (2026-10-11)
- **Document-level aggregation** = max confidence per label across provisions, kept if ≥ threshold. A single short clause can therefore yield an empty `document` list while `clauses[0].predictions` still has the top-1. (2026-10-11)
- **In-training eval uses a fixed 3,000-example validation subset** (`train.eval_limit`) to keep evals around a minute; final metrics use the full splits. (2026-10-11)
- Package import name is `classification` (folder `src/classification`); distribution name is `contract-clause-classifier`. (2026-10-11)

## Measurements (Apple M4 Pro, 48 GB, MPS, torch 2.14.1)

- LoRA training: about 990 tokens/s, about 4.4 s per optimizer step (32 examples), so 2 epochs (3,750 steps) take about 5 hours including evals. (2026-10-11)
- Micro-batch 8: MPS driver memory about 35 GB and system swap nearly full; steps slowed from 3.7 s to 8 s. Micro-batch 4 with accumulation 8: about 19.5 GB, same throughput. (2026-10-11)
- bf16 autocast on MPS: no faster than fp32 (1.01 s vs 0.99 s per 8×160 batch), so `precision: auto` uses fp32 on MPS. (2026-10-11)
- LEDGAR token lengths with the Qwen3 tokenizer: mean 138, median 105, p90 292; 1.9% exceed 512 and are truncated. (2026-10-11)
- Probe snapshot (493 examples, all hidden states): 120 s unsorted vs 24 s length-sorted. (2026-10-11)
- Sanity run (256 examples, 64 steps): val accuracy 0.8% → 17.6%, proving the loop learns; training loss on a 256-example subset reaches about 0.0002 when run longer. (2026-10-11)

## Gotchas

- `torch.mps` caching allocator grows with every new batch shape. `train.empty_cache_steps` (25) calls `torch.mps.empty_cache()`; it helps only briefly, so batch size is the real lever. (2026-10-11)
- PEFT adapter checkpoints do not store the base config; loading with `AutoPeftModelForSequenceClassification` rebuilds a 2-label head and fails. `model.load_trained` builds the base with `num_labels` from `labels.json` first. (2026-10-11)
- `save_pretrained` writes safetensors with mode 0600; Docker's non-root `app` user then cannot read them. `model.make_readable` chmods exports to 0644. (2026-10-11)
- Docker on this machine is **Colima**, which only shares `~/sirion_space`. Bind mounts from `~/my_space/...` arrive empty. Use `docker cp` or add the folder to Colima `mounts:`; do not edit the owner's Colima config without asking. (2026-10-11)
- Rich `Live` renders nothing useful when stdout is not a TTY; the trainer prints a plain line every `log_steps` instead. (2026-10-11)
- The segmenter must not merge short *numbered* sections (e.g. "11. Counterparts. ...") into neighbours: they are real provisions. Only unnumbered fragments under `min_words` are merged. (2026-10-11)
- `transformers` 5.x uses `dtype=` (not `torch_dtype=`) in `from_pretrained`. (2026-10-11)
- HF Hub requests log at INFO via `httpx`; `scripts/02_train.py` lowers those loggers to WARNING. (2026-10-11)
