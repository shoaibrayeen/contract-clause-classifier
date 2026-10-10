# MEMORY.md

Project memory: decisions, measurements and gotchas that are not obvious from the code. Agents and humans should read this before larger changes and append to it when they learn something new. Newest entries at the top of each section. Dates are absolute.

## Owner preferences

- Label grouping must be done like a legal analyst: merge only headings that denote the same legal concept, verified on the rows; statistics alone are not enough. (2026-10-11)
- The dataset is committed to the repo so others get identical data; every file in `data/raw/` must credit its exact download URL, date, checksum and license. (2026-10-11)
- Wants validation that there are no workarounds and that training is set up for the highest achievable accuracy; accepts evidence-based limits. (2026-10-11)

- Target accuracy is 96-99%. Told the owner (2026-10-11) that 99% top-1 is not reachable on LEDGAR (published SOTA about 88% on 100 labels) and that the default config is expected to reach about 94-96% top-1 on 72 merged clause types, 98%+ top-3. Do not overpromise; quote measured numbers. (2026-10-11)
- Every training run must be a new model version (`<run_name>-v<N>`, starting at v1, bumped after `make all` completes); all versions are kept; chat completions default to the latest, or the `model` in the payload. (2026-10-11)
- The owner starts training; never interrupt or alter a running `make all`. While it runs, scripts it will call later (`03_evaluate.py --config --model-dir`, `06_render_gif.py --run <path>`) must keep their CLI, and `pyproject.toml` / `uv.lock` must not change (uv run would re-sync). (2026-10-11)
- Console output shows project-relative paths (`contract-clause-classifier/...`), never `/Users/...`. Dashboard shows 10 rows. A browser playground is required. (2026-10-11)
- Rules for every AI assistant must exist and stay in sync (AGENTS.md is the source; `make rules` generates the rest). (2026-10-11)

- After finishing any task, update this file and CHANGELOG.md before reporting done (AGENTS.md rule 11). Agents for Claude, Codex and Cursor must all follow the same rules, kept in AGENTS.md. (2026-10-11)
- README must show a sample response next to every curl, state what data the model is trained on, and carry a license plus the owner's contact (shoaibrayeen.me@gmail.com, https://shoaibrayeen.github.io/). (2026-10-11)

- Training is started by the owner, not by agents. Agents provide the exact command (`make train` / `make all`) and verify changes with `make sanity`. (2026-10-11)
- Use stable, exactly pinned library versions and Python 3.12. (2026-10-11)
- Training must print progress live in the terminal and save every step for later visualisation (charts, GIF, video). (2026-10-11)
- The API must be OpenAI-compatible, use FastAPI, have `/health` and a paginated dataset endpoint (10 per page), and ship with Docker. (2026-10-11)
- Training must be machine-independent; the owner's laptop (Apple M4 Pro, 48 GB) must be able to train and serve. (2026-10-11)
- Rendered visuals go in `demo/`. Plans go in `plans/`. (2026-10-11)

## Design decisions

- **Final label grouping (2026-10-11, legal-analyst review): 19 groups, 41 labels merged, 78 clause types** (replaces the earlier 21 groups / 72 types). Criterion = legal test on sample rows AND statistical test on an independent TF-IDF+LR baseline (87.79% on the full 10k validation; ~8% pairwise mutual confusion or ~40% of a member's errors). Verified by regex on all rows: 75% of "Warranties" rows are represents-and-warrants language (5% product warranties); 75% of "Authorizations" rows are due-authorization reps (8% permits); both waiver labels are no-waiver-by-conduct boilerplate (7% substantive waivers); 77%/48% of Approvals/Consents rows are the consents-and-approvals rep; only 30% of "Taxes" rows concern withholding. Rejected on legal grounds or evidence: Miscellaneous/General (0.9% mutual), Remedies/Specific Performance/Enforcements (3-6%), Taxes vs Withholdings (5.3%), Fees vs Expenses (5.8%), Venues vs jurisdiction labels (0-6%). Effect on the stopped run's step-1,000 confusion matrix: 82.97% -> 88.87% (step 500: 79.6% -> 85.8%); TF-IDF baseline 87.79% -> 91.20%. Rarest class 23 -> 125 train rows; probe set 390. (2026-10-11)
- **Expected accuracy for the default config: about 92-95% top-1** on 78 clause types (fully trained 100-label models reach ~88%; merge +3 to +6; full FT +1 to +2); 98%+ top-3. Communicated to the owner that 96-99% top-1 is not supported by LEDGAR. (2026-10-11)
- **Every model saves `label_space.json`** (names, raw names, raw->id map, groups, source) in checkpoints and `merged/`; `scripts/03_evaluate.py` scores a model against its own label space via `model.label_space_for_model`, so changing the groups file never breaks evaluation of older versions. Models without the file are accepted only if their labels equal the current space or the original 100; otherwise a clear error. Found when evaluating the 72-type sanity model against the 78-type config ("Target 73 is out of bounds"). (2026-10-11)
- **Dataset committed**: `data/raw/ledgar/*.parquet` + `labels.json` + per-file provenance README (exact HF parquet URLs, SHA-256, date, CC BY 4.0, BibTeX); `data/raw/README.md` sets the policy; `data/cache/` stays ignored; the Docker image bakes `data/raw/ledgar` in so `/v1/dataset` works without a mount. (2026-10-11)
- `build_model` passes `ignore_mismatched_sizes` only when the base already carries a classification head of another size (e.g. retraining from a fine-tuned model); for a causal-LM base nothing is ignored. (2026-10-11)

- **Label grouping (100 -> 72 clause types).** Evidence: first run (LoRA r=16, raw labels, stopped at step 1,288 by the owner) had 82.97% val accuracy at step 1,000; 38% of its 511 errors (3,000-example val subset) were between synonym labels (Applicable Laws->Governing Laws 18, Tax Withholdings->Withholdings 17, Representations->Warranties 17, Definitions->Defined Terms 16, Integration->Entire Agreements 13, ...). Merging 49 labels into 21 groups (`configs/ledgar_label_groups.yaml`) scores the same checkpoint at 90.03% (step 500: 79.6% -> 87.4%). Exact-duplicate label noise is negligible (3 texts). The rarest class grows from 23 to 125 train examples. (2026-10-11)
- **Default training is full fine-tuning** (lr 2e-5, warmup 6%, 3 epochs, label smoothing 0.1, best checkpoint by accuracy). Measured on M4 Pro with batch 4 random LEDGAR samples: LoRA r16 1.17 s / 19.3 GB, LoRA r64 1.19 s / 19.9 GB, full FT 1.37 s / 22.5 GB per micro-batch. Full FT costs ~17% more time for an expected 1-2 points. Estimated ~2.6 h/epoch, ~8 h for 3 epochs. (2026-10-11)
- **Base name changed to `ledgar-qwen3-0.6b`** (was `ledgar-qwen3-0.6b-lora`) because the default is no longer LoRA. First version will be `ledgar-qwen3-0.6b-v1`. The owner's stopped LoRA run stays in `artifacts/runs/ledgar-qwen3-0.6b-lora` (unfinished; 100 labels) as the evidence for the analysis above; it is not servable. (2026-10-11)
- **Model registry = the run folders.** No index file: a run is complete when `merged/config.json` exists; `merged/model_info.json` holds id, version, created, metrics. Latest = newest completed *versioned* model; unversioned runs (sanity) only when no versioned model exists. `artifacts/best` -> latest. (2026-10-11)
- **`model_version` lives in the training YAML** (next version to train). `scripts/07_models.py bump` sets it to max(current, highest completed + 1), so it never skips a failed run and is idempotent. Planning skips completed versions and refuses unfinished ones unless `--resume` / `--restart`. (2026-10-11)
- **API serves all versions** via `api/models.py` `ModelStore`: lazy load, LRU (`MAX_LOADED_MODELS=2`), rescan every 5 s so new versions appear without restart. `model` omitted / "latest" / `MODEL_ID` alias -> latest; version id -> that; base name -> its latest; else 404 listing ids. Response `model` = concrete version. (2026-10-11)
- **Splitting into provisions is on by default** (API, CLI, playground). The old ">350 words" rule left short multi-section contracts as one clause; a single clause still yields one provision. (2026-10-11)
- **Top-3 accuracy** is reported in training evals, `make evaluate`, `make models` and `/v1/models`. (2026-10-11)
- **Agent rules:** AGENTS.md holds a core-rules block between markers; `scripts/sync_agent_rules.py` writes Copilot, Gemini, Windsurf, Cline, Roo, Continue, Amazon Q, Junie, Zed, Aider and Cursor project files; `tests/test_agent_rules.py` fails if they drift. (2026-10-11)

- **License: Apache-2.0** (copyright 2026 Shoaib Rayeen). Matches Qwen3-0.6B-Base and the Hugging Face stack, permissive, includes a patent grant. LexGLUE/LEDGAR is CC BY 4.0 (attribution required), derived from public SEC EDGAR filings; the README's license table records all components. (2026-10-11)
- **README sample responses come from the 5-minute sanity model** (low confidences, e.g. Governing Laws 0.81 on a clear clause, 0.56 on the sample contract). They were captured by serving on CPU on port 8011; regenerate them after a full training run. (2026-10-11)
- `changelog.html` is a hand-written standalone page mirroring CHANGELOG.md; update both. (2026-10-11)

- **Dataset: LEDGAR** via LexGLUE parquet (`coastalcph/lex_glue`, config `ledgar`, revision `refs/convert/parquet`; fallback mirror `lighteval/lexglue` via the datasets-server API). CUAD was explicitly excluded by the owner. LEDGAR is single-label per provision; multi-label output comes from segmentation + top-k. (2026-10-11)
- **Model: Qwen3-0.6B-Base** with a sequence-classification head; originally LoRA r=16 (superseded the same day by full fine-tuning, see above). Base rather than instruct for a classification head. (2026-10-11)
- **PyTorch, not MLX.** MLX was the first choice but is Mac-only; the owner then required machine-independent training. (2026-10-11)
- **Classification head, not generation.** A softmax head gives real, calibratable confidences; generating JSON would not. (2026-10-11)
- **Serving uses a merged model** (`merge_and_unload`, bf16 safetensors) so the API and Docker image do not need `peft`. (2026-10-11)
- **Temperature scaling** is fitted on validation by `scripts/03_evaluate.py` and stored as `calibration.json` next to the model; `HFBackend` applies it. (2026-10-11)
- **Document-level aggregation** = max confidence per label across provisions, kept if ≥ threshold. A single short clause can therefore yield an empty `document` list while `clauses[0].predictions` still has the top-1. (2026-10-11)
- **In-training eval uses a fixed 3,000-example validation subset** (`train.eval_limit`) to keep evals around a minute; final metrics use the full splits. (2026-10-11)
- Package import name is `classification` (folder `src/classification`); distribution name is `contract-clause-classifier`. (2026-10-11)

## Measurements (Apple M4 Pro, 48 GB, MPS, torch 2.14.1)

- **Full fine-tune path verified end to end** on the real Qwen3-0.6B-Base (320 train rows, 10 optimizer steps of 32): loss 6.16 -> 3.75, 5.4 s/step, 840 tok/s, peak 25.0 GB accelerator memory, gradients clipped at every step; checkpoint-best 2.2 GB, checkpoint-last 6.7 GB (fp32 weights + AdamW state), merged 1.1 GB -> ~10 GB per version. Projection: 5,625 steps ~8.4 h + 11 evals x ~3 min -> ~9 h for 3 epochs. (2026-10-11)
- Stopped LoRA run (lr 2e-4): loss by step 1-100 3.92, 100-300 1.11, 300-600 0.79, 600-900 0.67, 900-1288 0.61; batch accuracy 0.25 -> 0.83; grad norm 20 -> 5 (clip 1.0 active 100% of steps); eval on 3,000 rows took 159-177 s. 10% of late examples still had loss > 2, dominated by General, Applicable Laws, Authority, Payments, Miscellaneous, Enforcements; hard examples were not longer (2.6% vs 1.9% truncated), so max_len 512 stays. (2026-10-11)
- TF-IDF (1-2 grams, 300k features) + logistic regression: 87.79% on the full 10k validation, 176 s to train; useful as an independent, model-free check of label noise. (2026-10-11)

- LoRA training: about 990 tokens/s, about 4.4 s per optimizer step (32 examples), so 2 epochs (3,750 steps) take about 5 hours including evals. (2026-10-11)
- Micro-batch 8: MPS driver memory about 35 GB and system swap nearly full; steps slowed from 3.7 s to 8 s. Micro-batch 4 with accumulation 8: about 19.5 GB, same throughput. (2026-10-11)
- bf16 autocast on MPS: no faster than fp32 (1.01 s vs 0.99 s per 8×160 batch), so `precision: auto` uses fp32 on MPS. (2026-10-11)
- LEDGAR token lengths with the Qwen3 tokenizer: mean 138, median 105, p90 292; 1.9% exceed 512 and are truncated. (2026-10-11)
- Probe snapshot (493 examples, all hidden states): 120 s unsorted vs 24 s length-sorted. (2026-10-11)
- Sanity run (256 examples, 64 steps): val accuracy 0.8% → 17.6%, proving the loop learns; training loss on a 256-example subset reaches about 0.0002 when run longer. (2026-10-11)

## Gotchas

- Changing `configs/ledgar_label_groups.yaml` changes the label space of *new* models only; evaluating an older model needs its saved `label_space.json` (now written automatically). The sanity model trained before this fix had the file reconstructed by hand. (2026-10-11)
- A random-init tiny Qwen3 for CPU tests must set `layer_types` to match `num_hidden_layers` or `save_pretrained` fails validation. (2026-10-11)

- The live dashboard re-printed itself ("repeated boxes") because the panel was taller than the terminal; rich cannot redraw above the top of the screen. The panel now computes rows from the terminal height (`_rows_that_fit`, chrome = 13 lines) and Live uses `vertical_overflow="crop"`. `tests/test_dashboard.py` checks heights 20-60. (2026-10-11)
- The tokenisation cache stores original LEDGAR label ids; `data.get_label_space` maps them, so changing groups never needs re-tokenising. Old caches stay valid. (2026-10-11)
- A random-init tiny Qwen3 for CPU tests needs `layer_types` resized to `num_hidden_layers` or `save_pretrained` fails validation; `build_model` uses `ignore_mismatched_sizes=True` so a base that already has a different-size head works. (2026-10-11)
- zsh does not word-split `$VAR`, so `T="cmd args"; $T` runs a command literally named "cmd args" (exit 127). Use a shell function. (2026-10-11)
- `docker cp src ctr:/dest/` copies src *as* /dest when /dest does not exist; create the parent first to get /dest/src. (2026-10-11)
- Modules loaded with `importlib.util.spec_from_file_location` must be put in `sys.modules` before `exec_module` if they define dataclasses. (2026-10-11)

- `pyproject.toml` declares `license-files = ["LICENSE"]`, so the Dockerfile must `COPY LICENSE` before `pip install .` or the build fails. (2026-10-11)

- `torch.mps` caching allocator grows with every new batch shape. `train.empty_cache_steps` (25) calls `torch.mps.empty_cache()`; it helps only briefly, so batch size is the real lever. (2026-10-11)
- PEFT adapter checkpoints do not store the base config; loading with `AutoPeftModelForSequenceClassification` rebuilds a 2-label head and fails. `model.load_trained` builds the base with `num_labels` from `labels.json` first. (2026-10-11)
- `save_pretrained` writes safetensors with mode 0600; Docker's non-root `app` user then cannot read them. `model.make_readable` chmods exports to 0644. (2026-10-11)
- Docker on this machine is **Colima**, which only shares `~/sirion_space`. Bind mounts from `~/my_space/...` arrive empty. Use `docker cp` or add the folder to Colima `mounts:`; do not edit the owner's Colima config without asking. (2026-10-11)
- Rich `Live` renders nothing useful when stdout is not a TTY; the trainer prints a plain line every `log_steps` instead. (2026-10-11)
- The segmenter must not merge short *numbered* sections (e.g. "11. Counterparts. ...") into neighbours: they are real provisions. Only unnumbered fragments under `min_words` are merged. (2026-10-11)
- `transformers` 5.x uses `dtype=` (not `torch_dtype=`) in `from_pretrained`. (2026-10-11)
- HF Hub requests log at INFO via `httpx`; `scripts/02_train.py` lowers those loggers to WARNING. (2026-10-11)
