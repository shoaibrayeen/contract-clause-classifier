# Training telemetry schema

Every training run writes its full history to `artifacts/runs/<run-name>/telemetry/`. This page documents each file so the data can be turned into charts, GIFs or videos later. The writer is `src/classification/telemetry.py`; the GIF renderer `scripts/06_render_gif.py` is a working example of reading it.

General rules:

* JSONL files are append-only and line-buffered. They can be read while training is running; skip a final line that fails to parse (it may still be being written).
* Every JSONL record has a `time` field (Unix seconds).
* `step` always means **optimizer step** (one step = `batch_size × grad_accum` examples = 32 by default). Step 0 is before any training.
* `--resume` appends to the same files. `meta.json["sessions"]` lists each start.
* Example ids (`ids`, `probe_ids`) are row numbers in the original LEDGAR split parquet (`data/raw/ledgar/<split>.parquet`), so any record can be joined back to its text and label. `GET /v1/dataset?split=train&offset=<id>&limit=1` returns the same row.
* Label ids index into `meta.json["label_names"]` (100 names).

## Directory

```
telemetry/
├── meta.json
├── events.jsonl
├── steps.jsonl
├── layers.jsonl
├── evals.jsonl
├── system.jsonl
├── evals/step_XXXXXX.npz
└── snapshots/step_XXXXXX.npz
```

## meta.json

Written at start (and updated at the end with `summary`).

| Key | Content |
|---|---|
| `run_name` | run name |
| `config` | full resolved config (same as `../config.yaml`) |
| `environment` | python, torch, transformers, peft versions; platform; device; chip and memory on macOS; GPU name and memory on CUDA |
| `label_names` | list of 100 clause-type names |
| `model` | `name`, `num_layers` (28), `hidden_size` (1024), `trainable_params`, `total_params`, `train_examples`, `val_examples`, `steps_per_epoch`, `total_steps`, `device`, `amp` |
| `datasets.train` / `datasets.validation` | `n`, `label_counts` (length 100), `length_hist` {`counts`, `edges`} over 32 bins, `length_mean`, `length_p50`, `length_p90` |
| `datasets.eval_subset_size` | validation examples used for in-training evaluation |
| `probe` | `split` ("validation"), `indices` (row ids), `labels` |
| `sessions` | list of {`started`, `resumed`} |
| `summary` | at the end: `best_macro_f1`, `steps`, `merged` path |

## events.jsonl

Lifecycle events, in order.

| `event` | Extra fields |
|---|---|
| `start` | `step`, `resumed`, plus the `model` info above |
| `snapshot` | `step`, `epoch`, `probe_accuracy`, `n` |
| `eval` | `step`, `epoch`, `improved`, `accuracy`, `micro_f1`, `macro_f1`, `loss` |
| `checkpoint` | `step`, `path`, `best` |
| `early_stop` | `step`, `patience` |
| `done` | `best_macro_f1`, `steps`, `merged` |

## steps.jsonl

One record per optimizer step. This is the "how data moved through training" record.

| Field | Type | Meaning |
|---|---|---|
| `step` | int | optimizer step (1-based) |
| `epoch` | float | fractional epoch at the end of the step |
| `loss` | float | mean cross-entropy over the step's micro-batches |
| `lr` | float | learning rate used for this step |
| `grad_norm` | float | global gradient norm before clipping |
| `clipped` | bool | `grad_norm > max_grad_norm` |
| `step_time_s` | float | wall time of the step (evaluation time excluded) |
| `tokens` | int | real (non-pad) tokens processed |
| `padded_tokens` | int | tokens including padding |
| `padding_ratio` | float | `1 - tokens / padded_tokens` |
| `tokens_per_s` | float | throughput |
| `mem_gb` | float | accelerator memory held by the process (MPS driver or CUDA peak) |
| `ids` | int[] | source row ids of every example in the step (32 by default) |
| `labels` | int[] | their gold label ids |
| `lengths` | int[] | their token lengths |
| `example_loss` | float[] | per-example cross-entropy (unweighted), rounded to 4 decimals |
| `correct` | 0/1[] | whether the model's argmax was right, at the moment it was trained on |
| `batch_accuracy` | float | mean of `correct` |

The batch fields are present when `telemetry.record_batches: true`. Typical uses: loss curves, learning-rate schedule, throughput and padding over time, which clause types were seen when, per-example difficulty, and dataset cartography (an example's loss each time it is seen across epochs).

## layers.jsonl

Every `telemetry.layer_stats_every` steps (default 10).

| Field | Meaning |
|---|---|
| `step` | optimizer step |
| `grad_norm` | object: `layer_00` … `layer_27`, `head`, possibly `embed` / `other` → L2 norm of that group's gradients before clipping |

Use it for a layer-by-step gradient heatmap: how the learning signal flows backwards through the network.

## evals.jsonl

One record per evaluation (every `train.eval_steps`, and at the final step).

| Field | Meaning |
|---|---|
| `step`, `epoch` | when |
| `accuracy`, `micro_f1`, `macro_f1`, `loss` | on the in-training validation subset |
| `eval_time_s`, `n` | evaluation duration and size |
| `per_class_f1` | list of 100 floats (or `null` when a class is absent from the subset), indexed by label id |
| `weight_norm` | object per layer (`layer_00` …, `head`): L2 norm of the trainable weights (LoRA A/B and head) → how far each layer has moved from its starting point |

## evals/step_XXXXXX.npz

| Array | Shape | Meaning |
|---|---|---|
| `confusion` | (100, 100) int32 | rows = true label, columns = predicted |
| `cal_edges` | (16,) | confidence bin edges |
| `cal_accuracy` | (15,) | accuracy per bin (NaN if empty) |
| `cal_confidence` | (15,) | mean confidence per bin (NaN if empty) |
| `cal_count` | (15,) | examples per bin |

## snapshots/step_XXXXXX.npz

Taken at step 0 (before training) and at every evaluation, always on the same fixed probe set: up to `telemetry.probe_per_label` (5) validation examples per clause type, about 500 in total. This is the "how representations move" record.

| Array | Shape | dtype | Meaning |
|---|---|---|---|
| `step` | () | int64 | optimizer step |
| `epoch` | () | float64 | fractional epoch |
| `probe_ids` | (N,) | int32 | validation row ids |
| `labels` | (N,) | int32 | gold label ids |
| `probs` | (N, 100) | float16 | predicted probabilities (before temperature calibration) |
| `embeddings` | (N, 1024) | float16 | pooled final-layer vector (last real token, after the final norm): the input to the classification head |
| `layer_norms` | (N, 29) | float32 | L2 norm of the pooled vector at every layer; index 0 = token embeddings, 1…28 = transformer layers |
| `layer_subset` | (K,) | int32 | positions (into the probe arrays) of the examples with full per-layer vectors; `telemetry.layer_probe_per_label` (1) per class, about 100 |
| `layer_embeddings` | (K, 29, 1024) | float16 | pooled vector at every layer for those K examples ("node" data) |

Typical uses:

* **Clusters forming over training:** project `embeddings` from every snapshot with one PCA or UMAP fitted on the last snapshot, colour by label, animate across steps (what `06_render_gif.py` does).
* **Data flowing through layers:** for one snapshot, project `layer_embeddings[:, l, :]` for l = 0…28 into a shared basis and animate across layers; compare step 0 with the final step to see where in the network clause types become separable.
* **Learning dynamics per example:** `probs[i, labels[i]]` across snapshots gives each probe example's gold-label confidence over time; its mean and variability give a data-cartography map (easy, ambiguous, hard examples).
* **Per-class progress:** combine with `evals.jsonl["per_class_f1"]` for a label-by-step heatmap.

## system.jsonl

Sampled every `telemetry.system_interval_s` seconds (default 5) by a background thread.

| Field | Meaning |
|---|---|
| `step` | latest optimizer step at sampling time |
| `phase` | `init`, `train`, `eval` or `probe` |
| `cpu_percent` | system-wide CPU utilisation |
| `ram_used_gb`, `ram_total_gb`, `ram_percent` | system memory |
| `process_rss_gb`, `process_threads` | training process |
| `accel_allocated_gb` | live tensors on the accelerator |
| `accel_driver_gb` (MPS) / `accel_reserved_gb` (CUDA) | memory held by the allocator, including cache |
| `gpu_util_percent`, `gpu_temp_c` | CUDA only, when `pynvml` is installed |
| `load_avg_1m` | 1-minute load average |

## Size

For the default 2-epoch run on LEDGAR, expect roughly: `steps.jsonl` 9 MB (about 2.4 KB per step), `layers.jsonl` 0.4 MB, snapshots 9 files × 6.6 MB, evals under 1 MB, system about 1 MB. About 70 MB in total.

## Loading example

```python
import json, numpy as np, glob

tel = "artifacts/runs/ledgar-qwen3-0.6b-lora/telemetry"
meta = json.load(open(f"{tel}/meta.json"))
steps = [json.loads(l) for l in open(f"{tel}/steps.jsonl")]
loss = np.array([s["loss"] for s in steps])

snaps = [np.load(f) for f in sorted(glob.glob(f"{tel}/snapshots/*.npz"))]
first, last = snaps[0], snaps[-1]
gold_conf = [s["probs"][np.arange(len(s["labels"])), s["labels"]].astype(float) for s in snaps]
print(meta["label_names"][:5], loss[-10:].mean(), [round(g.mean(), 3) for g in gold_conf])
```
