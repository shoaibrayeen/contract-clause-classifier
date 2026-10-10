"""Training telemetry: record everything that happens during a run for later visualisation.

Everything lands in <run_dir>/telemetry/ as append-only files (safe across --resume):

    meta.json                run config, hardware/software environment, dataset stats, probe set ids
    events.jsonl             lifecycle events: start, eval, checkpoint, early_stop, done
    system.jsonl             every few seconds: CPU %, RAM, process RSS, accelerator memory, step reached
    steps.jsonl              one line per optimizer step: loss, lr, grad-norm, throughput, memory,
                             padding ratio, and the batch itself (example ids, labels, lengths,
                             per-example losses) -> "how data moved through training"
    layers.jsonl             every `layer_stats_every` steps: grad-norm per transformer layer / head
    evals.jsonl              one line per evaluation: metrics, per-class F1, LoRA weight norms per layer
    evals/step_XXXXXX.npz    per evaluation: confusion matrix (int32) and calibration bins
    snapshots/step_XXXXXX.npz  per evaluation, for the fixed probe set: probabilities (float16),
                             pooled last-layer embeddings (float16), labels -> cluster movement
                             and per-example learning dynamics (dataset cartography).
                             Also "node" data: per-layer activation norms for every probe example and
                             per-layer pooled vectors for a small subset -> how an example's representation
                             flows through the 28 transformer layers and how that changes over training.

See docs/TELEMETRY.md for the exact schema.
"""

from __future__ import annotations

import json
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v.tolist()
    if isinstance(v, Path):
        return str(v)
    return v


def environment_info(device: torch.device) -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "device": str(device),
        "hostname": platform.node(),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import peft

        info["peft"] = peft.__version__
    except Exception:
        pass
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    if sys.platform == "darwin":
        for key, name in (("machdep.cpu.brand_string", "chip"), ("hw.memsize", "memory_bytes")):
            try:
                info[name] = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
            except Exception:
                pass
        if "memory_bytes" in info:
            info["memory_gb"] = round(int(info.pop("memory_bytes")) / 2**30, 1)
    return info


def dataset_stats(name: str, lengths: list[int], labels: list[int], num_labels: int) -> dict:
    lens = np.asarray(lengths)
    hist, edges = np.histogram(lens, bins=32, range=(0, max(int(lens.max()), 1)))
    return {
        "split": name,
        "n": int(len(lens)),
        "label_counts": np.bincount(np.asarray(labels), minlength=num_labels).tolist(),
        "length_hist": {"counts": hist.tolist(), "edges": edges.tolist()},
        "length_mean": float(lens.mean()),
        "length_p50": float(np.percentile(lens, 50)),
        "length_p90": float(np.percentile(lens, 90)),
    }


def choose_probe_indices(labels: list[int], per_label: int, seed: int) -> list[int]:
    """Stratified fixed subset of validation: up to `per_label` examples per class."""
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_label[y].append(i)
    rng = random.Random(seed)
    out: list[int] = []
    for y in sorted(by_label):
        idx = by_label[y]
        out.extend(rng.sample(idx, min(per_label, len(idx))))
    return sorted(out)


def layer_key(param_name: str) -> str:
    m = LAYER_RE.search(param_name)
    if m:
        return f"layer_{int(m.group(1)):02d}"
    if "score" in param_name:
        return "head"
    if "embed" in param_name:
        return "embed"
    return "other"


def grad_norms_by_layer(model: torch.nn.Module) -> dict[str, float]:
    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for name, p in model.named_parameters():
        if p.grad is not None:
            groups[layer_key(name)].append(p.grad.detach().float().pow(2).sum())
    if not groups:
        return {}
    keys = sorted(groups)
    vals = torch.stack([torch.stack(groups[k]).sum() for k in keys]).sqrt().cpu().tolist()
    return dict(zip(keys, vals))


def weight_norms_by_layer(model: torch.nn.Module) -> dict[str, float]:
    """Norm of trainable weights per layer (LoRA A/B + head) — how far each layer has moved."""
    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups[layer_key(name)].append(p.detach().float().pow(2).sum())
    if not groups:
        return {}
    keys = sorted(groups)
    vals = torch.stack([torch.stack(groups[k]).sum() for k in keys]).sqrt().cpu().tolist()
    return dict(zip(keys, vals))


def per_class_f1(pred: np.ndarray, labels: np.ndarray, num_labels: int) -> list[float | None]:
    out: list[float | None] = []
    for c in range(num_labels):
        tp = int(np.sum((pred == c) & (labels == c)))
        fp = int(np.sum((pred == c) & (labels != c)))
        fn = int(np.sum((pred != c) & (labels == c)))
        out.append(None if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return out


def confusion(pred: np.ndarray, labels: np.ndarray, num_labels: int) -> np.ndarray:
    cm = np.zeros((num_labels, num_labels), dtype=np.int32)
    np.add.at(cm, (labels, pred), 1)
    return cm


def calibration_bins(logits: np.ndarray, labels: np.ndarray, bins: int = 15) -> dict[str, list]:
    z = logits - logits.max(-1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    conf, correct = p.max(-1), p.argmax(-1) == labels
    edges = np.linspace(0, 1, bins + 1)
    acc, cf, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        cnt.append(int(m.sum()))
        acc.append(float(correct[m].mean()) if m.any() else None)
        cf.append(float(conf[m].mean()) if m.any() else None)
    return {"edges": edges.tolist(), "accuracy": acc, "confidence": cf, "count": cnt}


class RunRecorder:
    """Append-only writer for one run's telemetry. No-ops when disabled."""

    def __init__(self, run_dir: Path, enabled: bool = True):
        self.enabled = enabled
        self.dir = Path(run_dir) / "telemetry"
        if enabled:
            (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)
            (self.dir / "evals").mkdir(parents=True, exist_ok=True)
        self._fh: dict[str, object] = {}

    def _write(self, name: str, record: dict) -> None:
        if not self.enabled:
            return
        fh = self._fh.get(name)
        if fh is None:
            fh = open(self.dir / name, "a", buffering=1)  # line-buffered: survives crashes
            self._fh[name] = fh
        record = {k: _jsonable(v) for k, v in record.items()}
        record.setdefault("time", time.time())
        fh.write(json.dumps(record) + "\n")

    def write_meta(self, meta: dict) -> None:
        if not self.enabled:
            return
        path = self.dir / "meta.json"
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing.setdefault("sessions", []).append({"started": time.time(), "resumed": bool(existing)})
        existing.update({k: v for k, v in meta.items() if k != "sessions"})
        path.write_text(json.dumps(existing, indent=2, default=_jsonable))

    def step(self, **record) -> None:
        self._write("steps.jsonl", record)

    def event(self, name: str, **record) -> None:
        self._write("events.jsonl", {"event": name, **record})

    def system(self, **record) -> None:
        self._write("system.jsonl", record)

    def layers(self, step: int, grad_norms: dict[str, float]) -> None:
        self._write("layers.jsonl", {"step": step, "grad_norm": grad_norms})

    def evaluation(self, step: int, epoch: float, metrics: dict, logits: np.ndarray, labels: np.ndarray,
                   num_labels: int, weight_norms: dict[str, float]) -> None:
        if not self.enabled:
            return
        pred = logits.argmax(-1)
        self._write("evals.jsonl", {
            "step": step, "epoch": epoch, **metrics,
            "per_class_f1": per_class_f1(pred, labels, num_labels),
            "weight_norm": weight_norms,
        })
        cal = calibration_bins(logits, labels)
        np.savez_compressed(
            self.dir / "evals" / f"step_{step:06d}.npz",
            confusion=confusion(pred, labels, num_labels),
            cal_edges=np.asarray(cal["edges"]),
            cal_accuracy=np.asarray([np.nan if a is None else a for a in cal["accuracy"]]),
            cal_confidence=np.asarray([np.nan if c is None else c for c in cal["confidence"]]),
            cal_count=np.asarray(cal["count"]),
        )

    def snapshot(self, step: int, epoch: float, probe_ids: np.ndarray, labels: np.ndarray, probe: dict) -> None:
        """probe = output of probe_forward()."""
        if not self.enabled:
            return
        np.savez_compressed(
            self.dir / "snapshots" / f"step_{step:06d}.npz",
            step=np.int64(step), epoch=np.float64(epoch), probe_ids=probe_ids.astype(np.int32),
            labels=labels.astype(np.int32),
            probs=probe["probs"].astype(np.float16),
            embeddings=probe["embeddings"].astype(np.float16),
            layer_norms=probe["layer_norms"].astype(np.float32),
            layer_subset=probe["layer_subset"].astype(np.int32),
            layer_embeddings=probe["layer_embeddings"].astype(np.float16),
        )

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()


@torch.inference_mode()
def probe_forward(model, input_ids: list[list[int]], pad_id: int, device: torch.device, batch_size: int = 32,
                  amp_ctx=None, layer_subset: list[int] | None = None, store_embeddings: bool = True) -> dict:
    """Run the probe set and collect predictions plus per-layer ("node") data, in input order.

    Returns:
        probs             (N, num_labels)   softmax probabilities
        embeddings        (N, H)            pooled final-layer vector (last real token, after final norm)
        layer_norms       (N, L+1)          L2 norm of the pooled vector at every layer (0 = embeddings)
        layer_subset      (K,)              row positions (into the probe set) with full per-layer vectors
        layer_embeddings  (K, L+1, H)       pooled vector at every layer for those K examples
    """
    import contextlib

    model.eval()
    subset = set(layer_subset or [])
    # Length-sorted batches avoid padding short probes to 512 tokens (2-4x faster on MPS).
    order = sorted(range(len(input_ids)), key=lambda i: len(input_ids[i]))
    n_labels = None
    probs_out: list = [None] * len(input_ids)
    embs_out: list = [None] * len(input_ids)
    norms_out: list = [None] * len(input_ids)
    layer_vecs: dict[int, np.ndarray] = {}
    for s in range(0, len(order), batch_size):
        rows_idx = order[s : s + batch_size]
        chunk = [input_ids[i] for i in rows_idx]
        L = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, x in enumerate(chunk):
            ids[r, : len(x)] = torch.tensor(x)
            mask[r, : len(x)] = 1
        ids, mask = ids.to(device), mask.to(device)
        with amp_ctx() if amp_ctx else contextlib.nullcontext():
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        last = mask.sum(1) - 1  # right padding -> last real token
        rows = torch.arange(len(chunk), device=device)
        pooled = torch.stack([h[rows, last].float() for h in out.hidden_states], dim=1).cpu().numpy()  # (B, L+1, H)
        p = torch.softmax(out.logits.float(), -1).cpu().numpy()
        n_labels = p.shape[-1]
        for r, i in enumerate(rows_idx):
            probs_out[i] = p[r]
            norms_out[i] = np.linalg.norm(pooled[r], axis=-1)
            if store_embeddings:
                embs_out[i] = pooled[r, -1]
            if i in subset:
                layer_vecs[i] = pooled[r]
    probs = [np.stack(probs_out)]
    norms = [np.stack(norms_out)]
    embs = [np.stack(embs_out)] if store_embeddings else []
    layer_rows = sorted(layer_vecs)
    layer_vecs = [layer_vecs[i] for i in layer_rows]
    n, hidden = len(input_ids), (embs[0].shape[-1] if embs else 0)
    return {
        "probs": np.concatenate(probs),
        "embeddings": np.concatenate(embs) if embs else np.zeros((n, 0), dtype=np.float32),
        "layer_norms": np.concatenate(norms),
        "layer_subset": np.asarray(layer_rows, dtype=np.int32),
        "layer_embeddings": np.stack(layer_vecs) if layer_vecs else np.zeros((0, 0, hidden), dtype=np.float32),
    }


class SystemMonitor(threading.Thread):
    """Background sampler: CPU, RAM, process RSS, accelerator memory -> system.jsonl."""

    def __init__(self, recorder: RunRecorder, device: torch.device, interval_s: float = 5.0):
        super().__init__(daemon=True, name="system-monitor")
        self.recorder, self.device, self.interval = recorder, device, interval_s
        self.step = 0  # updated by the trainer so samples can be aligned with steps
        self.phase = "init"
        self._stop = threading.Event()
        self.latest: dict = {}

    def sample(self) -> dict:
        import psutil

        vm = psutil.virtual_memory()
        proc = psutil.Process()
        rec = {
            "step": self.step, "phase": self.phase,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_gb": round((vm.total - vm.available) / 2**30, 3),
            "ram_total_gb": round(vm.total / 2**30, 3),
            "ram_percent": vm.percent,
            "process_rss_gb": round(proc.memory_info().rss / 2**30, 3),
            "process_threads": proc.num_threads(),
        }
        try:
            if self.device.type == "mps":
                rec["accel_allocated_gb"] = round(torch.mps.current_allocated_memory() / 2**30, 3)
                rec["accel_driver_gb"] = round(torch.mps.driver_allocated_memory() / 2**30, 3)
            elif self.device.type == "cuda":
                rec["accel_allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
                rec["accel_reserved_gb"] = round(torch.cuda.memory_reserved() / 2**30, 3)
                try:
                    import pynvml  # optional

                    pynvml.nvmlInit()
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    rec["gpu_util_percent"] = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    rec["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(h, 0)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            la = os.getloadavg()
            rec["load_avg_1m"] = la[0]
        except OSError:
            pass
        return rec

    def run(self) -> None:
        import psutil

        psutil.cpu_percent(interval=None)  # prime
        while not self._stop.wait(self.interval):
            try:
                self.latest = self.sample()
                self.recorder.system(**self.latest)
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()
