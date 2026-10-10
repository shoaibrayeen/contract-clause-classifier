"""Device-agnostic training loop (plain PyTorch, no HF Trainer).

Layout of a run directory (artifacts/runs/<run_name>/):
    config.yaml          resolved config
    metrics.jsonl        one JSON object per log / eval event
    checkpoint-best/     best adapter (or full weights) by validation macro-F1
    checkpoint-last/     latest state incl. optimizer/scheduler for --resume
    merged/              merged, serving-ready model + model_info.json (written at the end)
    telemetry/           everything needed to visualise the run (see telemetry.py, scripts/06_visualize.py)
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.live import Live
from rich.progress import Progress
from torch.utils.data import DataLoader

from .config import Config, display_path, resolve_autocast_dtype, resolve_device, resolve_path
from .data import (
    LengthGroupedBatchSampler,
    TokenizedDataset,
    class_weights,
    get_label_names,
    get_label_space,
    load_tokenized,
    make_collate,
)
from .model import build_model, count_parameters, export_merged, load_tokenizer, save_checkpoint
from .registry import refresh_best_link, split_name, write_model_info
from .dashboard import TrainingDashboard
from .telemetry import (
    RunRecorder,
    SystemMonitor,
    choose_probe_indices,
    dataset_stats,
    environment_info,
    grad_norms_by_layer,
    probe_forward,
    weight_norms_by_layer,
)

log = logging.getLogger(__name__)
console = Console()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast_ctx(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def release_cache(device: torch.device) -> None:
    """Return cached allocator blocks to the OS. Variable-length batches make MPS cache grow to 30+ GB."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def peak_memory_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0


def cosine_with_warmup(optimizer, warmup: int, total: int):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.inference_mode()
def predict_logits(model, loader, device, amp_dtype, progress: Progress | None = None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    task = progress.add_task("eval", total=len(loader), status="") if progress else None
    for batch in loader:
        with autocast_ctx(device, amp_dtype):
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        all_logits.append(out.logits.float().cpu().numpy())
        all_labels.append(batch["labels"].numpy())
        if progress:
            progress.advance(task)
    if progress:
        progress.remove_task(task)
    return np.concatenate(all_logits), np.concatenate(all_labels)


def quick_metrics(logits: np.ndarray, labels: np.ndarray, num_labels: int) -> dict[str, float]:
    """Accuracy (= micro-F1 for single-label) and macro-F1 without sklearn."""
    pred = logits.argmax(-1)
    acc = float((pred == labels).mean())
    f1s = []
    for c in range(num_labels):
        tp = np.sum((pred == c) & (labels == c))
        fp = np.sum((pred == c) & (labels != c))
        fn = np.sum((pred != c) & (labels == c))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    logp = logits - logits.max(-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
    nll = float(-logp[np.arange(len(labels)), labels].mean())
    return {"accuracy": acc, "micro_f1": acc, "macro_f1": float(np.mean(f1s)), "loss": nll}


class Trainer:
    """Trains one run into <output_dir>/<train.run_name>.

    `version` is the model version this run produces (None for explicitly named, unversioned runs such
    as `sanity`). scripts/02_train.py picks the run name and version via registry.plan_training_run.
    """

    def __init__(self, cfg: Config, resume: bool = False, version: int | None = None):
        self.cfg = cfg
        self.resume = resume
        self.version = version
        self.device = resolve_device(cfg.device)
        self.amp_dtype = resolve_autocast_dtype(cfg.precision, self.device)
        self.run_dir = resolve_path(cfg.train.output_dir) / cfg.train.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

    # ------------------------------------------------------------------ utils
    def log_event(self, **event) -> None:
        event["time"] = time.time()
        with self.metrics_path.open("a") as fh:
            fh.write(json.dumps(event) + "\n")

    def _loaders(self, tok):
        t = self.cfg.train
        train_ids, train_labels = load_tokenized(self.cfg.data, "train", tok, self.cfg.model_name)
        val_ids, val_labels = load_tokenized(self.cfg.data, "validation", tok, self.cfg.model_name)
        tel = self.cfg.telemetry
        self.probe_idx = choose_probe_indices(val_labels, tel.probe_per_label, t.seed + 2) if tel.enabled and tel.probe_per_label else []
        self.probe_ids = [val_ids[i] for i in self.probe_idx]
        self.probe_labels = [val_labels[i] for i in self.probe_idx]
        self.full_val_stats = (val_ids, val_labels)
        train_src = list(range(len(train_ids)))
        if t.limit:
            rng = random.Random(t.seed)
            train_src = rng.sample(range(len(train_ids)), min(t.limit, len(train_ids)))
            train_ids, train_labels = [train_ids[i] for i in train_src], [train_labels[i] for i in train_src]
        val_src = list(range(len(val_ids)))
        if t.eval_limit:
            rng = random.Random(t.seed + 1)
            val_src = sorted(rng.sample(range(len(val_ids)), min(t.eval_limit, len(val_ids))))
            val_ids, val_labels = [val_ids[i] for i in val_src], [val_labels[i] for i in val_src]
        train_ds = TokenizedDataset(train_ids, train_labels, train_src)
        val_ds = TokenizedDataset(val_ids, val_labels, val_src)
        collate = make_collate(tok.pad_token_id)
        self.train_sampler = LengthGroupedBatchSampler(train_ds.lengths(), t.batch_size, seed=t.seed)
        train_loader = DataLoader(train_ds, batch_sampler=self.train_sampler, collate_fn=collate, num_workers=t.num_workers)
        val_sampler = LengthGroupedBatchSampler(val_ds.lengths(), t.eval_batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collate, num_workers=t.num_workers)
        return train_loader, val_loader, train_labels

    # ------------------------------------------------------------------ main
    def fit(self) -> dict:
        cfg, t = self.cfg, self.cfg.train
        set_seed(t.seed)
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

        label_names = get_label_names(cfg.data)
        num_labels = len(label_names)
        tok = load_tokenizer(cfg.model_name)
        train_loader, val_loader, train_labels = self._loaders(tok)

        last_dir = self.run_dir / "checkpoint-last"
        best_dir = self.run_dir / "checkpoint-best"
        state = {"global_step": 0, "best_score": -1.0, "best_macro_f1": -1.0, "best_accuracy": -1.0, "bad_evals": 0}

        model = build_model(cfg, label_names, tok)
        if self.resume and (last_dir / "trainer_state.json").exists():
            state = json.loads((last_dir / "trainer_state.json").read_text())
            self._load_weights(model, last_dir)
            console.print(f"[yellow]Resuming from step {state['global_step']}[/]")
        model.to(self.device)
        trainable, total = count_parameters(model)

        params = [p for p in model.parameters() if p.requires_grad]
        decay = [p for p in params if p.ndim >= 2]
        no_decay = [p for p in params if p.ndim < 2]
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": t.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=t.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        steps_per_epoch = math.ceil(len(train_loader) / t.grad_accum)
        total_steps = max(1, math.ceil(steps_per_epoch * t.epochs))
        scheduler = cosine_with_warmup(optimizer, int(t.warmup_ratio * total_steps), total_steps)
        if self.resume and (last_dir / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(last_dir / "optimizer.pt", map_location=self.device))
            scheduler.load_state_dict(torch.load(last_dir / "scheduler.pt"))

        weight = class_weights(train_labels, num_labels).to(self.device) if t.class_weighted_loss else None
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=t.label_smoothing)

        info = {
            "device": str(self.device),
            "amp": str(self.amp_dtype),
            "trainable_params": trainable,
            "total_params": total,
            "train_examples": len(train_loader.dataset),
            "val_examples": len(val_loader.dataset),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
        }
        console.print(info)
        self.log_event(event="start", **info)

        tel = cfg.telemetry
        rec = self.recorder = RunRecorder(self.run_dir, enabled=tel.enabled)
        base_cfg = model.config
        rec.write_meta({
            "run_name": t.run_name,
            "model_id": t.run_name,
            "model_version": self.version,
            "config": cfg.to_dict(),
            "environment": environment_info(self.device),
            "label_names": label_names,
            "label_groups": {"source": cfg.data.label_groups, "groups": get_label_space(cfg.data).groups,
                             "raw_label_names": get_label_space(cfg.data).raw_names},
            "model": {
                "name": cfg.model_name, "num_layers": getattr(base_cfg, "num_hidden_layers", None),
                "hidden_size": getattr(base_cfg, "hidden_size", None), **info,
            },
            "datasets": {
                "train": dataset_stats("train", train_loader.dataset.lengths(), train_labels, num_labels),
                "validation": dataset_stats("validation", [len(x) for x in self.full_val_stats[0]],
                                            self.full_val_stats[1], num_labels),
                "eval_subset_size": len(val_loader.dataset),
            },
            "probe": {"split": "validation", "indices": self.probe_idx, "labels": self.probe_labels},
        }, new_session=True)
        # Per-layer ("node") vectors for a small stratified subset of the probe set.
        seen: dict[int, int] = {}
        self.layer_subset = []
        for pos, y in enumerate(self.probe_labels):
            if seen.get(y, 0) < tel.layer_probe_per_label:
                seen[y] = seen.get(y, 0) + 1
                self.layer_subset.append(pos)
        rec.event("start", step=state["global_step"], resumed=self.resume, **info)
        self.monitor = None
        if tel.enabled and tel.system_interval_s:
            self.monitor = SystemMonitor(rec, self.device, tel.system_interval_s)
            self.monitor.step = state["global_step"]
            self.monitor.start()
        if tel.enabled and self.probe_ids and state["global_step"] == 0:
            self._snapshot(model, tok, 0, 0.0)

        global_step = state["global_step"]
        micro_to_skip = global_step * t.grad_accum  # resume: skip already-consumed micro-batches
        start_epoch = micro_to_skip // len(train_loader)
        micro_to_skip -= start_epoch * len(train_loader)
        stop = False
        running_loss, running_n, tokens_seen, t0 = 0.0, 0, 0, time.time()
        step_rec = self._new_step_rec()
        step_t0 = time.time()

        dash = self.dashboard = TrainingDashboard(console, total_steps, global_step, label_names, str(self.run_dir))
        progress = dash.progress
        live_ctx = Live(dash, console=console, refresh_per_second=2, vertical_overflow="crop") \
            if (tel.live_dashboard and dash.interactive) \
            else contextlib.nullcontext()
        with live_ctx:
            for epoch in range(start_epoch, math.ceil(t.epochs)):
                self.train_sampler.set_epoch(epoch)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for micro, batch in enumerate(train_loader):
                    if micro_to_skip > 0:
                        micro_to_skip -= 1
                        continue
                    src_index = batch.pop("index")
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    with autocast_ctx(self.device, self.amp_dtype):
                        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    logits = out.logits.float()
                    loss = loss_fn(logits, batch["labels"])
                    (loss / t.grad_accum).backward()
                    loss_val = loss.item()
                    running_loss += loss_val
                    running_n += 1
                    lengths = batch["attention_mask"].sum(1)
                    n_tokens = int(lengths.sum())
                    tokens_seen += n_tokens
                    step_rec["loss_sum"] += loss_val
                    step_rec["micro"] += 1
                    step_rec["tokens"] += n_tokens
                    step_rec["padded_tokens"] += int(batch["attention_mask"].numel())
                    if tel.enabled and tel.record_batches:
                        with torch.no_grad():
                            ex_loss = torch.nn.functional.cross_entropy(logits, batch["labels"], reduction="none")
                            ex_correct = logits.argmax(-1) == batch["labels"]
                        step_rec["ids"] += src_index.tolist()
                        step_rec["labels"] += batch["labels"].cpu().tolist()
                        step_rec["lengths"] += lengths.cpu().tolist()
                        step_rec["example_loss"] += [round(x, 4) for x in ex_loss.cpu().tolist()]
                        step_rec["correct"] += ex_correct.cpu().int().tolist()

                    last_micro = micro + 1 == len(train_loader)
                    if (micro + 1) % t.grad_accum and not last_micro:
                        continue
                    layer_norms = None
                    if tel.enabled and tel.layer_stats_every and (global_step + 1) % tel.layer_stats_every == 0:
                        layer_norms = grad_norms_by_layer(model)  # pre-clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, t.max_grad_norm)
                    lr_used = scheduler.get_last_lr()[0]
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if t.empty_cache_steps and global_step % t.empty_cache_steps == 0:
                        release_cache(self.device)
                    if self.monitor:
                        self.monitor.step = global_step
                        self.monitor.phase = "train"

                    if tel.enabled:
                        dt = time.time() - step_t0
                        r = step_rec
                        record = {
                            "step": global_step, "epoch": epoch + (micro + 1) / len(train_loader),
                            "loss": r["loss_sum"] / max(1, r["micro"]), "lr": lr_used,
                            "grad_norm": float(grad_norm), "clipped": float(grad_norm) > t.max_grad_norm,
                            "step_time_s": dt, "tokens": r["tokens"], "padded_tokens": r["padded_tokens"],
                            "padding_ratio": 1 - r["tokens"] / max(1, r["padded_tokens"]),
                            "tokens_per_s": r["tokens"] / max(dt, 1e-6), "mem_gb": peak_memory_gb(self.device),
                        }
                        if tel.record_batches:
                            record.update(ids=r["ids"], labels=r["labels"], lengths=r["lengths"],
                                          example_loss=r["example_loss"], correct=r["correct"],
                                          batch_accuracy=sum(r["correct"]) / max(1, len(r["correct"])))
                        rec.step(**record)
                        if layer_norms is not None:
                            rec.layers(global_step, layer_norms)
                    else:
                        record = {"loss": step_rec["loss_sum"] / max(1, step_rec["micro"]), "lr": lr_used,
                                  "grad_norm": float(grad_norm), "epoch": epoch + (micro + 1) / len(train_loader)}
                    dash.on_step(global_step, record, step_rec["labels"])
                    if self.monitor:
                        dash.on_system(self.monitor.latest)
                    if not dash.interactive and global_step % t.log_steps == 0:
                        print(dash.plain_line(global_step), flush=True)
                    step_rec = self._new_step_rec()
                    step_t0 = time.time()

                    if global_step % t.log_steps == 0:
                        elapsed = time.time() - t0
                        ev = {
                            "event": "train", "step": global_step, "epoch": epoch + (micro + 1) / len(train_loader),
                            "loss": running_loss / max(1, running_n), "lr": scheduler.get_last_lr()[0],
                            "grad_norm": float(grad_norm), "tokens_per_s": tokens_seen / max(elapsed, 1e-6),
                            "mem_gb": peak_memory_gb(self.device),
                        }
                        self.log_event(**ev)
                        running_loss, running_n, tokens_seen, t0 = 0.0, 0, 0, time.time()

                    end = global_step >= total_steps
                    if global_step % t.eval_steps == 0 or end:
                        stop = self._evaluate_and_checkpoint(
                            model, tok, val_loader, label_names, optimizer, scheduler, state, global_step,
                            best_dir, last_dir, progress, epoch + (micro + 1) / len(train_loader),
                        )
                        release_cache(self.device)
                        step_t0 = time.time()  # don't count eval time in the next step
                        model.train()
                    if stop or end:
                        break
                if stop or global_step >= total_steps:
                    break

        # Export the best checkpoint as a merged, serving-ready model.
        merged = export_merged(best_dir, self.run_dir / "merged")
        base, _ = split_name(t.run_name)
        write_model_info(
            merged, id=t.run_name, base_name=base if self.version is not None else t.run_name, version=self.version,
            base_model=cfg.model_name, method="full" if cfg.full_finetune or not cfg.lora.enabled else "lora",
            best_val_macro_f1=state["best_macro_f1"], best_val_accuracy=state.get("best_accuracy"),
            best_metric=t.best_metric, num_labels=num_labels, label_groups=cfg.data.label_groups,
            steps=global_step, train_examples=len(train_loader.dataset),
            run_dir=str(self.run_dir),
        )
        best_link = refresh_best_link(self.run_dir.parent)
        summary = {"event": "done", "model_id": t.run_name, "model_version": self.version,
                   "best_macro_f1": state["best_macro_f1"], "steps": global_step, "merged": str(merged),
                   "best_link": str(best_link.readlink()) if best_link else None}
        self.log_event(**summary)
        rec.event("done", **summary)
        console.print(f"[bold green]Model ready:[/] {t.run_name} -> {display_path(merged)}")
        rec.write_meta({"summary": summary})
        if self.monitor:
            self.monitor.stop()
        rec.close()
        return summary

    def _snapshot(self, model, tok, step: int, epoch: float) -> None:
        if self.monitor:
            self.monitor.phase = "probe"
        probe = probe_forward(
            model, self.probe_ids, tok.pad_token_id, self.device, 16,
            lambda: autocast_ctx(self.device, self.amp_dtype), self.layer_subset, self.cfg.telemetry.embeddings,
        )
        self.recorder.snapshot(step, epoch, np.asarray(self.probe_idx), np.asarray(self.probe_labels), probe)
        probe_acc = float((probe["probs"].argmax(-1) == np.asarray(self.probe_labels)).mean())
        self.recorder.event("snapshot", step=step, epoch=epoch, probe_accuracy=probe_acc, n=len(self.probe_ids))

    @staticmethod
    def _new_step_rec() -> dict:
        return {"loss_sum": 0.0, "micro": 0, "tokens": 0, "padded_tokens": 0,
                "ids": [], "labels": [], "lengths": [], "example_loss": [], "correct": []}

    def _evaluate_and_checkpoint(self, model, tok, val_loader, label_names, optimizer, scheduler, state, step,
                                 best_dir, last_dir, progress, epoch: float = 0.0) -> bool:
        t_eval = time.time()
        if self.monitor:
            self.monitor.phase = "eval"
        logits, labels = predict_logits(model, val_loader, self.device, self.amp_dtype, progress)
        m = quick_metrics(logits, labels, len(label_names))
        tel, rec = self.cfg.telemetry, self.recorder
        if tel.enabled:
            m_eval = {**m, "eval_time_s": time.time() - t_eval, "n": int(len(labels))}
            rec.evaluation(step, epoch, m_eval, logits, labels, len(label_names), weight_norms_by_layer(model))
            if self.probe_ids:
                self._snapshot(model, tok, step, epoch)
        metric = self.cfg.train.best_metric
        if metric not in ("accuracy", "macro_f1"):
            raise ValueError(f"train.best_metric must be 'accuracy' or 'macro_f1', got {metric!r}")
        state.setdefault("best_score", state.get("best_macro_f1", -1.0))  # checkpoints from older versions
        improved = m[metric] > state["best_score"]
        if improved:
            state["best_score"] = m[metric]
            state["best_macro_f1"] = m["macro_f1"]
            state["best_accuracy"] = m["accuracy"]
            state["bad_evals"] = 0
            save_checkpoint(model, tok, best_dir, label_names)
        else:
            state["bad_evals"] += 1
        state["global_step"] = step
        save_checkpoint(model, tok, last_dir, label_names)
        torch.save(optimizer.state_dict(), last_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), last_dir / "scheduler.pt")
        (last_dir / "trainer_state.json").write_text(json.dumps(state))
        self.log_event(event="eval", step=step, improved=improved, **{f"val_{k}": v for k, v in m.items()})
        self.recorder.event("eval", step=step, epoch=epoch, improved=improved, **m)
        self.recorder.event("checkpoint", step=step, path=str(last_dir), best=improved)
        self.dashboard.on_eval(step, m, improved)
        if not self.dashboard.interactive:
            console.print(
                f"[cyan]eval step {step}[/] acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
                f"loss={m['loss']:.4f} {'[green]best[/]' if improved else ''}"
            )
        patience = self.cfg.train.early_stopping_patience
        if patience and state["bad_evals"] >= patience:
            console.print("[yellow]Early stopping[/]")
            self.recorder.event("early_stop", step=step, patience=patience)
            return True
        return False

    def _load_weights(self, model, ckpt: Path) -> None:
        if hasattr(model, "load_adapter"):
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            set_peft_model_state_dict(model, load_file(ckpt / "adapter_model.safetensors"))
        else:
            from transformers import AutoModelForSequenceClassification

            src = AutoModelForSequenceClassification.from_pretrained(str(ckpt), dtype=torch.float32)
            model.load_state_dict(src.state_dict())


def update_best_link(target: Path) -> Path | None:
    """Deprecated: kept for compatibility. `artifacts/best` now always points at the latest model."""
    runs_dir = Path(target).resolve().parent.parent
    return refresh_best_link(runs_dir)
