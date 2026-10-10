"""Evaluation: full metrics on a split, temperature-scaling calibration, ECE."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DataConfig, resolve_autocast_dtype
from .data import LengthGroupedBatchSampler, TokenizedDataset, load_tokenized, make_collate
from .train import predict_logits


def split_logits(model, tok, data_cfg: DataConfig, split: str, tokenizer_name: str, device, precision: str = "auto",
                 batch_size: int = 32, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    ids, labels = load_tokenized(data_cfg, split, tok, tokenizer_name)
    if limit:
        ids, labels = ids[:limit], labels[:limit]
    ds = TokenizedDataset(ids, labels)
    sampler = LengthGroupedBatchSampler(ds.lengths(), batch_size, shuffle=False)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=make_collate(tok.pad_token_id))
    # The sampler reorders by length; predict_logits returns labels in the same order.
    return predict_logits(model, loader, device, resolve_autocast_dtype(precision, device))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(-1)
    correct = probs.argmax(-1) == labels
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, iters: int = 200) -> float:
    """Single scalar T minimising NLL on held-out logits (Guo et al., 2017)."""
    lg = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def full_report(logits: np.ndarray, labels: np.ndarray, names: list[str], temperature: float = 1.0,
                top_confusions: int = 20) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    pred = logits.argmax(-1)
    probs = softmax(logits, temperature)
    present = sorted(set(labels.tolist()) | set(pred.tolist()))
    report = classification_report(
        labels, pred, labels=present, target_names=[names[i] for i in present], output_dict=True, zero_division=0
    )
    confusions = Counter((int(t), int(p)) for t, p in zip(labels, pred) if t != p)
    nll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, pred)),
        "micro_f1": float(f1_score(labels, pred, average="micro")),
        "macro_f1": float(f1_score(labels, pred, average="macro", labels=list(range(len(names))), zero_division=0)),
        "weighted_f1": float(f1_score(labels, pred, average="weighted")),
        "nll": nll,
        "ece": expected_calibration_error(probs, labels),
        "ece_uncalibrated": expected_calibration_error(softmax(logits), labels),
        "temperature": temperature,
        "top_confusions": [
            {"true": names[t], "pred": names[p], "count": c} for (t, p), c in confusions.most_common(top_confusions)
        ],
        "per_class": {k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg", "micro avg")},
    }
