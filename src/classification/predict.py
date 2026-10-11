"""Inference: contract text -> [{name, confidence}] per provision and per document.

`Predictor` is split from the scoring backend so the API and tests can inject a stub:
    backend.predict_proba(texts) -> np.ndarray (n, num_labels)   # calibrated probabilities
    backend.label_names -> list[str]
    backend.count_tokens(text) -> int
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from .segment import segment_contract

log = logging.getLogger(__name__)


class Backend(Protocol):
    label_names: list[str]
    model_id: str
    device: str

    def predict_proba(self, texts: list[str]) -> np.ndarray: ...

    def count_tokens(self, text: str) -> int: ...


class HFBackend:
    """A merged HF model directory (artifacts/runs/<model>/merged) on cuda / mps / cpu."""

    def __init__(self, model_dir: str | Path, device: str = "auto", batch_size: int = 16, max_len: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification

        from .config import resolve_device
        from .model import load_temperature, load_tokenizer

        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"No trained model at {model_dir} (run scripts/02_train.py first)")
        self.torch = torch
        self.dev = resolve_device(device)
        self.device = str(self.dev)
        self.tokenizer = load_tokenizer(model_dir)
        # Merged checkpoints are stored in bf16; compute in fp32 on mps/cpu for accuracy.
        dtype = torch.bfloat16 if self.dev.type == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), dtype=dtype).to(self.dev).eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        cfg = self.model.config
        self.label_names = [cfg.id2label[i] for i in range(cfg.num_labels)]
        self.model_dir = model_dir
        self._load_temperature = load_temperature
        self._calibration_mtime: float | None = None
        self.temperature = 1.0
        self._refresh_temperature()
        self.batch_size = batch_size
        self.max_len = max_len
        resolved = model_dir.resolve()
        info_path = resolved / "model_info.json"
        info_id = None
        if info_path.exists():
            import json

            info_id = json.loads(info_path.read_text()).get("id")
        # artifacts/runs/<run>/merged -> the run name (e.g. ledgar-qwen3-0.6b-lora-v2) identifies the model
        self.model_id = info_id or (resolved.parent.name if resolved.name == "merged" else resolved.name)
        log.info("Loaded %s on %s (T=%.3f, %d labels)", model_dir, self.device, self.temperature, len(self.label_names))

    def _refresh_temperature(self) -> None:
        """Pick up calibration.json written by `make evaluate`, even while the API is running."""
        path = Path(self.model_dir) / "calibration.json"
        mtime = path.stat().st_mtime if path.exists() else None
        if mtime != self._calibration_mtime:
            self._calibration_mtime = mtime
            self.temperature = self._load_temperature(self.model_dir)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        if not texts:
            return np.zeros((0, len(self.label_names)), dtype=np.float32)
        self._refresh_temperature()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-sort to reduce padding
        probs = np.zeros((len(texts), len(self.label_names)), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s : s + self.batch_size]
                enc = self.tokenizer(
                    [texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
                ).to(self.dev)
                logits = self.model(**enc).logits.float() / self.temperature
                probs[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs


def top_predictions(probs: np.ndarray, names: list[str], top_k: int, threshold: float) -> list[dict]:
    """Top-k labels with probability >= threshold; the top-1 is always kept."""
    order = np.argsort(-probs)[:top_k]
    out = []
    for rank, i in enumerate(order):
        p = float(probs[i])
        if rank == 0 or p >= threshold:
            out.append({"name": names[i], "confidence": round(p, 4)})
    return out


class Predictor:
    def __init__(self, backend: Backend, top_k: int = 3, threshold: float = 0.30,
                 min_words: int = 25, max_words: int = 350):
        self.backend = backend
        self.top_k = top_k
        self.threshold = threshold
        self.min_words = min_words
        self.max_words = max_words

    @property
    def label_names(self) -> list[str]:
        return self.backend.label_names

    def predict_provisions(self, texts: list[str], top_k: int | None = None, threshold: float | None = None) -> list[list[dict]]:
        k = top_k or self.top_k
        th = self.threshold if threshold is None else threshold
        probs = self.backend.predict_proba(texts)
        return [top_predictions(p, self.label_names, k, th) for p in probs]

    def split(self, text: str, segment: bool = True) -> list[str]:
        """The provisions a text is classified as (shared by every model in POST /v1/compare)."""
        if segment:
            return [p.text for p in segment_contract(text, self.min_words, self.max_words)]
        return [text.strip()] if text.strip() else []

    def predict_document(self, text: str, top_k: int | None = None, threshold: float | None = None,
                         segment: bool = True) -> dict:
        """Classify each provision, then aggregate to document level (max confidence per label)."""
        return self.predict_provisions_document(self.split(text, segment), top_k, threshold)

    def predict_provisions_document(self, provisions: list[str], top_k: int | None = None,
                                    threshold: float | None = None) -> dict:
        """Same output as predict_document, for provisions that are already split."""
        k = top_k or self.top_k
        th = self.threshold if threshold is None else threshold
        if not provisions:
            return {"document": [], "clauses": [], "num_clauses": 0}
        probs = self.backend.predict_proba(provisions)
        clauses = []
        for i, (txt, p) in enumerate(zip(provisions, probs)):
            clauses.append({
                "index": i,
                "text": txt if len(txt) <= 200 else txt[:197] + "...",
                "predictions": top_predictions(p, self.label_names, k, th),
            })
        # Document level: a label is present if any provision predicts it confidently.
        best = probs.max(axis=0)
        argmax_labels = set(probs.argmax(axis=1).tolist())
        doc = [
            {"name": self.label_names[i], "confidence": round(float(best[i]), 4)}
            for i in np.argsort(-best)
            if best[i] >= th or i in argmax_labels and best[i] >= th / 2
        ]
        return {"document": doc, "clauses": clauses, "num_clauses": len(clauses)}

    def count_tokens(self, text: str) -> int:
        return self.backend.count_tokens(text)


def load_predictor(model_dir: str | Path, device: str = "auto", top_k: int = 3, threshold: float = 0.30,
                   batch_size: int = 16) -> Predictor:
    return Predictor(HFBackend(model_dir, device=device, batch_size=batch_size), top_k=top_k, threshold=threshold)
