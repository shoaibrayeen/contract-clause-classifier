"""Serving backend for classical models (implements `classification.predict.Backend`).

Loads `merged/model.joblib` (a fitted scikit-learn Pipeline: TF-IDF features + linear classifier) and returns
temperature-calibrated probabilities. CPU only; no torch import.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
MODEL_FILE = "model.joblib"


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def scores(pipeline, texts: list[str]) -> np.ndarray:
    """Class scores usable as logits: decision_function for linear models, log-probabilities otherwise."""
    if hasattr(pipeline, "decision_function"):
        try:
            return np.asarray(pipeline.decision_function(texts), dtype=np.float64)
        except AttributeError:  # e.g. a calibrated wrapper without decision_function
            pass
    return np.log(np.clip(np.asarray(pipeline.predict_proba(texts), dtype=np.float64), 1e-12, 1.0))


class SklearnBackend:
    device = "cpu"

    def __init__(self, model_dir: str | Path):
        import joblib

        model_dir = Path(model_dir)
        path = model_dir / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"No classical model at {model_dir} ({MODEL_FILE} missing)")
        self.model_dir = model_dir
        self.pipeline = joblib.load(path)
        self.label_names = json.loads((model_dir / "labels.json").read_text())["names"]
        classes = list(getattr(self.pipeline, "classes_", range(len(self.label_names))))
        if classes != list(range(len(self.label_names))):
            raise ValueError(f"{path}: classifier classes {classes[:5]}... do not match labels.json ids")
        info_path = model_dir / "model_info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        resolved = model_dir.resolve()
        self.model_id = info.get("id") or (resolved.parent.name if resolved.name == "merged" else resolved.name)
        self._cal_mtime: float | None = None
        self.temperature = 1.0
        self._refresh_temperature()
        log.info("Loaded classical model %s (T=%.3f, %d labels)", self.model_id, self.temperature, len(self.label_names))

    def _refresh_temperature(self) -> None:
        path = self.model_dir / "calibration.json"
        mtime = path.stat().st_mtime if path.exists() else None
        if mtime != self._cal_mtime:
            self._cal_mtime = mtime
            self.temperature = float(json.loads(path.read_text()).get("temperature", 1.0)) if path.exists() else 1.0

    def logits(self, texts: list[str]) -> np.ndarray:
        return scores(self.pipeline, texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.label_names)), dtype=np.float32)
        self._refresh_temperature()
        return softmax(self.logits(texts) / self.temperature).astype(np.float32)

    def count_tokens(self, text: str) -> int:
        return len(text.split())  # words: this model has no sub-word tokenizer
