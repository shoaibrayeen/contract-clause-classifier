"""Shared fixtures: a stub scoring backend so tests never need the trained model."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from classification.config import DataConfig
from classification.labels import save_labels
from classification.predict import Predictor

LABELS = ["Governing Laws", "Confidentiality", "Terminations", "Notices", "Assignments"]
KEYWORDS = {"governed": 0, "laws of": 0, "confidential": 1, "terminate": 2, "notice": 3, "assign": 4}


class StubBackend:
    """Keyword scorer that mimics the HF backend interface."""

    label_names = LABELS
    model_id = "stub"
    device = "cpu"

    def predict_proba(self, texts):
        out = np.full((len(texts), len(LABELS)), 0.02, dtype=np.float32)
        for r, t in enumerate(texts):
            low = t.lower()
            for kw, idx in KEYWORDS.items():
                if kw in low:
                    out[r, idx] += 1.0
        return out / out.sum(1, keepdims=True)

    def count_tokens(self, text):
        return len(text.split())


@pytest.fixture
def predictor():
    return Predictor(StubBackend(), top_k=3, threshold=0.3)


@pytest.fixture
def tiny_data(tmp_path):
    """A fake LEDGAR layout (raw parquet + labels.json) so no download is needed."""
    cfg = DataConfig(data_dir=str(tmp_path))
    raw = tmp_path / "raw" / "ledgar"
    raw.mkdir(parents=True)
    for split, n in (("train", 25), ("validation", 7), ("test", 5)):
        pq.write_table(
            pa.table({"text": [f"{split} provision {i}" for i in range(n)], "label": [i % len(LABELS) for i in range(n)]}),
            raw / f"{split}.parquet",
        )
    save_labels(LABELS, raw / "labels.json")
    return cfg


def make_run(runs_dir, name, created, version="auto", complete=True, base=None):
    """Create a fake run directory; complete runs get merged/config.json + model_info.json."""
    import json

    from classification.registry import split_name

    run = runs_dir / name
    (run / "checkpoint-last").mkdir(parents=True, exist_ok=True)
    (run / "checkpoint-last" / "trainer_state.json").write_text('{"global_step": 500}')
    if complete:
        (run / "merged").mkdir(parents=True, exist_ok=True)
        (run / "merged" / "config.json").write_text("{}")
        b, v = split_name(name)
        info = {"id": name, "base_name": base or b, "version": v if version == "auto" else version, "created": created}
        (run / "merged" / "model_info.json").write_text(json.dumps(info))
    return run
