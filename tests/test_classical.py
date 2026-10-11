"""Classical model: backend, explanations, config, registry families, and POST /v1/compare (no training data,
GPU or network needed: a tiny pipeline is fitted in the test)."""

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from classification.api.app import create_app
from classification.api.models import StaticModelStore
from classification.api.settings import Settings
from classification.classical.backend import SklearnBackend
from classification.classical.config import load_classical_config
from classification.data import DatasetStore
from classification.predict import Predictor
from classification.registry import CLASSICAL_ALIAS, ModelNotFoundError, latest_model, list_models, resolve_model
from conftest import LABELS, StubBackend, make_run

TEXTS = {
    0: ["This Agreement shall be governed by the laws of New York.", "Governed by the laws of Delaware."],
    1: ["Each party shall keep confidential information secret.", "Confidential information must not be disclosed."],
    2: ["Either party may terminate this Agreement on notice.", "This Agreement terminates upon breach."],
    3: ["All notices shall be in writing and delivered by courier.", "Notice must be given in writing."],
    4: ["Neither party may assign this Agreement without consent.", "No assignment without prior consent."],
}


@pytest.fixture
def classical_dir(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from classification.classical.config import FeaturesConfig, NgramConfig
    from classification.classical.features import build_features

    texts = [t for k in sorted(TEXTS) for t in TEXTS[k]]
    y = [k for k in sorted(TEXTS) for _ in TEXTS[k]]
    feats = build_features(FeaturesConfig(word=NgramConfig(ngram_range=[1, 2], min_df=1, max_features=None),
                                          char=NgramConfig(ngram_range=[3, 4], min_df=1, max_features=None)))
    pipe = Pipeline([("features", feats), ("clf", LogisticRegression(C=10, max_iter=500))]).fit(texts, y)
    merged = tmp_path / "ledgar-tfidf-lr-v1" / "merged"
    merged.mkdir(parents=True)
    joblib.dump(pipe, merged / "model.joblib")
    (merged / "labels.json").write_text(json.dumps({"names": LABELS}))
    (merged / "config.json").write_text("{}")
    (merged / "model_info.json").write_text(json.dumps({"id": "ledgar-tfidf-lr-v1", "version": 1, "family": "classical",
                                                        "base_name": "ledgar-tfidf-lr", "created": time.time()}))
    return merged


def test_backend_predicts_calibrated_probabilities(classical_dir):
    b = SklearnBackend(classical_dir)
    p = b.predict_proba(["This Agreement is governed by the laws of Texas.", "Keep it confidential."])
    assert p.shape == (2, len(LABELS)) and np.allclose(p.sum(1), 1, atol=1e-5)
    assert LABELS[int(p[0].argmax())] == "Governing Laws" and LABELS[int(p[1].argmax())] == "Confidentiality"
    sharp = p.max()
    (classical_dir / "calibration.json").write_text(json.dumps({"temperature": 3.0}))
    assert b.predict_proba(["This Agreement is governed by the laws of Texas."]).max() < sharp  # T>1 softens
    assert b.model_id == "ledgar-tfidf-lr-v1" and b.device == "cpu" and b.count_tokens("a b c") == 3


def test_top_features_are_readable_phrases(classical_dir):
    import joblib

    from classification.classical.explain import top_features

    tf = top_features(joblib.load(classical_dir / "model.joblib"), LABELS, k=5)
    assert set(tf) == set(LABELS) and len(tf["Governing Laws"]) == 5
    assert any("governed" in f["term"] or "laws" in f["term"] for f in tf["Governing Laws"])


def test_real_config_loads():
    cfg = load_classical_config("configs/classical/ledgar_tfidf_lr.yaml")
    assert cfg.run_name == "ledgar-tfidf-lr" and cfg.output_dir == "artifacts/classical/runs"
    assert cfg.data.label_groups == "configs/ledgar_label_groups.yaml" and cfg.model_version >= 1
    with pytest.raises(KeyError):
        from classification.classical.config import _merge

        _merge(cfg, {"nope": 1})


def test_registry_families(tmp_path, classical_dir):
    runs = tmp_path
    make_run(runs, "ledgar-qwen3-0.6b-v1", 100)
    models = list_models(runs)  # transformer v1 (created 100) + classical (created now, newer)
    assert {m.id: m.family for m in models} == {"ledgar-qwen3-0.6b-v1": "transformer", "ledgar-tfidf-lr-v1": "classical"}
    assert latest_model(models).id == "ledgar-qwen3-0.6b-v1"          # default stays the transformer
    assert resolve_model(models, None).id == "ledgar-qwen3-0.6b-v1"
    assert resolve_model(models, CLASSICAL_ALIAS).id == "ledgar-tfidf-lr-v1"
    assert resolve_model(models, "ledgar-tfidf-lr").id == "ledgar-tfidf-lr-v1"   # base name
    with pytest.raises(ModelNotFoundError, match="make classical"):
        resolve_model([m for m in models if m.family == "transformer"], CLASSICAL_ALIAS)


# --------------------------------------------------------------------------- POST /v1/compare

class _Tagged(StubBackend):
    def __init__(self, model_id, flip=False):
        self.model_id, self.flip = model_id, flip

    def predict_proba(self, texts):
        p = super().predict_proba(texts)
        if self.flip:  # disagree on clauses that mention "notice"
            for i, t in enumerate(texts):
                if "notice" in t.lower():
                    p[i] = np.roll(p[i], 1)
        return p


@pytest.fixture
def compare_client(tiny_data):
    store = StaticModelStore(
        {"ledgar-qwen3-0.6b-v1": Predictor(_Tagged("ledgar-qwen3-0.6b-v1")),
         "ledgar-tfidf-lr-v1": Predictor(_Tagged("ledgar-tfidf-lr-v1", flip=True))},
        latest="ledgar-qwen3-0.6b-v1", families={"ledgar-tfidf-lr-v1": "classical"})
    app = create_app(Settings(), model_store=store, dataset_store=DatasetStore(tiny_data))
    with TestClient(app) as c:
        yield c


CONTRACT = ("1. Governing Law. This Agreement shall be governed by the laws of New York and the courts there.\n\n"
            "2. Confidentiality. Each party shall keep all confidential information strictly secret at all times.\n\n"
            "3. Notices. All notices under this Agreement shall be in writing and delivered by courier promptly.")


def test_compare_defaults_to_latest_of_each_family(compare_client):
    r = compare_client.post("/v1/compare", json={"text": CONTRACT})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [(m["id"], m["family"]) for m in body["models"]] == [("ledgar-qwen3-0.6b-v1", "transformer"),
                                                                ("ledgar-tfidf-lr-v1", "classical")]
    assert body["num_clauses"] == 3
    assert all(len(m["clauses"]) == 3 for m in body["models"])            # same provisions for both
    assert body["agreement"]["per_clause"] == [True, True, False]          # differ on the notices clause
    assert body["agreement"]["clauses_agree"] == 2 and body["agreement"]["rate"] == round(2 / 3, 4)


def test_compare_messages_and_explicit_models(compare_client):
    r = compare_client.post("/v1/compare", json={"messages": [{"role": "user", "content": CONTRACT}],
                                                 "models": ["ledgar-tfidf-lr-v1"], "top_k": 1})
    body = r.json()
    assert [m["id"] for m in body["models"]] == ["ledgar-tfidf-lr-v1"] and body["agreement"] is None
    assert compare_client.post("/v1/compare", json={"text": "x", "models": ["nope-v9"]}).status_code == 404
    assert compare_client.post("/v1/compare", json={"text": "  "}).status_code == 400


def test_compare_needs_both_families(tiny_data):
    store = StaticModelStore({"ledgar-qwen3-0.6b-v1": Predictor(StubBackend())})
    with TestClient(create_app(Settings(), model_store=store, dataset_store=DatasetStore(tiny_data))) as c:
        r = c.post("/v1/compare", json={"text": CONTRACT})
        assert r.status_code == 503 and "make classical" in r.json()["error"]["message"]


def test_models_and_health_show_families(compare_client):
    data = compare_client.get("/v1/models").json()["data"]
    assert {m["id"]: m["family"] for m in data} == {"ledgar-qwen3-0.6b-v1": "transformer", "ledgar-tfidf-lr-v1": "classical"}
    h = compare_client.get("/health").json()
    assert h["latest_model"] == "ledgar-qwen3-0.6b-v1" and h["latest_classical"] == "ledgar-tfidf-lr-v1"
    # chat completions without a model still use the transformer
    r = compare_client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Governed by NY law."}]})
    assert r.json()["model"] == "ledgar-qwen3-0.6b-v1"


def test_model_pages_are_served(tiny_data, tmp_path):
    pages = tmp_path / "docs" / "models"
    (pages / "assets").mkdir(parents=True)
    (pages / "compare.html").write_text("<h1>Transformer vs classical</h1>")
    (pages / "assets" / "chart.png").write_bytes(b"\x89PNG fake")
    (tmp_path / "changelog.html").write_text("<h1>Changelog</h1>")
    store = StaticModelStore({"m": Predictor(StubBackend())})
    with TestClient(create_app(Settings(pages_dir=str(pages)), model_store=store, dataset_store=DatasetStore(tiny_data))) as c:
        assert "Transformer vs classical" in c.get("/models/compare").text
        assert c.get("/models/compare.html").status_code == 200 and c.get("/models").status_code == 200
        assert c.get("/models/assets/chart.png").content.startswith(b"\x89PNG")
        assert c.get("/models/assets/..%2F..%2Fchangelog.html").status_code == 404     # no path escape
        assert c.get("/models/assets/%2e%2e").status_code == 404
        r = c.get("/models/transformer")
        assert r.status_code == 404 and "make model-pages" in r.json()["error"]["message"]
        assert c.get("/models/nope").status_code == 404
        assert "Changelog" in c.get("/changelog.html").text
