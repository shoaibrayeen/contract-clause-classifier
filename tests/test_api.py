import json
import time

import pytest
from fastapi.testclient import TestClient

from classification.api.app import create_app
from classification.api.settings import Settings
from classification.data import DatasetStore
from tests_helpers import assert_prediction_list


@pytest.fixture
def client(predictor, tiny_data):
    app = create_app(Settings(model_id="clause-test"), predictor=predictor, dataset_store=DatasetStore(tiny_data))
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] and body["num_labels"] == 5


def test_health_unavailable_without_model(tiny_data, tmp_path):
    app = create_app(Settings(runs_dir=str(tmp_path / "runs"), classical_runs_dir=str(tmp_path / "classical")),
                     dataset_store=DatasetStore(tiny_data))
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 503 and r.json()["status"] == "unavailable"
        # Dataset still works without a model.
        assert c.get("/v1/dataset").status_code == 200
        # Chat returns an OpenAI-style 503 error.
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_not_loaded"


def test_models(client):
    r = client.get("/v1/models")
    assert r.json()["object"] == "list" and r.json()["data"][0]["id"] == "clause-test"
    assert client.get("/v1/models/clause-test").status_code == 200
    assert client.get("/v1/models/nope").json()["error"]["type"] == "not_found_error"


def test_chat_completion_openai_shape(client):
    r = client.post("/v1/chat/completions", json={
        "model": "clause-test",
        "messages": [{"role": "system", "content": "ignored"},
                     {"role": "user", "content": "This Agreement shall be governed by the laws of New York."}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    result = json.loads(body["choices"][0]["message"]["content"])
    assert result["document"][0]["name"] == "Governing Laws"
    assert_prediction_list(result["document"])


def test_chat_content_parts_and_extras(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Either party may terminate."}]}],
        "top_k": 1, "threshold": 0.5,
    })
    result = json.loads(r.json()["choices"][0]["message"]["content"])
    assert len(result["clauses"][0]["predictions"]) == 1


@pytest.mark.parametrize("payload,code", [
    ({"messages": [{"role": "user", "content": "x"}], "stream": True}, 400),
    ({"messages": [{"role": "user", "content": "x"}], "model": "other"}, 404),
    ({"messages": [{"role": "system", "content": "x"}]}, 400),
    ({"messages": []}, 422),
])
def test_chat_errors(client, payload, code):
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == code
    assert set(r.json()["error"]) == {"message", "type", "param", "code"}


def test_dataset_pagination(client):
    r = client.get("/v1/dataset")
    body = r.json()
    assert r.status_code == 200 and body["limit"] == 10 and len(body["items"]) == 10 and body["next_offset"] == 10
    body2 = client.get("/v1/dataset", params={"split": "train", "offset": body["next_offset"]}).json()
    assert body2["items"][0]["id"] == "train-10"
    assert client.get("/v1/dataset", params={"split": "validation"}).json()["total"] == 7
    assert client.get("/v1/dataset", params={"split": "bad"}).status_code == 422
    assert client.get("/v1/dataset", params={"limit": 101}).status_code == 422


def test_labels(client):
    assert client.get("/v1/labels").json()["total"] == 5


def test_api_key(predictor, tiny_data):
    app = create_app(Settings(api_key="secret"), predictor=predictor, dataset_store=DatasetStore(tiny_data))
    with TestClient(app) as c:
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
        assert c.get("/health").status_code == 200  # health is open


def test_openapi_schema(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/health", "/v1/models", "/v1/chat/completions", "/v1/dataset", "/v1/labels"):
        assert p in paths


def test_openai_sdk_compat(client):
    """The official OpenAI SDK can talk to the API (routed through the in-process TestClient)."""
    openai = pytest.importorskip("openai")
    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=client)
    resp = sdk.chat.completions.create(
        model="clause-test",
        messages=[{"role": "user", "content": "Each party shall keep information confidential."}],
    )
    result = json.loads(resp.choices[0].message.content)
    assert result["document"][0]["name"] == "Confidentiality"


# --------------------------------------------------------------------------- multi-version serving

class _Labelled:
    """Stub backend that tags every prediction with its model id, to prove which version answered."""

    def __init__(self, model_id):
        from conftest import StubBackend

        self.inner = StubBackend()
        self.model_id = model_id
        self.label_names = self.inner.label_names
        self.device = "cpu"

    def predict_proba(self, texts):
        return self.inner.predict_proba(texts)

    def count_tokens(self, text):
        return self.inner.count_tokens(text)


@pytest.fixture
def versioned_client(tiny_data, tmp_path):
    """Two versions on disk, loaded through the real ModelStore with a stub loader."""
    from classification.api.models import ModelStore
    from classification.predict import Predictor
    from conftest import make_run

    runs = tmp_path / "runs"
    make_run(runs, "ledgar-qwen3-0.6b-lora-v1", 100)
    make_run(runs, "ledgar-qwen3-0.6b-lora-v2", 200)
    loads = []

    def loader(path):
        loads.append(path.parent.name)
        return Predictor(_Labelled(path.parent.name))

    settings = Settings(runs_dir=str(runs))
    store = ModelStore(runs, loader, aliases=settings.aliases, max_loaded=1, rescan_s=0)
    app = create_app(settings, model_store=store, dataset_store=DatasetStore(tiny_data))
    with TestClient(app) as c:
        c.loads, c.runs = loads, runs
        yield c


def _chat(client, **extra):
    return client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Governed by the laws of New York."}], **extra})


def test_default_model_is_latest(versioned_client):
    r = _chat(versioned_client)
    assert r.status_code == 200 and r.json()["model"] == "ledgar-qwen3-0.6b-lora-v2"
    assert r.json()["system_fingerprint"] == "fp_ledgar-qwen3-0.6b-lora-v2"


@pytest.mark.parametrize("requested", ["latest", "clause-classifier-qwen3-0.6b", "ledgar-qwen3-0.6b-lora", ""])
def test_aliases_and_base_name_resolve_to_latest(versioned_client, requested):
    assert _chat(versioned_client, model=requested).json()["model"] == "ledgar-qwen3-0.6b-lora-v2"


def test_explicit_version_is_used(versioned_client):
    r = _chat(versioned_client, model="ledgar-qwen3-0.6b-lora-v1")
    assert r.status_code == 200 and r.json()["model"] == "ledgar-qwen3-0.6b-lora-v1"


def test_unknown_version_404_lists_available(versioned_client):
    r = _chat(versioned_client, model="ledgar-qwen3-0.6b-lora-v9")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "model_not_found" and "ledgar-qwen3-0.6b-lora-v1" in err["message"]


def test_models_endpoint_lists_all_versions(versioned_client):
    data = versioned_client.get("/v1/models").json()["data"]
    assert [m["id"] for m in data] == ["ledgar-qwen3-0.6b-lora-v2", "ledgar-qwen3-0.6b-lora-v1"]
    assert [m["is_latest"] for m in data] == [True, False] and data[0]["version"] == 2
    card = versioned_client.get("/v1/models/latest").json()
    assert card["id"] == "ledgar-qwen3-0.6b-lora-v2"


def test_health_reports_versions(versioned_client):
    body = versioned_client.get("/health").json()
    assert body["status"] == "ok" and body["latest_model"] == "ledgar-qwen3-0.6b-lora-v2"
    assert body["latest_version"] == 2 and len(body["available_models"]) == 2
    assert body["loaded_models"] == ["ledgar-qwen3-0.6b-lora-v2"]  # preloaded at startup


def test_lru_and_lazy_loading(versioned_client):
    _chat(versioned_client, model="ledgar-qwen3-0.6b-lora-v1")
    _chat(versioned_client, model="ledgar-qwen3-0.6b-lora-v1")
    assert versioned_client.loads == ["ledgar-qwen3-0.6b-lora-v2", "ledgar-qwen3-0.6b-lora-v1"]  # loaded once each
    assert versioned_client.get("/health").json()["loaded_models"] == ["ledgar-qwen3-0.6b-lora-v1"]  # max_loaded=1


def test_newly_trained_model_becomes_latest_without_restart(versioned_client):
    from conftest import make_run

    make_run(versioned_client.runs, "ledgar-qwen3-0.6b-lora-v3", time.time())
    assert _chat(versioned_client).json()["model"] == "ledgar-qwen3-0.6b-lora-v3"


def test_openai_sdk_selects_versions(versioned_client):
    openai = pytest.importorskip("openai")
    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=versioned_client)
    msgs = [{"role": "user", "content": "Each party shall keep information confidential."}]
    assert sdk.chat.completions.create(model="latest", messages=msgs).model == "ledgar-qwen3-0.6b-lora-v2"
    assert sdk.chat.completions.create(model="ledgar-qwen3-0.6b-lora-v1", messages=msgs).model == "ledgar-qwen3-0.6b-lora-v1"
    assert [m.id for m in sdk.models.list().data][0] == "ledgar-qwen3-0.6b-lora-v2"


def test_playground_page(client):
    r = client.get("/playground")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Clause Classifier Playground" in r.text and "/v1/chat/completions" in r.text
    root = client.get("/", follow_redirects=False)
    assert root.status_code in (302, 307) and root.headers["location"] == "/playground"


def test_short_multi_section_contract_is_split(client):
    text = ("1. Confidentiality. Each party shall keep confidential information secret at all times.\n\n"
            "2. Termination. Either party may terminate this agreement upon thirty days notice.")
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": text}]})
    result = json.loads(r.json()["choices"][0]["message"]["content"])
    assert result["num_clauses"] == 2
    one = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": text}], "segment": False})
    assert json.loads(one.json()["choices"][0]["message"]["content"])["num_clauses"] == 1
