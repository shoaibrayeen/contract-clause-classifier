import json

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
    app = create_app(Settings(model_dir=str(tmp_path / "missing")), dataset_store=DatasetStore(tiny_data))
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
