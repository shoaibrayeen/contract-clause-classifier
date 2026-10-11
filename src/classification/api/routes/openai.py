"""OpenAI-compatible endpoints: GET /v1/models, POST /v1/chat/completions.

Model selection: no `model` (or "latest" / the MODEL_ID alias) -> latest trained version; an exact id such as
"ledgar-qwen3-0.6b-lora-v1" -> that version; a base name -> its latest version. The response's `model` field is
always the concrete version that answered.

The contract text is the last `user` message. The assistant reply content is a JSON string:
    {"document": [{"name", "confidence"}], "clauses": [{"index", "text", "predictions": [...]}], "num_clauses"}
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Depends, Request

from ...registry import ModelInfo, ModelNotFoundError
from ..errors import APIError
from ..models import NoModelsError
from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ErrorResponse,
    ModelCard,
    ModelList,
    ResponseMessage,
    Usage,
)
from ..security import require_api_key

router = APIRouter(prefix="/v1", tags=["openai"], dependencies=[Depends(require_api_key)])

ERRORS = {400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 404: {"model": ErrorResponse},
          422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}


def _card(info: ModelInfo, latest_id: str | None, loaded: bool) -> ModelCard:
    test = info.info.get("test") or {}
    metrics = {k: v for k, v in {
        "best_val_accuracy": info.info.get("best_val_accuracy"),
        "best_val_macro_f1": info.info.get("best_val_macro_f1"),
        "test_accuracy": test.get("accuracy"), "test_top3_accuracy": test.get("top3_accuracy"),
        "test_macro_f1": test.get("macro_f1"),
    }.items() if v is not None} or None
    return ModelCard(id=info.id, created=int(info.created), version=info.version, family=info.family,
                     is_latest=info.id == latest_id,
                     base_name=info.base_name, base_model=info.info.get("base_model"), loaded=loaded, metrics=metrics)


def _resolve(request: Request, requested: str | None) -> ModelInfo:
    store = request.app.state.models
    try:
        return store.resolve(requested)
    except NoModelsError as exc:
        raise APIError(503, str(exc), "model", "model_not_loaded") from exc
    except ModelNotFoundError as exc:
        raise APIError(404, str(exc), "model", "model_not_found") from exc


@router.get("/models", response_model=ModelList)
def list_models(request: Request) -> ModelList:
    """Every trained version, newest first; `is_latest` marks the default model."""
    store = request.app.state.models
    models = store.models()
    latest = store.latest()
    loaded = store.loaded()
    cards = [_card(m, latest.id if latest else None, m.id in loaded) for m in reversed(models)]
    return ModelList(data=cards)


@router.get("/models/{model_id}", response_model=ModelCard, responses=ERRORS)
def get_model(model_id: str, request: Request) -> ModelCard:
    """A version by id; aliases ("latest", MODEL_ID) and base names resolve to the matching version."""
    info = _resolve(request, model_id)
    store = request.app.state.models
    latest = store.latest()
    return _card(info, latest.id if latest else None, store.is_loaded(info.id))


@router.post("/chat/completions", response_model=ChatCompletionResponse, responses=ERRORS)
def chat_completions(body: ChatCompletionRequest, request: Request) -> ChatCompletionResponse:
    s = request.app.state
    if body.stream:
        raise APIError(400, "stream=true is not supported by this classifier", "stream", "unsupported_parameter")
    if body.n not in (None, 1):
        raise APIError(400, "Only n=1 is supported", "n", "unsupported_parameter")

    user_msgs = [m for m in body.messages if m.role == "user"]
    if not user_msgs or not user_msgs[-1].text().strip():
        raise APIError(400, "Provide the contract text as the last user message", "messages", "missing_user_message")
    text = user_msgs[-1].text()
    if len(text) > s.settings.max_input_chars:
        raise APIError(413, f"Input exceeds {s.settings.max_input_chars} characters", "messages", "input_too_large")

    _resolve(request, body.model)  # 404 / 503 before loading anything
    try:
        info, predictor = s.models.get(body.model)
    except ModelNotFoundError as exc:
        raise APIError(404, str(exc), "model", "model_not_found") from exc
    except Exception as exc:  # model files present but unloadable
        raise APIError(503, f"Model could not be loaded: {exc}", "model", "model_not_loaded") from exc

    # Splitting is safe for a single clause (it stays one provision), so it is on unless disabled.
    segment = True if body.segment is None else body.segment
    result = predictor.predict_document(text, top_k=body.top_k, threshold=body.threshold, segment=segment)
    content = json.dumps(result, ensure_ascii=False)
    prompt_tokens = predictor.count_tokens(text)
    completion_tokens = predictor.count_tokens(content)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=info.id,
        choices=[Choice(message=ResponseMessage(content=content))],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens),
        system_fingerprint=f"fp_{info.id}"[:64],
    )
