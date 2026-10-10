"""OpenAI-compatible endpoints: GET /v1/models, POST /v1/chat/completions.

The contract text is the last `user` message. The assistant reply content is a JSON string:
    {"document": [{"name", "confidence"}], "clauses": [{"index", "text", "predictions": [...]}], "num_clauses"}
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Depends, Request

from ..errors import APIError
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


@router.get("/models", response_model=ModelList)
def list_models(request: Request) -> ModelList:
    s = request.app.state
    return ModelList(data=[ModelCard(id=s.settings.model_id, created=int(s.started_at))])


@router.get("/models/{model_id}", response_model=ModelCard, responses=ERRORS)
def get_model(model_id: str, request: Request) -> ModelCard:
    s = request.app.state
    if model_id != s.settings.model_id:
        raise APIError(404, f"The model '{model_id}' does not exist", "model", "model_not_found")
    return ModelCard(id=model_id, created=int(s.started_at))


@router.post("/chat/completions", response_model=ChatCompletionResponse, responses=ERRORS)
def chat_completions(body: ChatCompletionRequest, request: Request) -> ChatCompletionResponse:
    s = request.app.state
    if body.model and body.model != s.settings.model_id:
        raise APIError(404, f"The model '{body.model}' does not exist; use '{s.settings.model_id}'", "model", "model_not_found")
    if body.stream:
        raise APIError(400, "stream=true is not supported by this classifier", "stream", "unsupported_parameter")
    if body.n not in (None, 1):
        raise APIError(400, "Only n=1 is supported", "n", "unsupported_parameter")
    if s.predictor is None:
        raise APIError(503, f"Model not loaded: {s.load_error or 'unknown error'}", None, "model_not_loaded")

    user_msgs = [m for m in body.messages if m.role == "user"]
    if not user_msgs or not user_msgs[-1].text().strip():
        raise APIError(400, "Provide the contract text as the last user message", "messages", "missing_user_message")
    text = user_msgs[-1].text()
    if len(text) > s.settings.max_input_chars:
        raise APIError(413, f"Input exceeds {s.settings.max_input_chars} characters", "messages", "input_too_large")

    predictor = s.predictor
    segment = body.segment if body.segment is not None else len(text.split()) > 350
    result = predictor.predict_document(text, top_k=body.top_k, threshold=body.threshold, segment=segment)
    content = json.dumps(result, ensure_ascii=False)
    prompt_tokens = predictor.count_tokens(text)
    completion_tokens = predictor.count_tokens(content)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=s.settings.model_id,
        choices=[Choice(message=ResponseMessage(content=content))],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens),
        system_fingerprint=f"fp_{predictor.backend.model_id}"[:64] if hasattr(predictor.backend, "model_id") else None,
    )
