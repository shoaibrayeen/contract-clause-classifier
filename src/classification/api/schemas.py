"""Pydantic schemas: OpenAI chat-completions (subset), dataset pages, health, predictions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- OpenAI-compatible

class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    text: str | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool", "developer", "function"]
    content: str | list[ContentPart] | None = None

    def text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(p.text or "" for p in self.content if p.type in ("text", "input_text"))


class ChatCompletionRequest(BaseModel):
    """OpenAI `POST /v1/chat/completions` body. Unknown OpenAI params are accepted and ignored.

    `model`: omit it (or send "latest" / the MODEL_ID alias) for the latest trained model, send a version id
    such as "ledgar-qwen3-0.6b-lora-v1" for that version, or a base name for its latest version.

    Extra (non-OpenAI) knobs, passable via the SDK's `extra_body`:
        top_k, threshold, segment
    """

    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool | None = False
    temperature: float | None = None
    n: int | None = 1
    response_format: dict[str, Any] | None = None
    user: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    segment: bool | None = None  # split into provisions (default true); false = classify the input as one clause


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    refusal: str | None = None


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop", "length"] = "stop"
    logprobs: None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: str | None = None


class ModelCard(BaseModel):
    """OpenAI model object plus non-standard fields (clients ignore unknown fields)."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "local"
    version: int | None = None
    family: str = "transformer"  # transformer | classical
    is_latest: bool = False
    base_name: str | None = None
    base_model: str | None = None
    loaded: bool = False
    metrics: dict[str, Any] | None = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class ErrorBody(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# --------------------------------------------------------------------------- predictions

class ClausePrediction(BaseModel):
    name: str
    confidence: float


class ClauseResult(BaseModel):
    index: int
    text: str
    predictions: list[ClausePrediction]


class ClassificationResult(BaseModel):
    """Shape of the JSON string returned in `choices[0].message.content`."""

    document: list[ClausePrediction]
    clauses: list[ClauseResult]
    num_clauses: int


# --------------------------------------------------------------------------- dataset / health

class DatasetItem(BaseModel):
    id: str
    text: str
    label_id: int
    label: str                       # clause type the model is trained on (after label grouping)
    original_label: str | None = None  # LEDGAR's original heading label


class DatasetPage(BaseModel):
    split: Literal["train", "validation", "test"]
    offset: int
    limit: int
    total: int
    next_offset: int | None
    items: list[DatasetItem]


class LabelList(BaseModel):
    object: Literal["list"] = "list"
    total: int
    data: list[str]


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model: str | None
    latest_model: str | None = None
    latest_version: int | None = None
    latest_classical: str | None = None
    available_models: list[str] = []
    loaded_models: list[str] = []
    models_dir: str
    device: str | None = None
    num_labels: int | None = None
    uptime_s: float
    version: str
    detail: str | None = None


# --------------------------------------------------------------------------- compare (transformer vs classical)

class CompareRequest(BaseModel):
    """POST /v1/compare: run several models on the same provisions.

    Give the text either as `text` or OpenAI-style `messages` (last user message). `models` defaults to the
    latest transformer and the latest classical model.
    """

    model_config = ConfigDict(extra="allow")
    text: str | None = None
    messages: list[ChatMessage] | None = None
    models: list[str] | None = Field(default=None, min_length=1, max_length=4)
    top_k: int | None = Field(default=None, ge=1, le=100)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    segment: bool | None = None


class CompareModelResult(BaseModel):
    id: str
    family: str
    version: int | None = None
    latency_ms: float
    document: list[ClausePrediction]
    clauses: list[ClauseResult]


class CompareAgreement(BaseModel):
    clauses_agree: int
    clauses_total: int
    rate: float | None
    per_clause: list[bool]


class CompareResponse(BaseModel):
    object: Literal["clause.comparison"] = "clause.comparison"
    num_clauses: int
    models: list[CompareModelResult]
    agreement: CompareAgreement | None = None
