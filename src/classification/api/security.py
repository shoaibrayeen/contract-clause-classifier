"""Optional bearer-token auth. Enabled only when API_KEY is set (OpenAI SDKs send `Authorization: Bearer ...`)."""

from __future__ import annotations

import secrets

from fastapi import Request

from .errors import APIError


def require_api_key(request: Request) -> None:
    expected = request.app.state.settings.api_key
    if not expected:
        return
    header = request.headers.get("authorization", "")
    token = header[7:] if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    if not token or not secrets.compare_digest(token, expected):
        raise APIError(401, "Invalid or missing API key", None, "invalid_api_key")
