"""Browser playground: GET /playground (and / redirects there).

A single self-contained page (no external assets) that calls the same OpenAI-compatible endpoints a
client would: /health, /v1/models and /v1/chat/completions. Works as soon as a trained model exists.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(tags=["playground"])


@lru_cache(maxsize=1)
def _page() -> str:
    return resources.files("classification.api").joinpath("static/playground.html").read_text(encoding="utf-8")


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/playground")


@router.get("/playground", response_class=HTMLResponse, summary="Interactive playground")
def playground() -> HTMLResponse:
    """Paste contract text, pick a model version, and see clause types with confidences."""
    return HTMLResponse(_page())
