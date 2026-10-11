"""GET /health — 200 when the latest model is loaded, 503 otherwise (dataset routes still work)."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ... import __version__
from ...config import display_path
from ..schemas import Health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health, responses={503: {"model": Health}})
def health(request: Request):
    state = request.app.state
    store = state.models
    models = store.models()
    latest = store.latest()
    loaded = store.loaded()
    latest_pred = loaded.get(latest.id) if latest else None
    detail = store.last_error
    if latest is None:
        detail = detail or f"No trained models in {display_path(store.runs_dir)}. Train one with `make all`."
    body = Health(
        status="ok" if latest_pred else "unavailable",
        model_loaded=latest_pred is not None,
        model=latest.id if latest else None,
        latest_model=latest.id if latest else None,
        latest_version=latest.version if latest else None,
        latest_classical=(lambda c: c.id if c else None)(store.latest_of("classical")) if hasattr(store, "latest_of") else None,
        available_models=[m.id for m in models],
        loaded_models=list(loaded),
        models_dir=display_path(store.runs_dir),
        device=getattr(latest_pred.backend, "device", None) if latest_pred else None,
        num_labels=len(latest_pred.label_names) if latest_pred else None,
        uptime_s=round(time.time() - state.started_at, 3),
        version=__version__,
        detail=None if latest_pred else detail,
    )
    return JSONResponse(body.model_dump(), status_code=200 if latest_pred else 503)
