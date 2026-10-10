"""GET /health — 200 when the model is loaded, 503 otherwise (dataset routes still work)."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ... import __version__
from ..schemas import Health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health, responses={503: {"model": Health}})
def health(request: Request):
    state = request.app.state
    predictor = state.predictor
    body = Health(
        status="ok" if predictor else "unavailable",
        model_loaded=predictor is not None,
        model=state.settings.model_id,
        model_dir=str(state.settings.model_dir),
        device=getattr(predictor.backend, "device", None) if predictor else None,
        num_labels=len(predictor.label_names) if predictor else None,
        uptime_s=round(time.time() - state.started_at, 3),
        version=__version__,
        detail=state.load_error,
    )
    return JSONResponse(body.model_dump(), status_code=200 if predictor else 503)
