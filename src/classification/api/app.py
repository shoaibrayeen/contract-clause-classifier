"""FastAPI application factory.

    uvicorn classification.api.app:create_app --factory --port 8000

OpenAPI schema: /openapi.json, Swagger UI: /docs, ReDoc: /redoc.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..config import DataConfig, resolve_path
from ..data import DatasetStore
from ..predict import Predictor, load_predictor
from .errors import install_error_handlers
from .routes import dataset, health, openai
from .settings import Settings

log = logging.getLogger("classification.api")


def create_app(settings: Settings | None = None, predictor: Predictor | None = None,
               dataset_store: DatasetStore | None = None) -> FastAPI:
    """`predictor` / `dataset_store` can be injected (tests); otherwise loaded from settings."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.started_at = time.time()
        if app.state.predictor is None:
            model_dir = resolve_path(settings.model_dir)
            try:
                app.state.predictor = load_predictor(
                    model_dir, device=settings.device, top_k=settings.top_k,
                    threshold=settings.threshold, batch_size=settings.batch_size,
                )
                app.state.load_error = None
            except Exception as exc:  # keep serving /health and /v1/dataset
                log.error("Model failed to load from %s: %s", model_dir, exc)
                app.state.load_error = str(exc)
        yield

    app = FastAPI(
        title="Contract Clause Classifier",
        version=__version__,
        description=(
            "Classifies contract provisions into 100 LEDGAR clause types with calibrated confidences. "
            "OpenAI-compatible: point any OpenAI SDK at `/v1` and call `chat.completions.create`."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.predictor = predictor
    app.state.load_error = None
    app.state.started_at = time.time()
    app.state.dataset = dataset_store or DatasetStore(DataConfig(data_dir=settings.data_dir))

    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"],
    )
    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(openai.router)
    app.include_router(dataset.router)
    return app
