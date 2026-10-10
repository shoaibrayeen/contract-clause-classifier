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
from .models import ModelStore, StaticModelStore
from .routes import dataset, health, openai, playground
from .settings import Settings

log = logging.getLogger("classification.api")


def create_app(settings: Settings | None = None, predictor: Predictor | None = None,
               dataset_store: DatasetStore | None = None, model_store: ModelStore | None = None) -> FastAPI:
    """Build the app. Inject `model_store` (or a single `predictor`) and `dataset_store` in tests;
    otherwise every completed run under RUNS_DIR is served and the latest is preloaded."""
    settings = settings or Settings()
    if model_store is None and predictor is not None:
        model_store = StaticModelStore({settings.model_id: predictor}, aliases=settings.aliases)
    if model_store is None:
        model_store = ModelStore(
            resolve_path(settings.runs_dir),
            loader=lambda path: load_predictor(path, device=settings.device, top_k=settings.top_k,
                                               threshold=settings.threshold, batch_size=settings.batch_size),
            aliases=settings.aliases, max_loaded=settings.max_loaded_models, rescan_s=settings.rescan_seconds,
            extra_dirs=[resolve_path(settings.model_dir)] if settings.model_dir else [],
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.started_at = time.time()
        app.state.models.preload_latest()
        yield

    app = FastAPI(
        title="Contract Clause Classifier",
        version=__version__,
        description=(
            "Classifies contract provisions into 100 LEDGAR clause types with calibrated confidences. "
            "OpenAI-compatible: point any OpenAI SDK at `/v1` and call `chat.completions.create`. "
            "Every trained version is served; omit `model` (or send `latest`) for the newest one."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.models = model_store
    app.state.started_at = time.time()
    groups = settings.label_groups if settings.label_groups and resolve_path(settings.label_groups).exists() else None
    app.state.dataset = dataset_store or DatasetStore(DataConfig(data_dir=settings.data_dir, label_groups=groups))

    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"],
    )
    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(openai.router)
    app.include_router(dataset.router)
    app.include_router(playground.router)
    return app
