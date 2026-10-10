"""Multi-version model store for the API.

All completed runs under RUNS_DIR are servable (see classification/registry.py):
    - request without `model`, or with an alias ("latest", MODEL_ID)  -> latest model
    - request with an exact id, e.g. "ledgar-qwen3-0.6b-lora-v1"     -> that version
    - request with a base name, e.g. "ledgar-qwen3-0.6b-lora"         -> latest version of that base

Models load lazily on first use and are kept in a small LRU cache (MAX_LOADED_MODELS); the latest is
preloaded at startup. The runs directory is rescanned every few seconds, so a model trained while the
server is running becomes available (and becomes "latest") without a restart.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path

from ..config import display_path
from ..predict import Predictor
from ..registry import ModelInfo, ModelNotFoundError, latest_model, list_models, read_model_info, resolve_model

log = logging.getLogger(__name__)

Loader = Callable[[Path], Predictor]


class NoModelsError(ModelNotFoundError):
    """No trained model exists at all (API answers 503 rather than 404)."""


class ModelStore:
    def __init__(self, runs_dir: str | Path, loader: Loader, aliases: Iterable[str] = ("latest",),
                 max_loaded: int = 2, rescan_s: float = 5.0, extra_dirs: Iterable[str | Path] = ()):
        self.runs_dir = Path(runs_dir)
        self.loader = loader
        self.aliases = tuple(a for a in aliases if a)
        self.max_loaded = max(1, max_loaded)
        self.rescan_s = rescan_s
        self.extra_dirs = [Path(p) for p in extra_dirs]
        self._models: list[ModelInfo] = []
        self._scanned_at = 0.0
        self._loaded: OrderedDict[str, Predictor] = OrderedDict()
        self._lock = threading.RLock()
        self.last_error: str | None = None

    # ------------------------------------------------------------------ discovery
    def models(self, refresh: bool = False) -> list[ModelInfo]:
        with self._lock:
            if refresh or time.time() - self._scanned_at >= self.rescan_s:
                found = list_models(self.runs_dir)
                seen = {m.path.resolve() for m in found}
                for d in self.extra_dirs:  # e.g. MODEL_DIR pointing outside RUNS_DIR
                    run_dir = d.parent if d.name == "merged" else d
                    if (run_dir / "merged" / "config.json").exists() and (run_dir / "merged").resolve() not in seen:
                        found.append(read_model_info(run_dir))
                self._models = sorted(found, key=lambda m: (m.created, m.version or 0, m.id))
                self._scanned_at = time.time()
            return list(self._models)

    def latest(self) -> ModelInfo | None:
        return latest_model(self.models())

    def resolve(self, requested: str | None) -> ModelInfo:
        models = self.models()
        if not models:
            raise NoModelsError(f"No trained models in {display_path(self.runs_dir)}. Train one with `make all`.")
        try:
            return resolve_model(models, requested, self.aliases)
        except ModelNotFoundError:
            # A model trained a moment ago may not be in the cached scan yet.
            return resolve_model(self.models(refresh=True), requested, self.aliases)

    # ------------------------------------------------------------------ loading
    def get(self, requested: str | None = None) -> tuple[ModelInfo, Predictor]:
        info = self.resolve(requested)
        with self._lock:
            if info.id in self._loaded:
                self._loaded.move_to_end(info.id)
                return info, self._loaded[info.id]
            log.info("Loading model %s from %s", info.id, info.path)
            predictor = self.loader(info.path)
            self._loaded[info.id] = predictor
            while len(self._loaded) > self.max_loaded:
                evicted, _ = self._loaded.popitem(last=False)
                log.info("Unloaded model %s (MAX_LOADED_MODELS=%d)", evicted, self.max_loaded)
            return info, predictor

    def loaded(self) -> dict[str, Predictor]:
        with self._lock:
            return dict(self._loaded)

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    def preload_latest(self) -> None:
        try:
            self.get(None)
            self.last_error = None
        except Exception as exc:  # keep serving /health and /v1/dataset
            self.last_error = str(exc)
            log.error("Could not load the latest model: %s", exc)


def _no_loader(path: Path) -> Predictor:
    raise RuntimeError(f"StaticModelStore cannot load models from disk ({path})")


class StaticModelStore(ModelStore):
    """Fixed, already-loaded predictors (tests, or embedding the API in another program)."""

    def __init__(self, predictors: dict[str, Predictor], latest: str | None = None,
                 aliases: Iterable[str] = ("latest",), versions: dict[str, int | None] | None = None):
        super().__init__(Path("."), loader=_no_loader, aliases=aliases, max_loaded=max(1, len(predictors)))
        from ..registry import split_name

        order = list(predictors)
        if latest is not None:
            order.remove(latest)
            order.append(latest)
        now = time.time()
        self._models = []
        for i, mid in enumerate(order):
            base, version = split_name(mid)
            if versions and mid in versions:
                version = versions[mid]
            self._models.append(ModelInfo(id=mid, run_dir=Path(mid), base_name=base, version=version,
                                          created=now - (len(order) - i), info={}))
        self._loaded = OrderedDict((mid, predictors[mid]) for mid in order)

    def models(self, refresh: bool = False) -> list[ModelInfo]:
        return list(self._models)
