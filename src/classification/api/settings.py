"""API settings from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # Every completed run under RUNS_DIR is servable; requests pick one by id (default: latest).
    runs_dir: str = field(default_factory=lambda: _env("RUNS_DIR", "artifacts/runs"))
    # Classical (TF-IDF + linear) models, served next to the transformer and used by POST /v1/compare.
    classical_runs_dir: str = field(default_factory=lambda: _env("CLASSICAL_RUNS_DIR", "artifacts/classical/runs"))
    # Optional extra model directory outside RUNS_DIR (e.g. a copied merged/ folder).
    model_dir: str | None = field(default_factory=lambda: os.environ.get("MODEL_DIR") or None)
    # Alias that always means "latest" (kept for clients written before versioning), besides "latest".
    model_id: str = field(default_factory=lambda: _env("MODEL_ID", "clause-classifier-qwen3-0.6b"))
    # Generated model pages (scripts/10_model_pages.py), served at /models/*.
    pages_dir: str = field(default_factory=lambda: _env("PAGES_DIR", "docs/models"))
    max_loaded_models: int = field(default_factory=lambda: int(_env("MAX_LOADED_MODELS", "2")))
    rescan_seconds: float = field(default_factory=lambda: float(_env("MODEL_RESCAN_SECONDS", "5")))
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", "data"))
    # Label grouping used by /v1/dataset and /v1/labels (same file as data.label_groups in the training config).
    label_groups: str | None = field(default_factory=lambda: _env("LABEL_GROUPS", "configs/ledgar_label_groups.yaml") or None)
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "3")))
    threshold: float = field(default_factory=lambda: float(_env("THRESHOLD", "0.30")))
    batch_size: int = field(default_factory=lambda: int(_env("BATCH_SIZE", "16")))
    api_key: str | None = field(default_factory=lambda: os.environ.get("API_KEY") or None)
    max_input_chars: int = field(default_factory=lambda: int(_env("MAX_INPUT_CHARS", "2000000")))
    cors_origins: list[str] = field(default_factory=lambda: [o for o in _env("CORS_ORIGINS", "*").split(",") if o])

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(a for a in ("latest", self.model_id) if a))
