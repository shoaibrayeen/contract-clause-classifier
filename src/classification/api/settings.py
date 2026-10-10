"""API settings from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    model_dir: str = field(default_factory=lambda: _env("MODEL_DIR", "artifacts/best"))
    model_id: str = field(default_factory=lambda: _env("MODEL_ID", "clause-classifier-qwen3-0.6b"))
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", "data"))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "3")))
    threshold: float = field(default_factory=lambda: float(_env("THRESHOLD", "0.30")))
    batch_size: int = field(default_factory=lambda: int(_env("BATCH_SIZE", "16")))
    api_key: str | None = field(default_factory=lambda: os.environ.get("API_KEY") or None)
    max_input_chars: int = field(default_factory=lambda: int(_env("MAX_INPUT_CHARS", "2000000")))
    cors_origins: list[str] = field(default_factory=lambda: [o for o in _env("CORS_ORIGINS", "*").split(",") if o])
