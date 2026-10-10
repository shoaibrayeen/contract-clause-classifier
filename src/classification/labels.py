"""Label map (id <-> clause-type name) persisted as JSON next to models and data."""

from __future__ import annotations

import json
from pathlib import Path


def save_labels(names: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"names": names}, indent=2))


def load_labels(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text())
    names = data["names"] if isinstance(data, dict) else data
    if not names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"Invalid label file: {path}")
    return names


def id2label(names: list[str]) -> dict[int, str]:
    return dict(enumerate(names))


def label2id(names: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(names)}
