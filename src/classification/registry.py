"""Model versions and the local model registry.

Every trained model lives in its own run directory and is never overwritten:

    artifacts/runs/<run_name>-v<N>/merged/      serving model for version N
    artifacts/runs/<run_name>-v<N>/merged/model_info.json   id, version, created, metrics

`model_version` in the YAML config is the version the *next* training run will produce
(default 1). `make all` trains `<run_name>-v<model_version>`, evaluates it, renders its GIF and
then bumps `model_version` by one, so the next run becomes `-v<N+1>` and all versions are kept.

The registry has no separate index file: the completed run directories *are* the registry.
A run is complete when `merged/config.json` exists. `artifacts/best` is a convenience symlink to
the latest model.

"Latest" = the most recently created versioned model; unversioned runs (e.g. `sanity`, or runs
named explicitly with --run-name) are only chosen when no versioned model exists.

Model families: `model_info.json["family"]` is "transformer" (default, also for models written before
families existed) or "classical" (TF-IDF + linear models, `artifacts/classical/runs/`). `latest` prefers the
transformer family; `latest-classical` resolves to the newest classical model.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

MODEL_INFO_FILE = "model_info.json"
VERSION_RE = re.compile(r"^(?P<base>.+)-v(?P<version>\d+)$")
CONFIG_VERSION_RE = re.compile(r"^(?P<key>model_version:[ \t]*)(?P<num>\d+)", re.MULTILINE)
DEFAULT_ALIASES = ("latest",)
CLASSICAL_ALIAS = "latest-classical"
FAMILIES = ("transformer", "classical")


class RegistryError(Exception):
    """Base class for registry problems that should be shown to the user as-is."""


class UnfinishedRunError(RegistryError):
    pass


class ModelNotFoundError(RegistryError):
    pass


@dataclass
class ModelInfo:
    id: str
    run_dir: Path
    base_name: str
    version: int | None
    created: float
    info: dict[str, Any] = field(default_factory=dict)
    family: str = "transformer"

    @property
    def path(self) -> Path:
        """The servable (merged) model directory."""
        return self.run_dir / "merged"

    @property
    def versioned(self) -> bool:
        return self.version is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "version": self.version, "base_name": self.base_name, "created": self.created,
            "family": self.family, "path": str(self.path), **{k: v for k, v in self.info.items() if k not in ("id", "version", "base_name")},
        }


# --------------------------------------------------------------------------- names

def _short(p: Path) -> str:
    from .config import display_path  # local import: registry stays importable without torch-heavy config

    return display_path(p)


def split_name(name: str) -> tuple[str, int | None]:
    """'ledgar-qwen3-0.6b-lora-v3' -> ('ledgar-qwen3-0.6b-lora', 3); 'sanity' -> ('sanity', None)."""
    m = VERSION_RE.match(name)
    return (m.group("base"), int(m.group("version"))) if m else (name, None)


def versioned_name(base: str, version: int) -> str:
    return f"{base}-v{version}"


def best_link_path(runs_dir: Path) -> Path:
    return Path(runs_dir).parent / "best"


# --------------------------------------------------------------------------- scanning

def is_complete(run_dir: Path) -> bool:
    return (Path(run_dir) / "merged" / "config.json").exists()


def has_content(run_dir: Path) -> bool:
    return Path(run_dir).is_dir() and any(Path(run_dir).iterdir())


def read_model_info(run_dir: Path) -> ModelInfo:
    run_dir = Path(run_dir)
    merged = run_dir / "merged"
    info: dict[str, Any] = {}
    info_path = merged / MODEL_INFO_FILE
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
        except json.JSONDecodeError:
            info = {}
    model_id = info.get("id") or run_dir.name
    base, version = split_name(model_id)
    if "version" in info:
        version = info["version"]
    created = info.get("created") or (merged / "config.json").stat().st_mtime
    return ModelInfo(id=model_id, run_dir=run_dir, base_name=info.get("base_name", base), version=version,
                     created=float(created), info=info, family=info.get("family", "transformer"))


def list_models(runs_dir: str | Path) -> list[ModelInfo]:
    """All completed models under runs_dir, oldest first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    models = []
    for d in runs_dir.iterdir():
        if d.is_dir() and not d.name.startswith((".", "_")) and is_complete(d):
            models.append(read_model_info(d))
    return sorted(models, key=lambda m: (m.created, m.version or 0, m.id))


def latest_model(models: list[ModelInfo], base: str | None = None, family: str | None = "transformer",
                 strict: bool = False) -> ModelInfo | None:
    """Newest model, preferring versioned runs. `family` is a preference (falls back to any family when none of
    that family exists) unless `strict`; `family=None` ignores families; a `base` name ignores families."""
    pool = [m for m in models if base is None or m.base_name == base]
    if family and base is None:
        same = [m for m in pool if m.family == family]
        pool = same if (same or strict) else pool
    versioned = [m for m in pool if m.versioned]
    pool = versioned or pool
    return max(pool, key=lambda m: (m.created, m.version or 0)) if pool else None


def resolve_model(models: list[ModelInfo], requested: str | None,
                  aliases: tuple[str, ...] | list[str] = DEFAULT_ALIASES) -> ModelInfo:
    """None / '' / an alias -> latest; exact id -> that model; base name -> latest version of that base."""
    if not models:
        raise ModelNotFoundError("No trained models found. Train one with `make all`.")
    if requested == CLASSICAL_ALIAS:
        found = latest_model(models, family="classical", strict=True)
        if found is None:
            raise ModelNotFoundError("No classical model is trained. Train one with `make classical`.")
        return found
    if not requested or requested in aliases:
        return latest_model(models)  # type: ignore[return-value]
    for m in models:
        if m.id == requested:
            return m
    by_base = latest_model(models, base=requested)
    if by_base is not None and any(m.base_name == requested for m in models):
        return by_base
    raise ModelNotFoundError(
        f"The model '{requested}' does not exist. Available: {', '.join(m.id for m in models)}"
        f" (or use {', '.join(repr(a) for a in aliases)} for the latest, {CLASSICAL_ALIAS!r} for the latest classical model)."
    )


def resolve_model_dir(runs_dir: str | Path, requested: str | None = "latest") -> Path:
    """CLI helper: a model id / base name / alias, or a path to a model directory -> merged dir."""
    if requested and Path(requested).exists():
        p = Path(requested)
        return p / "merged" if (p / "merged" / "config.json").exists() else p
    return resolve_model(list_models(runs_dir), requested).path


# --------------------------------------------------------------------------- model_info.json

def write_model_info(merged_dir: str | Path, **info: Any) -> dict[str, Any]:
    merged_dir = Path(merged_dir)
    now = time.time()
    info.setdefault("created", now)
    info.setdefault("created_at", datetime.fromtimestamp(info["created"], timezone.utc).isoformat(timespec="seconds"))
    (merged_dir / MODEL_INFO_FILE).write_text(json.dumps(info, indent=2, default=str))
    (merged_dir / MODEL_INFO_FILE).chmod(0o644)
    return info


def update_model_info(merged_dir: str | Path, **fields: Any) -> dict[str, Any]:
    """Merge fields into model_info.json, creating it (from the directory name) if missing."""
    merged_dir = Path(merged_dir).resolve()
    path = merged_dir / MODEL_INFO_FILE
    if path.exists():
        info = json.loads(path.read_text())
    else:
        run_name = merged_dir.parent.name if merged_dir.name == "merged" else merged_dir.name
        base, version = split_name(run_name)
        info = {"id": run_name, "base_name": base, "version": version,
                "created": (merged_dir / "config.json").stat().st_mtime if (merged_dir / "config.json").exists() else time.time()}
    info.update(fields)
    return write_model_info(merged_dir, **info)


def refresh_best_link(runs_dir: str | Path) -> Path | None:
    """Point <runs_dir>/../best at the latest model (relative symlink, works inside Docker mounts)."""
    runs_dir = Path(runs_dir)
    latest = latest_model(list_models(runs_dir))
    if latest is None:
        return None
    link = best_link_path(runs_dir)
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    link.symlink_to(latest.path.resolve().relative_to(link.parent.resolve()), target_is_directory=True)
    return link


# --------------------------------------------------------------------------- config version

def read_config_version(config_path: str | Path) -> int:
    data = yaml.safe_load(Path(config_path).read_text()) or {}
    return max(1, int(data.get("model_version", 1)))


def write_config_version(config_path: str | Path, version: int) -> None:
    """Rewrite only the `model_version:` line so comments and layout are preserved."""
    path = Path(config_path)
    text = path.read_text()
    if CONFIG_VERSION_RE.search(text):
        text = CONFIG_VERSION_RE.sub(lambda m: f"{m.group('key')}{version}", text, count=1)
    else:
        line = f"model_version: {version}          # next model version to train (bumped after `make all`)\n"
        m = re.search(r"^model_name:.*\n", text, re.MULTILINE)
        text = text[: m.end()] + line + text[m.end():] if m else line + text
    path.write_text(text)


def completed_versions(runs_dir: str | Path, base: str) -> list[int]:
    return sorted(m.version for m in list_models(runs_dir) if m.base_name == base and m.version is not None)


def plan_training_run(runs_dir: str | Path, base_name: str, version: int, resume: bool = False,
                      restart: bool = False) -> tuple[str, int, list[str]]:
    """Decide which run directory a training run uses. Never returns a completed run.

    - fresh: `<base>-v<version>`; if that version is already complete, move to the next free one.
    - unfinished directory: error, unless `resume` (continue it) or `restart` (delete it, start over).
    """
    runs_dir = Path(runs_dir)
    base, suffix = split_name(base_name)
    notes: list[str] = []
    if suffix is not None:
        notes.append(f"train.run_name '{base_name}' already ends in -v{suffix}; using base '{base}'.")
    v = max(1, int(version))
    while True:
        name = versioned_name(base, v)
        d = runs_dir / name
        if is_complete(d):
            if resume:
                raise RegistryError(f"{name} is already complete; nothing to resume. Run `make all` to train the next version.")
            notes.append(f"{name} already exists and is complete; using v{v + 1} instead (existing models are never overwritten).")
            v += 1
            continue
        if has_content(d):
            if resume:
                if not (d / "checkpoint-last" / "trainer_state.json").exists():
                    raise RegistryError(
                        f"{name} has no checkpoint to resume from (it stopped before its first evaluation).\n"
                        f"  Start it again from scratch:  make restart"
                    )
                return name, v, notes
            if restart:
                shutil.rmtree(d)
                notes.append(f"Discarded the unfinished run {name} and starting it again.")
                return name, v, notes
            raise UnfinishedRunError(
                f"{name} exists but did not finish ({_short(d)}).\n"
                f"  Continue it:          make resume\n"
                f"  Discard and restart:  make restart"
            )
        if resume:
            raise RegistryError(f"No unfinished run {name} to resume. Start a new one with `make all`.")
        return name, v, notes


def bump_config_version(config_path: str | Path, runs_dir: str | Path, base_name: str) -> tuple[int, int]:
    """After a successful run: model_version = max(current, highest completed version + 1).

    Does nothing if the current version has not completed yet, so a failed run is retried
    under the same version rather than skipped.
    """
    base, _ = split_name(base_name)
    current = read_config_version(config_path)
    done = completed_versions(runs_dir, base)
    new = max(current, (done[-1] + 1) if done else current)
    if new != current:
        write_config_version(config_path, new)
    return current, new


def adopt_run(runs_dir: str | Path, run_name: str, base_name: str, version: int) -> Path:
    """Rename an unversioned completed run (e.g. one trained before versioning) to `<base>-v<version>`."""
    runs_dir = Path(runs_dir)
    src = runs_dir / run_name
    if not is_complete(src):
        raise RegistryError(f"{src} is not a completed run (no merged/config.json).")
    base, _ = split_name(base_name)
    dst = runs_dir / versioned_name(base, version)
    if dst.exists():
        raise RegistryError(f"{dst} already exists.")
    src.rename(dst)
    update_model_info(dst / "merged", id=dst.name, base_name=base, version=version, adopted_from=run_name)
    return dst
