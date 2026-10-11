"""Model versions: list trained models, show the next version, bump the version, adopt old runs.

    uv run python scripts/07_models.py list                 # all trained models (latest marked)
    uv run python scripts/07_models.py next                 # name and version the next training run will use
    uv run python scripts/07_models.py latest [--path]      # latest model id (or its directory)
    uv run python scripts/07_models.py bump                 # after a completed run: model_version += 1 (run by `make all`)
    uv run python scripts/07_models.py adopt <run-name>     # rename an unversioned run to <run_name>-v<model_version>
    uv run python scripts/07_models.py best                 # re-point artifacts/best at the latest model
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from classification.config import display_path, load_config, resolve_path
from classification.registry import (
    RegistryError,
    adopt_run,
    bump_config_version,
    latest_model,
    list_models,
    plan_training_run,
    refresh_best_link,
    split_name,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
CONFIG = typer.Option(Path("configs/ledgar_qwen3_0.6b.yaml"), help="YAML config")


class _View:
    """model_version / run_name / runs dir for either config type (transformer or classical)."""

    def __init__(self, config: Path):
        import yaml

        raw = yaml.safe_load(Path(config).read_text()) or {}
        if "features" in raw or "logistic_regression" in raw:  # classical config
            from classification.classical.config import load_classical_config

            c = load_classical_config(config)
            self.model_version, self.run_name, self.family = c.model_version, c.run_name, "classical"
            self.runs = resolve_path(c.output_dir)
        else:
            c = load_config(config)
            self.model_version, self.run_name, self.family = c.model_version, c.train.run_name, "transformer"
            self.runs = resolve_path(c.train.output_dir)


def _runs(config: Path):
    v = _View(config)
    return v, v.runs


CLASSICAL_RUNS = Path("artifacts/classical/runs")


@app.command("list")
def list_cmd(config: Path = CONFIG) -> None:
    """All trained models, oldest first; the one the API uses by default is marked latest."""
    cfg, runs = _runs(config)
    dirs = [runs] + ([resolve_path(CLASSICAL_RUNS)] if resolve_path(CLASSICAL_RUNS) != runs else [])
    models = sorted([m for d in dirs for m in list_models(d)], key=lambda m: (m.family != "transformer", m.created))
    if not models:
        console.print(f"No trained models in {display_path(runs)}. Train one with `make all`.")
        return
    latest_ids = {x.id for x in (latest_model(models, family="transformer", strict=True),
                                 latest_model(models, family="classical", strict=True)) if x}
    t = Table(title="Trained models (" + ", ".join(display_path(d) for d in dirs) + ")")
    for col in ("model", "family", "version", "created", "val acc", "val macro-F1", "test acc", "test top-3", "test macro-F1", ""):
        t.add_column(col)
    for m in models:
        test = m.info.get("test", {})
        t.add_row(
            m.id, m.family, str(m.version) if m.version is not None else "-",
            datetime.fromtimestamp(m.created).strftime("%Y-%m-%d %H:%M"),
            f"{m.info['best_val_accuracy']:.4f}" if m.info.get("best_val_accuracy") is not None else "-",
            f"{m.info['best_val_macro_f1']:.4f}" if "best_val_macro_f1" in m.info else "-",
            f"{test['accuracy']:.4f}" if "accuracy" in test else "-",
            f"{test['top3_accuracy']:.4f}" if "top3_accuracy" in test else "-",
            f"{test['macro_f1']:.4f}" if "macro_f1" in test else "-",
            "[green]latest[/]" if m.id in latest_ids else "",
        )
    console.print(t)
    console.print(f"Next training run: model_version={cfg.model_version} in {display_path(config)}")


@app.command("next")
def next_cmd(config: Path = CONFIG) -> None:
    """The run name and version `make all` / `make train` will use next."""
    cfg, runs = _runs(config)
    try:
        name, version, notes = plan_training_run(runs, cfg.run_name, cfg.model_version)
    except RegistryError as exc:
        console.print(f"[yellow]{exc}[/]")
        raise typer.Exit(code=1)
    for n in notes:
        console.print(f"[yellow]{n}[/]")
    console.print(f"{name}  (version {version})")


@app.command("latest")
def latest_cmd(config: Path = CONFIG, path: bool = typer.Option(False, help="Print the model directory instead")) -> None:
    """Latest model id (used by the API when no model is requested)."""
    _, runs = _runs(config)
    m = latest_model(list_models(runs))
    if m is None:
        console.print("No trained models yet.")
        raise typer.Exit(code=1)
    print(m.path if path else m.id)


@app.command("bump")
def bump_cmd(config: Path = CONFIG) -> None:
    """Raise model_version after the current version finished training (no-op if it did not)."""
    cfg, runs = _runs(config)
    base, _ = split_name(cfg.run_name)
    old, new = bump_config_version(config, runs, base)
    refresh_best_link(runs)
    if new == old:
        console.print(f"[yellow]model_version stays {old}[/]: {base}-v{old} has not completed yet.")
    else:
        console.print(f"[green]model_version {old} -> {new}[/] in {display_path(config)}. Next training run: {base}-v{new}")


@app.command("adopt")
def adopt_cmd(run_name: str, config: Path = CONFIG,
              version: int = typer.Option(None, help="Version to assign (default: model_version from the config)")) -> None:
    """Turn an unversioned completed run (e.g. trained before versioning) into <run_name>-v<N>, then bump."""
    cfg, runs = _runs(config)
    base, _ = split_name(cfg.run_name)
    v = version or cfg.model_version
    try:
        dst = adopt_run(runs, run_name, base, v)
    except RegistryError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2)
    old, new = bump_config_version(config, runs, base)
    refresh_best_link(runs)
    console.print(f"[green]{run_name} -> {dst.name}[/]; model_version {old} -> {new}")


@app.command("best")
def best_cmd(config: Path = CONFIG) -> None:
    """Re-point artifacts/best at the latest model."""
    _, runs = _runs(config)
    link = refresh_best_link(runs)
    console.print(f"{link} -> {link.readlink()}" if link else "No trained models yet.")


if __name__ == "__main__":
    app()
