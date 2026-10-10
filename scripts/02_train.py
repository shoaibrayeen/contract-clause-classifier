"""Step 2: fine-tune the classifier. Runs on CUDA, MPS or CPU (auto-detected).

Every run produces a new model version and never overwrites an existing one:
    artifacts/runs/<train.run_name>-v<model_version>/      (model_version comes from the YAML)

Examples:
    uv run python scripts/02_train.py                         # train the next version, e.g. ...-lora-v1
    uv run python scripts/02_train.py --resume                # continue the unfinished current version
    uv run python scripts/02_train.py --restart               # discard the unfinished current version, start over
    uv run python scripts/02_train.py --run-name sanity --limit 256 --epochs 2   # explicit, unversioned run

After a versioned run completes, `scripts/07_models.py bump` (part of `make all`) raises model_version.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from classification.config import load_config, resolve_path
from classification.registry import RegistryError, plan_training_run
from classification.train import Trainer

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    config: Path = typer.Option(Path("configs/ledgar_qwen3_0.6b.yaml")),
    run_name: Optional[str] = typer.Option(
        None, help="Explicit run name (no automatic version suffix), e.g. 'sanity'. Default: <run_name>-v<model_version>"),
    limit: Optional[int] = typer.Option(None, help="Use only N training examples"),
    epochs: Optional[float] = typer.Option(None),
    eval_steps: Optional[int] = typer.Option(None),
    eval_limit: Optional[int] = typer.Option(None, help="Cap validation examples during training"),
    batch_size: Optional[int] = typer.Option(None),
    grad_accum: Optional[int] = typer.Option(None),
    lr: Optional[float] = typer.Option(None),
    device: Optional[str] = typer.Option(None, help="auto | cuda | mps | cpu"),
    precision: Optional[str] = typer.Option(None, help="auto | bf16 | fp16 | fp32"),
    model_name: Optional[str] = typer.Option(None, help="Override backbone, e.g. Qwen/Qwen2.5-0.5B"),
    resume: bool = typer.Option(False, help="Continue the unfinished run from its checkpoint-last"),
    restart: bool = typer.Option(False, help="Delete the unfinished current version and train it from scratch"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if resume and restart:
        raise typer.BadParameter("Use either --resume or --restart, not both")
    train_over = {k: v for k, v in dict(
        run_name=run_name, limit=limit, epochs=epochs, eval_steps=eval_steps, eval_limit=eval_limit,
        batch_size=batch_size, grad_accum=grad_accum, lr=lr,
    ).items() if v is not None}
    top_over = {k: v for k, v in dict(device=device, precision=precision, model_name=model_name).items() if v is not None}
    cfg = load_config(config, {**top_over, "train": train_over})

    version: int | None = None
    if run_name is None:  # versioned run: <base>-v<N>
        try:
            name, version, notes = plan_training_run(
                resolve_path(cfg.train.output_dir), cfg.train.run_name, cfg.model_version, resume=resume, restart=restart
            )
        except RegistryError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=2)
        for note in notes:
            console.print(f"[yellow]{note}[/]")
        cfg.train.run_name = name
    console.print(
        f"[bold green]Training model[/] [bold]{cfg.train.run_name}[/]"
        + (f" (version {version})" if version is not None else " (unversioned)")
        + f" -> {resolve_path(cfg.train.output_dir) / cfg.train.run_name}"
    )
    Trainer(cfg, resume=resume, version=version).fit()


if __name__ == "__main__":
    app()
