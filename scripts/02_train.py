"""Step 2: fine-tune the classifier. Runs on CUDA, MPS or CPU (auto-detected).

Examples:
    uv run python scripts/02_train.py                                   # full run from the YAML
    uv run python scripts/02_train.py --limit 512 --epochs 3 --run-name sanity --eval-steps 20
    uv run python scripts/02_train.py --resume                          # continue checkpoint-last
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from classification.config import load_config
from classification.train import Trainer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/ledgar_qwen3_0.6b.yaml")),
    run_name: Optional[str] = typer.Option(None),
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
    resume: bool = typer.Option(False, help="Resume from <run>/checkpoint-last"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    train_over = {k: v for k, v in dict(
        run_name=run_name, limit=limit, epochs=epochs, eval_steps=eval_steps, eval_limit=eval_limit,
        batch_size=batch_size, grad_accum=grad_accum, lr=lr,
    ).items() if v is not None}
    top_over = {k: v for k, v in dict(device=device, precision=precision, model_name=model_name).items() if v is not None}
    cfg = load_config(config, {**top_over, "train": train_over})
    Trainer(cfg, resume=resume).fit()


if __name__ == "__main__":
    app()
