"""Classical step 1: train the next classical model version (TF-IDF + logistic regression; SVM and Naive Bayes
reported for comparison). CPU only, about half an hour.

    uv run python scripts/classical/01_train.py                    # -> artifacts/classical/runs/ledgar-tfidf-lr-v<N>
    uv run python scripts/classical/01_train.py --restart          # discard an unfinished version and retrain it
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from classification.classical.config import load_classical_config  # noqa: E402
from classification.classical.train import train  # noqa: E402
from classification.registry import RegistryError  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/classical/ledgar_tfidf_lr.yaml")),
    restart: bool = typer.Option(False, help="Delete an unfinished current version and train it again"),
) -> None:
    try:
        train(load_classical_config(config), restart=restart)
    except RegistryError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
