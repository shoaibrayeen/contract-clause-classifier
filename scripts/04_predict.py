"""Step 4: classify a contract (file) or a single clause (--text) and print JSON.

    uv run python scripts/04_predict.py path/to/contract.pdf
    uv run python scripts/04_predict.py --text "This Agreement shall be governed by the laws of New York."
    uv run python scripts/04_predict.py contract.docx --top-k 5 --threshold 0.2 --out result.json
    uv run python scripts/04_predict.py contract.pdf --model ledgar-qwen3-0.6b-lora-v1    # a specific version

By default the latest trained model is used (see `make models`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from classification.config import load_config, resolve_path
from classification.io_utils import read_file
from classification.predict import load_predictor
from classification.registry import RegistryError, resolve_model_dir

app = typer.Typer(add_completion=False)


@app.command()
def main(
    path: Optional[Path] = typer.Argument(None, help=".txt / .pdf / .docx contract"),
    text: Optional[str] = typer.Option(None, help="Classify this text instead of a file"),
    model: str = typer.Option("latest", help="Model id (e.g. ledgar-qwen3-0.6b-lora-v2), base name, or 'latest'"),
    model_dir: Optional[Path] = typer.Option(None, help="Explicit model directory (overrides --model)"),
    config: Path = typer.Option(Path("configs/ledgar_qwen3_0.6b.yaml"), help="Config (for the runs directory)"),
    device: str = typer.Option("auto"),
    top_k: int = typer.Option(3),
    threshold: float = typer.Option(0.30),
    no_segment: bool = typer.Option(False, help="Treat the whole input as one provision"),
    out: Optional[Path] = typer.Option(None, help="Write JSON here as well"),
) -> None:
    logging.basicConfig(level=logging.WARNING)
    if (path is None) == (text is None):
        raise typer.BadParameter("Provide exactly one of PATH or --text")
    content = text if text is not None else read_file(path)
    if model_dir is None:
        try:
            model_dir = resolve_model_dir(resolve_path(load_config(config).train.output_dir), model)
        except RegistryError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2)
    predictor = load_predictor(resolve_path(model_dir), device=device, top_k=top_k, threshold=threshold)
    typer.echo(f"model: {predictor.backend.model_id}", err=True)
    # A single short clause should not be segmented.
    segment = not no_segment and (text is None or len(content.split()) > 350)
    result = predictor.predict_document(content, segment=segment)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if out:
        out.write_text(payload)


if __name__ == "__main__":
    app()
