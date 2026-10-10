"""Step 4: classify a contract (file) or a single clause (--text) and print JSON.

    uv run python scripts/04_predict.py path/to/contract.pdf
    uv run python scripts/04_predict.py --text "This Agreement shall be governed by the laws of New York."
    uv run python scripts/04_predict.py contract.docx --top-k 5 --threshold 0.2 --out result.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from classification.config import resolve_path
from classification.io_utils import read_file
from classification.predict import load_predictor

app = typer.Typer(add_completion=False)


@app.command()
def main(
    path: Optional[Path] = typer.Argument(None, help=".txt / .pdf / .docx contract"),
    text: Optional[str] = typer.Option(None, help="Classify this text instead of a file"),
    model_dir: Path = typer.Option(Path("artifacts/best")),
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
    predictor = load_predictor(resolve_path(model_dir), device=device, top_k=top_k, threshold=threshold)
    # A single short clause should not be segmented.
    segment = not no_segment and (text is None or len(content.split()) > 350)
    result = predictor.predict_document(content, segment=segment)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if out:
        out.write_text(payload)


if __name__ == "__main__":
    app()
