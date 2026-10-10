"""Step 5: serve the model with FastAPI (OpenAI-compatible) natively, using MPS/CUDA when available.

    uv run python scripts/05_serve.py                     # http://localhost:8000/docs
    MODEL_DIR=artifacts/runs/<run>/merged uv run python scripts/05_serve.py --port 9000
"""

from __future__ import annotations

import logging

import typer
import uvicorn

app = typer.Typer(add_completion=False)


@app.command()
def main(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev only)"),
    log_level: str = typer.Option("info"),
) -> None:
    logging.basicConfig(level=log_level.upper(), format="%(levelname)s %(name)s %(message)s")
    uvicorn.run("classification.api.app:create_app", factory=True, host=host, port=port, reload=reload,
                log_level=log_level)


if __name__ == "__main__":
    app()
