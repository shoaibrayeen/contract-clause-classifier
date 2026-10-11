"""Run the latest transformer and the latest classical model on the same text and show them side by side.

    uv run python scripts/11_compare.py demo/sample_contract.txt
    uv run python scripts/11_compare.py --text "This Agreement shall be governed by the laws of New York."
    uv run python scripts/11_compare.py contract.pdf --transformer ledgar-qwen3-0.6b-v1 --classical ledgar-tfidf-lr-v1
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classification.api.app import _load_any  # noqa: E402
from classification.api.settings import Settings  # noqa: E402
from classification.config import PROJECT_ROOT  # noqa: E402
from classification.io_utils import read_file  # noqa: E402
from classification.registry import CLASSICAL_ALIAS, RegistryError, list_models, resolve_model  # noqa: E402

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    path: Optional[Path] = typer.Argument(None, help=".txt / .pdf / .docx contract"),
    text: Optional[str] = typer.Option(None, help="Classify this text instead of a file"),
    transformer: str = typer.Option("latest", help="Transformer model id or 'latest'"),
    classical: str = typer.Option(CLASSICAL_ALIAS, help="Classical model id or 'latest-classical'"),
    device: str = typer.Option("auto", help="Device for the transformer"),
) -> None:
    if (path is None) == (text is None):
        raise typer.BadParameter("Provide exactly one of PATH or --text")
    content = text if text is not None else read_file(path)
    models = list_models(PROJECT_ROOT / "artifacts" / "runs") + list_models(PROJECT_ROOT / "artifacts" / "classical" / "runs")
    try:
        infos = [resolve_model(models, transformer), resolve_model(models, classical)]
    except RegistryError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2)
    settings = Settings(device=device, top_k=1)
    predictors = [_load_any(i.path, settings) for i in infos]
    provisions = predictors[0].split(content, segment=True)
    results, timings = [], []
    for p in predictors:
        t = time.perf_counter()
        results.append(p.predict_provisions_document(provisions, top_k=1))
        timings.append((time.perf_counter() - t) * 1000)

    table = Table(title=f"{len(provisions)} provisions: {infos[0].id} vs {infos[1].id}")
    table.add_column("#", justify="right")
    table.add_column("Provision")
    table.add_column(f"Transformer ({timings[0]:.0f} ms)")
    table.add_column(f"Classical ({timings[1]:.0f} ms)")
    table.add_column("")
    agree = 0
    for i, txt in enumerate(provisions):
        a, b = results[0]["clauses"][i]["predictions"][0], results[1]["clauses"][i]["predictions"][0]
        same = a["name"] == b["name"]
        agree += same
        table.add_row(str(i + 1), txt[:60] + ("…" if len(txt) > 60 else ""), f"{a['name']} {a['confidence']:.0%}",
                      f"{b['name']} {b['confidence']:.0%}", "[green]agree[/]" if same else "[yellow]differ[/]")
    console.print(table)
    console.print(f"Agreement: {agree}/{len(provisions)}")


if __name__ == "__main__":
    app()
