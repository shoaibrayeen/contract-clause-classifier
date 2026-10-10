"""Step 1: download LEDGAR, save label map, tokenise and cache all splits."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from classification.config import display_path, load_config
from classification.data import SPLITS, download_ledgar, get_label_names, length_stats, load_tokenized

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    config: Path = typer.Option(Path("configs/ledgar_qwen3_0.6b.yaml"), help="YAML config"),
    force: bool = typer.Option(False, help="Re-download even if files exist"),
    skip_tokenize: bool = typer.Option(False, help="Only download raw parquet"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(config)
    paths = download_ledgar(cfg.data, force=force)
    names = get_label_names(cfg.data)
    console.print(f"[green]Downloaded[/] {', '.join(f'{k}={display_path(v)}' for k, v in paths.items())}")
    console.print(f"[green]{len(names)} labels[/], e.g. {names[:5]} ...")
    if skip_tokenize:
        return

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    table = Table(title=f"Token lengths ({cfg.model_name}, max_len={cfg.data.max_len})")
    for col in ("split", "n", "mean", "p50", "p90", "p99", "max", "truncated_frac"):
        table.add_column(col)
    for split in SPLITS:
        ids, _ = load_tokenized(cfg.data, split, tok, cfg.model_name)
        s = length_stats(ids, cfg.data.max_len)
        table.add_row(split, *(f"{s[k]:.3f}" if isinstance(s[k], float) else str(s[k]) for k in list(s)))
        console.print_json(json.dumps({"split": split, **s}))
    console.print(table)


if __name__ == "__main__":
    app()
