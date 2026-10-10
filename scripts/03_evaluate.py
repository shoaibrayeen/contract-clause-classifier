"""Step 3: evaluate a trained model, fit temperature calibration on validation, report test metrics.

    uv run python scripts/03_evaluate.py                         # evaluates artifacts/best
    uv run python scripts/03_evaluate.py --model-dir artifacts/runs/<run>/merged --limit 1000

Writes:
    <model-dir>/calibration.json     temperature used by the API for calibrated confidences
    <model-dir>/eval.json            test metrics, per-class report, top confusions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from classification.config import load_config, resolve_device, resolve_path
from classification.evaluate import fit_temperature, full_report, softmax, expected_calibration_error, split_logits
from classification.labels import load_labels
from classification.model import LABELS_FILE, load_trained, save_calibration

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    model_dir: Path = typer.Option(Path("artifacts/best"), help="Merged model or training checkpoint dir"),
    config: Path = typer.Option(Path("configs/ledgar_qwen3_0.6b.yaml")),
    split: str = typer.Option("test", help="Split to report on"),
    limit: Optional[int] = typer.Option(None, help="Only first N examples of each split (quick check)"),
    device: str = typer.Option("auto"),
    batch_size: int = typer.Option(32),
    calibrate: bool = typer.Option(True, help="Fit temperature on validation and save calibration.json"),
) -> None:
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(config)
    model_dir = resolve_path(model_dir).resolve()
    dev = resolve_device(device)
    model, tok = load_trained(model_dir, dev)
    names = load_labels(model_dir / LABELS_FILE) if (model_dir / LABELS_FILE).exists() else [
        model.config.id2label[i] for i in range(model.config.num_labels)
    ]
    # Tokenisation cache is keyed by the base tokenizer name (same vocab as the fine-tuned model).
    tok_name = cfg.model_name

    temperature = 1.0
    if calibrate:
        v_logits, v_labels = split_logits(model, tok, cfg.data, "validation", tok_name, dev, batch_size=batch_size, limit=limit)
        temperature = fit_temperature(v_logits, v_labels)
        save_calibration(
            model_dir, temperature,
            fitted_on="validation", n=int(len(v_labels)),
            ece_before=expected_calibration_error(softmax(v_logits), v_labels),
            ece_after=expected_calibration_error(softmax(v_logits, temperature), v_labels),
        )
        console.print(f"[green]Temperature[/] T={temperature:.4f} saved to {model_dir / 'calibration.json'}")

    logits, labels = split_logits(model, tok, cfg.data, split, tok_name, dev, batch_size=batch_size, limit=limit)
    report = full_report(logits, labels, names, temperature)
    report["split"] = split
    (model_dir / "eval.json").write_text(json.dumps(report, indent=2))

    t = Table(title=f"{split} metrics ({report['n']} examples)")
    for k in ("accuracy", "micro_f1", "macro_f1", "weighted_f1", "nll", "ece_uncalibrated", "ece", "temperature"):
        t.add_row(k, f"{report[k]:.4f}")
    console.print(t)
    c = Table(title="Top confusions (true -> predicted)")
    for col in ("true", "pred", "count"):
        c.add_column(col)
    for row in report["top_confusions"][:10]:
        c.add_row(row["true"], row["pred"], str(row["count"]))
    console.print(c)
    console.print(f"Full report: {model_dir / 'eval.json'}")


if __name__ == "__main__":
    app()
