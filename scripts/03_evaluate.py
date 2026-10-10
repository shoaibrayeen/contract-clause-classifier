"""Step 3: evaluate a trained model, fit temperature calibration on validation, report test metrics.

    uv run python scripts/03_evaluate.py                                   # evaluates the latest model
    uv run python scripts/03_evaluate.py --model ledgar-qwen3-0.6b-lora-v1   # a specific version
    uv run python scripts/03_evaluate.py --model-dir artifacts/runs/<run>/merged --limit 1000

Writes:
    <model-dir>/calibration.json     temperature used by the API for calibrated confidences
    <model-dir>/eval.json            test metrics, per-class report, top confusions
    <model-dir>/model_info.json      updated with the headline test metrics (shown by `make models` and /v1/models)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from classification.config import display_path, load_config, resolve_device, resolve_path
from classification.evaluate import fit_temperature, full_report, softmax, expected_calibration_error, split_logits
from classification.labels import load_labels
from classification.model import LABELS_FILE, label_space_for_model, load_trained, save_calibration
from classification.registry import RegistryError, resolve_model_dir, update_model_info

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    model: str = typer.Option("latest", help="Model id (e.g. ledgar-qwen3-0.6b-lora-v2), base name, or 'latest'"),
    model_dir: Optional[Path] = typer.Option(None, help="Explicit model directory (overrides --model)"),
    config: Path = typer.Option(Path("configs/ledgar_qwen3_0.6b.yaml")),
    split: str = typer.Option("test", help="Split to report on"),
    limit: Optional[int] = typer.Option(None, help="Only first N examples of each split (quick check)"),
    device: str = typer.Option("auto"),
    batch_size: int = typer.Option(32),
    calibrate: bool = typer.Option(True, help="Fit temperature on validation and save calibration.json"),
) -> None:
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(config)
    if model_dir is not None:
        model_dir = resolve_path(model_dir).resolve()
    else:
        try:
            model_dir = resolve_model_dir(resolve_path(cfg.train.output_dir), model).resolve()
        except RegistryError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=2)
    console.print(f"[bold]Evaluating[/] {display_path(model_dir)}")
    dev = resolve_device(device)
    clf, tok = load_trained(model_dir, dev)
    names = load_labels(model_dir / LABELS_FILE) if (model_dir / LABELS_FILE).exists() else [
        clf.config.id2label[i] for i in range(clf.config.num_labels)
    ]
    # Tokenisation cache is keyed by the base tokenizer name (same vocab as the fine-tuned model).
    tok_name = cfg.model_name
    # Score against the label space this model was trained with (saved as label_space.json), so older
    # versions stay evaluable after configs/ledgar_label_groups.yaml changes.
    try:
        space = label_space_for_model(model_dir, cfg.data, names)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2)
    console.print(f"Label space: {len(space.names)} clause types" + (f" ({space.source})" if space.source else ""))

    temperature = 1.0
    if calibrate:
        v_logits, v_labels = split_logits(clf, tok, cfg.data, "validation", tok_name, dev, batch_size=batch_size, limit=limit, space=space)
        temperature = fit_temperature(v_logits, v_labels)
        save_calibration(
            model_dir, temperature,
            fitted_on="validation", n=int(len(v_labels)),
            ece_before=expected_calibration_error(softmax(v_logits), v_labels),
            ece_after=expected_calibration_error(softmax(v_logits, temperature), v_labels),
        )
        console.print(f"[green]Temperature[/] T={temperature:.4f} saved to {display_path(model_dir / 'calibration.json')}")

    logits, labels = split_logits(clf, tok, cfg.data, split, tok_name, dev, batch_size=batch_size, limit=limit, space=space)
    report = full_report(logits, labels, names, temperature)
    report["split"] = split
    (model_dir / "eval.json").write_text(json.dumps(report, indent=2))
    if model_dir.name == "merged":  # record headline metrics for `make models` and /v1/models
        try:
            update_model_info(model_dir, temperature=temperature, test={
                "split": split, "n": report["n"], "accuracy": report["accuracy"], "top3_accuracy": report["top3_accuracy"],
                "micro_f1": report["micro_f1"],
                "macro_f1": report["macro_f1"], "ece": report["ece"],
            })
        except OSError as exc:
            console.print(f"[yellow]Could not update model_info.json: {exc}[/]")

    t = Table(title=f"{split} metrics ({report['n']} examples)")
    for k in ("accuracy", "top3_accuracy", "micro_f1", "macro_f1", "weighted_f1", "nll", "ece_uncalibrated", "ece", "temperature"):
        t.add_row(k, f"{report[k]:.4f}")
    console.print(t)
    c = Table(title="Top confusions (true -> predicted)")
    for col in ("true", "pred", "count"):
        c.add_column(col)
    for row in report["top_confusions"][:10]:
        c.add_row(row["true"], row["pred"], str(row["count"]))
    console.print(c)
    console.print(f"Full report: {display_path(model_dir / 'eval.json')}")


if __name__ == "__main__":
    app()
