"""Classical step 2: calibrate on validation and report on the full test split (same report as the transformer).

    uv run python scripts/classical/02_evaluate.py                         # latest classical model
    uv run python scripts/classical/02_evaluate.py --model ledgar-tfidf-lr-v1

Writes into the model's merged/: calibration.json, eval.json, and model_info.json["test"] (shown by `make models`
and /v1/models); into telemetry/: test_predictions.npz (calibrated probabilities, for the comparison page).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from classification.classical.backend import SklearnBackend, softmax  # noqa: E402
from classification.classical.config import load_classical_config  # noqa: E402
from classification.config import display_path, resolve_path  # noqa: E402
from classification.data import LabelSpace, load_split  # noqa: E402
from classification.evaluate import expected_calibration_error, fit_temperature, full_report  # noqa: E402
from classification.registry import CLASSICAL_ALIAS, RegistryError, list_models, resolve_model, update_model_info  # noqa: E402

app = typer.Typer(add_completion=False)
console = Console()


def cpu_latency_ms(backend: SklearnBackend, texts: list[str], n: int = 300) -> float:
    sample = texts[:n]
    backend.predict_proba(sample[:5])  # warm up
    t = time.perf_counter()
    for text in sample:
        backend.predict_proba([text])
    return (time.perf_counter() - t) * 1000 / len(sample)


@app.command()
def main(
    model: str = typer.Option(CLASSICAL_ALIAS, help="Classical model id, base name, or 'latest-classical'"),
    config: Path = typer.Option(Path("configs/classical/ledgar_tfidf_lr.yaml")),
) -> None:
    cfg = load_classical_config(config)
    runs = resolve_path(cfg.output_dir)
    try:
        info = resolve_model(list_models(runs), model)
    except RegistryError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2)
    model_dir = info.path
    console.print(f"[bold]Evaluating[/] {info.id} ({display_path(model_dir)})")
    backend = SklearnBackend(model_dir)
    space = LabelSpace.load(model_dir / "label_space.json")

    def split(name: str):
        texts, raw = load_split(cfg.data, name, raw=True)
        return texts, np.asarray(space.map(raw))

    val_texts, y_val = split("validation")
    test_texts, y_test = split("test")
    v_logits = backend.logits(val_texts)
    temperature = fit_temperature(v_logits, y_val)
    (model_dir / "calibration.json").write_text(json.dumps({
        "temperature": temperature, "fitted_on": "validation", "n": int(len(y_val)),
        "ece_before": expected_calibration_error(softmax(v_logits), y_val),
        "ece_after": expected_calibration_error(softmax(v_logits / temperature), y_val),
    }, indent=2))
    t_logits = backend.logits(test_texts)
    report = full_report(t_logits, y_test, space.names, temperature)
    report["split"] = "test"
    (model_dir / "eval.json").write_text(json.dumps(report, indent=2))
    probs = softmax(t_logits / temperature)
    np.savez_compressed(info.run_dir / "telemetry" / "test_predictions.npz", probs=probs.astype(np.float16),
                        labels=y_test.astype(np.int32), pred=probs.argmax(1).astype(np.int32))
    latency = cpu_latency_ms(backend, test_texts)
    update_model_info(model_dir, temperature=temperature, latency_ms_per_clause_cpu=round(latency, 2), test={
        "split": "test", "n": report["n"], "accuracy": report["accuracy"], "top3_accuracy": report["top3_accuracy"],
        "micro_f1": report["micro_f1"], "macro_f1": report["macro_f1"], "ece": report["ece"],
    })

    t = Table(title=f"{info.id}: test metrics ({report['n']} examples)")
    for k in ("accuracy", "top3_accuracy", "macro_f1", "weighted_f1", "nll", "ece_uncalibrated", "ece", "temperature"):
        t.add_row(k, f"{report[k]:.4f}")
    t.add_row("CPU latency per clause", f"{latency:.2f} ms")
    console.print(t)
    console.print(f"Full report: {display_path(model_dir / 'eval.json')}")


if __name__ == "__main__":
    app()
