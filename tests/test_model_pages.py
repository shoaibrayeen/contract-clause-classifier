"""Model pages generator on synthetic run folders (no trained model, GPU or network)."""

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module(out_dir):
    spec = importlib.util.spec_from_file_location("model_pages", ROOT / "scripts" / "10_model_pages.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_pages"] = mod
    spec.loader.exec_module(mod)
    mod.OUT = out_dir
    return mod


def _report(acc):
    return {"n": 100, "accuracy": acc, "top3_accuracy": acc + 0.04, "micro_f1": acc, "macro_f1": acc - 0.03,
            "weighted_f1": acc, "nll": 0.4, "ece": 0.02, "ece_uncalibrated": 0.07, "temperature": 0.9,
            "per_class": {"Governing Laws": {"f1-score": 0.95, "support": 50}, "Notices": {"f1-score": 0.9, "support": 50}},
            "top_confusions": [{"true": "Notices", "pred": "Governing Laws", "count": 3}]}


def _transformer_run(root):
    run = root / "ledgar-qwen3-0.6b-v1"
    (run / "merged").mkdir(parents=True)
    (run / "telemetry").mkdir()
    (run / "merged" / "config.json").write_text("{}")
    (run / "merged" / "eval.json").write_text(json.dumps(_report(0.92)))
    (run / "merged" / "calibration.json").write_text(json.dumps({"temperature": 0.8}))
    (run / "merged" / "model_info.json").write_text(json.dumps({"id": run.name, "version": 1, "family": "transformer",
        "base_model": "no-such/model", "method": "full", "num_labels": 2, "best_val_accuracy": 0.92, "created": time.time()}))
    (run / "config.yaml").write_text("model_name: no-such/model\ntrain: {lr: 2.0e-5, epochs: 1, batch_size: 4, grad_accum: 8}\n")
    (run / "telemetry" / "meta.json").write_text(json.dumps({"model": {"steps_per_epoch": 10, "total_params": 596e6,
        "trainable_params": 596e6}, "environment": {"device": "mps", "memory_gb": 48}}))
    t0 = 1000.0
    (run / "telemetry" / "steps.jsonl").write_text("\n".join(json.dumps({"step": s, "loss": 3 / s, "lr": 1e-5, "tokens": 900,
        "mem_gb": 25}) for s in range(1, 21)))
    (run / "telemetry" / "evals.jsonl").write_text("\n".join(json.dumps({"step": s, "epoch": s / 10, "accuracy": 0.8 + s / 200,
        "top3_accuracy": 0.95, "macro_f1": 0.75, "loss": 0.5}) for s in (10, 20)))
    (run / "telemetry" / "events.jsonl").write_text(json.dumps({"event": "start", "time": t0}) + "\n"
                                                     + json.dumps({"event": "done", "time": t0 + 3600}))
    return run


def _classical_run(root):
    run = root / "ledgar-tfidf-lr-v1"
    (run / "merged").mkdir(parents=True)
    (run / "telemetry").mkdir()
    (run / "merged" / "config.json").write_text("{}")
    (run / "merged" / "eval.json").write_text(json.dumps(_report(0.91)))
    (run / "merged" / "calibration.json").write_text(json.dumps({"temperature": 1.1}))
    (run / "merged" / "top_features.json").write_text(json.dumps({"Governing Laws": [{"term": "governed by", "weight": 3.1}]}))
    (run / "merged" / "model_info.json").write_text(json.dumps({"id": run.name, "version": 1, "family": "classical",
        "classifier": {"C": 10}, "features": {"word": {"n_features": 100}, "char": {"n_features": 200}, "total": 300},
        "train_seconds": 1500, "model_size_mb": 90, "latency_ms_per_clause_cpu": 1.2, "best_val_accuracy": 0.91, "created": time.time()}))
    val = {"accuracy": 0.91, "top3_accuracy": 0.97, "macro_f1": 0.88, "log_loss": 0.34}
    (run / "telemetry" / "candidates.json").write_text(json.dumps([
        {"family": "logistic_regression", "params": {"C": 10}, "fit_seconds": 280, "validation": val, "train": {**val, "log_loss": 0.1}, "selected": True},
        {"family": "linear_svm", "params": {"C": 0.5}, "fit_seconds": 46, "validation": {**val, "accuracy": 0.915}, "train": val}]))
    (run / "telemetry" / "learning_curve.json").write_text(json.dumps([
        {"fraction": f, "train_examples": int(f * 60000), "validation": {**val, "accuracy": 0.8 + f / 10}} for f in (0.1, 0.5, 1.0)]))
    return run


def test_transformer_and_classical_pages(tmp_path):
    from classification.registry import list_models

    mod = _module(tmp_path / "docs" / "models")
    t_run, c_run = _transformer_run(tmp_path / "runs"), _classical_run(tmp_path / "classical")
    t_info, c_info = list_models(tmp_path / "runs")[0], list_models(tmp_path / "classical")[0]

    html_t, data_t = mod.transformer_page(t_info, [t_info], "now")
    assert "Transformer model: ledgar-qwen3-0.6b-v1" in html_t and "92.00%" in html_t
    assert "transformer_loss.png" in html_t and data_t["training_seconds"] == 3600
    html_c, data_c = mod.classical_page(c_info, [c_info], "now")
    assert "Classical model: ledgar-tfidf-lr-v1" in html_c and "governed by" in html_c and "learning curve" in html_c.lower()
    assert data_c["served"] == {"C": 10}
    for png in ("transformer_loss.png", "transformer_validation.png", "transformer_per_class_f1.png",
                "classical_candidates.png", "classical_learning_curve.png", "classical_per_class_f1.png"):
        assert (tmp_path / "docs" / "models" / "assets" / png).stat().st_size > 1000, png
    for page in (html_t, html_c):  # every page links to the others
        assert 'href="compare.html"' in page and 'href="transformer.html"' in page and 'href="classical.html"' in page
