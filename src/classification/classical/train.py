"""Train the classical clause classifier: TF-IDF features + logistic regression (served), with linear SVM
and Complement Naive Bayes trained on the same features for comparison.

Run directory (never overwritten; versioned like the transformer):

    artifacts/classical/runs/<run_name>-v<N>/
    ├── config.yaml
    ├── merged/      model.joblib, labels.json, label_space.json, config.json, model_info.json, top_features.json
    │                (+ calibration.json, eval.json after scripts/classical/02_evaluate.py)
    └── telemetry/   meta.json, candidates.json, learning_curve.json, events.jsonl
"""

from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from ..config import display_path, resolve_path
from ..data import get_label_space, load_split
from ..registry import plan_training_run, write_model_info
from .config import ClassicalConfig
from .features import build_features, describe

console = Console()


def _metrics(y: np.ndarray, scores: np.ndarray, probs: np.ndarray | None, num_labels: int) -> dict:
    from sklearn.metrics import f1_score, log_loss

    pred = scores.argmax(axis=1)
    top3 = np.argsort(-scores, axis=1)[:, :3]
    out = {
        "accuracy": float((pred == y).mean()),
        "top3_accuracy": float((top3 == y[:, None]).any(axis=1).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(num_labels)), zero_division=0)),
    }
    if probs is not None:
        out["log_loss"] = float(log_loss(y, np.clip(probs, 1e-12, 1), labels=list(range(num_labels))))
    return out


def _event(path: Path, **record) -> None:
    record["time"] = time.time()
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def train(cfg: ClassicalConfig, restart: bool = False) -> Path:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    import joblib

    runs = resolve_path(cfg.output_dir)
    name, version, notes = plan_training_run(runs, cfg.run_name, cfg.model_version, restart=restart)
    for n in notes:
        console.print(f"[yellow]{n}[/]")
    run_dir = runs / name
    merged, tel = run_dir / "merged", run_dir / "telemetry"
    merged.mkdir(parents=True, exist_ok=True)
    tel.mkdir(parents=True, exist_ok=True)
    events = tel / "events.jsonl"
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    console.print(f"[bold green]Training classical model[/] [bold]{name}[/] (version {version}) -> {display_path(run_dir)}")
    t_start = time.time()
    _event(events, event="start", model_id=name, model_version=version)

    space = get_label_space(cfg.data)
    K = len(space.names)
    X_train, y_train = load_split(cfg.data, "train")
    X_val, y_val = load_split(cfg.data, "validation")
    y_train, y_val = np.asarray(y_train), np.asarray(y_val)
    console.print(f"Data: {len(X_train):,} train / {len(X_val):,} validation provisions, {K} clause types")

    # ---- features (fitted once on the training split, shared by every candidate)
    t = time.time()
    features = build_features(cfg.features)
    A = features.fit_transform(X_train)
    B = features.transform(X_val)
    feat_info = describe(features)
    feat_seconds = time.time() - t
    console.print(f"Features: {feat_info['total']:,} TF-IDF features in {feat_seconds:.0f}s")
    _event(events, event="features", seconds=feat_seconds, **{k: v for k, v in feat_info.items() if k == "total"})

    candidates: list[dict] = []

    def record(family: str, params: dict, model, fit_s: float, with_probs: bool = True) -> dict:
        val_scores = model.decision_function(B) if hasattr(model, "decision_function") else np.log(np.clip(model.predict_proba(B), 1e-12, 1))
        val_probs = model.predict_proba(B) if (with_probs and hasattr(model, "predict_proba")) else None
        tr_scores = model.decision_function(A) if hasattr(model, "decision_function") else np.log(np.clip(model.predict_proba(A), 1e-12, 1))
        tr_probs = model.predict_proba(A) if (with_probs and hasattr(model, "predict_proba")) else None
        row = {"family": family, "params": params, "fit_seconds": round(fit_s, 1),
               "validation": _metrics(y_val, val_scores, val_probs, K), "train": _metrics(y_train, tr_scores, tr_probs, K)}
        candidates.append(row)
        v = row["validation"]
        console.print(f"  {family:22s} {json.dumps(params):22s} val acc {v['accuracy']:.4f}  top-3 {v['top3_accuracy']:.4f}  "
                      f"macro-F1 {v['macro_f1']:.4f}" + (f"  log loss {v['log_loss']:.4f}" if "log_loss" in v else "")
                      + f"  ({fit_s:.0f}s)")
        _event(events, event="candidate", family=family, params=params, fit_seconds=fit_s, **{f"val_{k}": x for k, x in v.items()})
        return row

    # ---- served family: logistic regression, C chosen on validation accuracy
    console.print("[bold]Logistic regression[/] (served model)")
    best, best_model = None, None
    for C in cfg.logistic_regression.C_grid:
        t = time.time()
        m = LogisticRegression(C=C, max_iter=cfg.logistic_regression.max_iter, random_state=cfg.seed).fit(A, y_train)
        row = record("logistic_regression", {"C": C}, m, time.time() - t)
        if best is None or row["validation"]["accuracy"] > best["validation"]["accuracy"]:
            best, best_model = row, m
    best["selected"] = True
    if list(best_model.classes_) != list(range(K)):
        raise RuntimeError("Not every clause type is present in the training split")

    # ---- comparison families (reported on the model page, not served)
    if cfg.linear_svm.enabled:
        console.print("[bold]Linear SVM[/] (comparison)")
        best_svm = None
        for C in cfg.linear_svm.C_grid:
            t = time.time()
            m = LinearSVC(C=C, random_state=cfg.seed).fit(A, y_train)
            row = record("linear_svm", {"C": C}, m, time.time() - t, with_probs=False)
            if best_svm is None or row["validation"]["accuracy"] > best_svm[0]["validation"]["accuracy"]:
                best_svm = (row, C)
        t = time.time()
        cal = CalibratedClassifierCV(LinearSVC(C=best_svm[1], random_state=cfg.seed), cv=cfg.linear_svm.calibration_cv,
                                     method="sigmoid").fit(A, y_train)
        record("linear_svm_calibrated", {"C": best_svm[1], "cv": cfg.linear_svm.calibration_cv}, cal, time.time() - t)
    if cfg.complement_nb.enabled:
        console.print("[bold]Complement Naive Bayes[/] (comparison)")
        for alpha in cfg.complement_nb.alpha_grid:
            t = time.time()
            m = ComplementNB(alpha=alpha).fit(A, y_train)
            record("complement_nb", {"alpha": alpha}, m, time.time() - t)

    # ---- learning curve for the served configuration
    console.print("[bold]Learning curve[/] (served configuration)")
    curve = []
    rng = random.Random(cfg.seed)
    C_best = best["params"]["C"]
    for frac in cfg.learning_curve:
        if frac >= 1.0:
            m, n, fit_s = best_model, len(y_train), best["fit_seconds"]
        else:
            idx = sorted(rng.sample(range(len(y_train)), int(frac * len(y_train))))
            t = time.time()
            m = LogisticRegression(C=C_best, max_iter=cfg.logistic_regression.max_iter, random_state=cfg.seed).fit(A[idx], y_train[idx])
            n, fit_s = len(idx), time.time() - t
        val = _metrics(y_val, m.decision_function(B), m.predict_proba(B), K) if len(m.classes_) == K else None
        point = {"fraction": frac, "train_examples": n, "fit_seconds": round(fit_s, 1), "validation": val}
        curve.append(point)
        if val:
            console.print(f"  {frac:>5.0%} of train ({n:>6,}): val acc {val['accuracy']:.4f}  log loss {val['log_loss']:.4f}")
        _event(events, event="learning_curve", fraction=frac, train_examples=n, **({f"val_{k}": x for k, x in val.items()} if val else {}))

    # ---- save the served pipeline
    pipeline = Pipeline([("features", features), ("clf", best_model)])
    joblib.dump(pipeline, merged / "model.joblib", compress=3)
    (merged / "labels.json").write_text(json.dumps({"names": space.names}, indent=2))
    space.save(merged / "label_space.json")
    (merged / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    from .explain import top_features

    (merged / "top_features.json").write_text(json.dumps(top_features(pipeline, space.names), indent=2))
    train_seconds = time.time() - t_start
    size_mb = (merged / "model.joblib").stat().st_size / 1e6
    write_model_info(
        merged, id=name, base_name=cfg.run_name, version=version, family="classical", backend="sklearn",
        base_model=None, method="tfidf+logistic_regression",
        classifier={"type": "LogisticRegression", "C": C_best, "max_iter": cfg.logistic_regression.max_iter},
        features=feat_info, num_labels=K, label_groups=cfg.data.label_groups,
        best_val_accuracy=best["validation"]["accuracy"], best_val_macro_f1=best["validation"]["macro_f1"],
        best_val_top3_accuracy=best["validation"]["top3_accuracy"], best_val_log_loss=best["validation"]["log_loss"],
        best_metric="accuracy", train_examples=len(y_train), train_seconds=round(train_seconds, 1),
        model_size_mb=round(size_mb, 1), run_dir=str(run_dir),
    )
    for p in merged.iterdir():
        p.chmod(0o644)
    (tel / "candidates.json").write_text(json.dumps(candidates, indent=2))
    (tel / "learning_curve.json").write_text(json.dumps(curve, indent=2))
    (tel / "meta.json").write_text(json.dumps({
        "run_name": name, "model_id": name, "model_version": version, "family": "classical",
        "config": cfg.to_dict(), "label_names": space.names,
        "label_groups": {"source": cfg.data.label_groups, "groups": space.groups, "raw_label_names": space.raw_names},
        "datasets": {"train": len(y_train), "validation": len(y_val)}, "features": feat_info,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "sklearn": __import__("sklearn").__version__, "processor": platform.processor()},
        "train_seconds": train_seconds,
    }, indent=2, default=str))
    _event(events, event="done", model_id=name, train_seconds=train_seconds, val_accuracy=best["validation"]["accuracy"])

    table = Table(title=f"{name}: candidates on the validation split")
    for col in ("model", "params", "val acc", "top-3", "macro-F1", "log loss", "fit s", ""):
        table.add_column(col)
    for c in candidates:
        v = c["validation"]
        table.add_row(c["family"], json.dumps(c["params"]), f"{v['accuracy']:.4f}", f"{v['top3_accuracy']:.4f}",
                      f"{v['macro_f1']:.4f}", f"{v['log_loss']:.4f}" if "log_loss" in v else "-", f"{c['fit_seconds']:.0f}",
                      "[green]served[/]" if c.get("selected") else "")
    console.print(table)
    console.print(f"[bold green]Model ready:[/] {name} -> {display_path(merged)} ({size_mb:.0f} MB, {train_seconds / 60:.1f} min)")
    return run_dir
