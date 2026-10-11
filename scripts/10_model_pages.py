"""Generate the model pages: docs/models/{compare,transformer,classical}.html (+ assets/*.png, data/*.json).

    uv run python scripts/10_model_pages.py                 # latest transformer + latest classical
    uv run python scripts/10_model_pages.py --transformer ledgar-qwen3-0.6b-v1 --classical ledgar-tfidf-lr-v1

Reads only existing artifacts (model_info.json, eval.json, config, telemetry). Model run folders are never
modified. The comparison needs both models' predictions on the full test split: the classical model saves
them during evaluation; the transformer's are computed once (about 3 minutes on a GPU) and cached in
artifacts/compare/<transformer-id>/test_predictions.npz. Pages are also served by the API at /models/*.
"""

from __future__ import annotations

import html
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from classification.config import PROJECT_ROOT, display_path  # noqa: E402
from classification.registry import CLASSICAL_ALIAS, ModelInfo, latest_model, list_models, resolve_model  # noqa: E402

app = typer.Typer(add_completion=False)
OUT = PROJECT_ROOT / "docs" / "models"
BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#b4b2a9", "#141413", "#6b6a66", "#e6e5e0"


# ----------------------------------------------------------------------------- reading artifacts

def jl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def jload(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def base_model_config(name: str) -> dict:
    try:
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(name, "config.json", local_files_only=True)).read_text())
    except Exception:
        return {}


def test_split(label_space_path: Path):
    from classification.config import DataConfig
    from classification.data import LabelSpace, load_split

    space = LabelSpace.load(label_space_path)
    texts, raw = load_split(DataConfig(), "test", raw=True)
    return texts, np.asarray(space.map(raw)), space


def transformer_test_predictions(info: ModelInfo, texts: list[str]) -> np.ndarray:
    """Calibrated probabilities on the test split, cached outside the model folder."""
    cache = PROJECT_ROOT / "artifacts" / "compare" / info.id / "test_predictions.npz"
    if cache.exists():
        return np.load(cache)["probs"].astype(np.float32)
    from classification.predict import load_predictor

    print(f"Scoring {len(texts):,} test provisions with {info.id} (cached afterwards) ...")
    t = time.time()
    probs = load_predictor(info.path, device="auto", batch_size=32).backend.predict_proba(texts)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, probs=probs.astype(np.float16), seconds=time.time() - t)
    return probs


def cpu_latency_ms(predict, texts: list[str], n: int) -> float:
    predict(texts[:2])
    t = time.perf_counter()
    for x in texts[:n]:
        predict([x])
    return (time.perf_counter() - t) * 1000 / n


# ----------------------------------------------------------------------------- charts

def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str) -> str:
    (OUT / "assets").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "assets" / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"assets/{name}"


def per_class_chart(report: dict, title: str, name: str, color: str) -> str:
    rows = sorted(((v["f1-score"], k, int(v["support"])) for k, v in report["per_class"].items()), reverse=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.19 * len(rows))))
    ax.barh(range(len(rows)), [r[0] for r in rows], color=color, height=0.7)
    ax.set_yticks(range(len(rows)), [f"{r[1]} ({r[2]})" for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    _style(ax, title, "F1 on the test split (support in brackets)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    return save(fig, name)


# ----------------------------------------------------------------------------- HTML

CSS = """
:root{--bg:#fcfcfb;--surface:#fff;--text:#141413;--muted:#5f5e5a;--faint:#8a8984;--border:rgba(20,20,19,.12);
--accent:#185fa5;--accent-bg:#e6f1fb;--good:#27500a;--good-bg:#eaf3de;--warn:#633806;--warn-bg:#faeeda;
--font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--surface:#20201f;--text:#efeee9;--muted:#c3c2b7;
--faint:#8a8984;--border:rgba(255,255,255,.12);--accent:#85b7eb;--accent-bg:#0c2f52;--good:#c0dd97;--good-bg:#233d0b;--warn:#fac775;--warn-bg:#4a2c05}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 var(--font)}
main{max-width:1060px;margin:0 auto;padding:28px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 10px}
h3{font-size:15px;margin:20px 0 6px}p{margin:0 0 10px}.sub{color:var(--muted);margin:0}.meta{color:var(--faint);font-size:13px}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 22px}nav a{color:var(--accent);text-decoration:none;border:1px solid var(--border);
border-radius:8px;padding:4px 10px;font-size:14px}nav a.on{background:var(--accent-bg)}.card{background:var(--surface);border:1px solid var(--border);
border-radius:12px;padding:18px 20px;margin:14px 0}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 14px}.kpi .l{font-size:12px;color:var(--muted)}
.kpi .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}table{border-collapse:collapse;width:100%;font-size:14px;margin:6px 0 12px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);vertical-align:top}th{color:var(--muted);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}code{font-family:var(--mono);font-size:.9em;background:var(--accent-bg);padding:1px 5px;border-radius:4px}
img{max-width:100%;border-radius:8px;border:1px solid var(--border);background:#fff}.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media (max-width:820px){.two{grid-template-columns:1fr}}.tag{font-size:12px;padding:2px 8px;border-radius:999px;background:var(--accent-bg);color:var(--accent)}
.win{background:var(--good-bg);color:var(--good)}.lose{background:var(--warn-bg);color:var(--warn)}details summary{cursor:pointer;color:var(--muted)}
.terms{font-size:13px;color:var(--muted)}footer{color:var(--faint);font-size:13px;margin-top:40px;border-top:1px solid var(--border);padding-top:14px}
a{color:var(--accent)}
"""


def page(title: str, active: str, body: str, generated: str) -> str:
    links = [("compare.html", "Compare models", "compare"), ("transformer.html", "Transformer", "transformer"),
             ("classical.html", "Classical", "classical"), ("../../changelog.html", "Changelog", ""),
             ("/playground", "Playground", "")]
    nav = "".join(f'<a href="{h}" class="{"on" if k == active else ""}">{t}</a>' for h, t, k in links)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · Contract Clause Classifier</title><style>{CSS}</style></head>
<body><main>
<h1>{html.escape(title)}</h1>
<p class="sub">Contract Clause Classifier: LEDGAR clause types, 78 classes, same data and splits for every model.</p>
<nav>{nav}</nav>
{body}
<footer>Generated {generated} by <code>scripts/10_model_pages.py</code> from the saved model artifacts. Regenerate with <code>make model-pages</code>.</footer>
</main></body></html>
"""


def pct(x: float | None, d: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{d}f}%"


def kpis(items: list[tuple[str, str]]) -> str:
    return '<div class="kpis">' + "".join(f'<div class="kpi"><div class="l">{html.escape(l)}</div><div class="v">{v}</div></div>'
                                           for l, v in items) + "</div>"


def table(head: list[str], rows: list[list], numeric: set[int] = frozenset()) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    trs = "".join("<tr>" + "".join(f'<td class="{"n" if i in numeric else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def confusions_table(report: dict, n: int = 20) -> str:
    return table(["True clause type", "Predicted as", "Count"],
                 [[html.escape(c["true"]), html.escape(c["pred"]), c["count"]] for c in report["top_confusions"][:n]], {2})


def versions_table(models: list[ModelInfo]) -> str:
    rows = []
    for m in reversed(models):
        t = m.info.get("test", {})
        rows.append([f"<code>{m.id}</code>", m.version if m.version is not None else "–",
                     datetime.fromtimestamp(m.created).strftime("%Y-%m-%d %H:%M"), pct(m.info.get("best_val_accuracy"), 2),
                     pct(t.get("accuracy"), 2), pct(t.get("top3_accuracy"), 2), pct(t.get("macro_f1"), 2)])
    return table(["Model", "Version", "Created", "Val acc", "Test acc", "Test top-3", "Test macro-F1"], rows, {3, 4, 5, 6})


# ----------------------------------------------------------------------------- transformer page

def transformer_page(info: ModelInfo, all_models: list[ModelInfo], generated: str) -> tuple[str, dict]:
    run, merged, tel = info.run_dir, info.path, info.run_dir / "telemetry"
    mi, ev = info.info, jload(merged / "eval.json", {})
    cal = jload(merged / "calibration.json", {})
    cfg = yaml.safe_load((run / "config.yaml").read_text()) if (run / "config.yaml").exists() else {}
    meta = jload(tel / "meta.json", {})
    steps, evals, events = jl(tel / "steps.jsonl"), jl(tel / "evals.jsonl"), jl(tel / "events.jsonl")
    base = base_model_config(mi.get("base_model") or cfg.get("model_name", "Qwen/Qwen3-0.6B-Base"))
    t_start = next((e["time"] for e in events if e["event"] == "start"), None)
    t_done = next((e["time"] for e in events if e["event"] == "done"), None)
    duration = (t_done - t_start) if (t_start and t_done) else None
    spe = (meta.get("model") or {}).get("steps_per_epoch") or 1875
    tokens = sum(s.get("tokens", 0) for s in steps)
    peak_mem = max((s.get("mem_gb", 0) for s in steps), default=0)
    tr = cfg.get("train", {})

    # charts
    x = np.array([s["step"] for s in steps])
    loss = np.array([s["loss"] for s in steps])
    ema, e = [], loss[0] if len(loss) else 0
    for v in loss:
        e = 0.98 * e + 0.02 * v
        ema.append(e)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [2, 1]})
    a1.plot(x, loss, color=GRAY, linewidth=0.5, alpha=0.7, label="per step")
    a1.plot(x, ema, color=BLUE, linewidth=2, label="smoothed")
    for k in range(1, int(x.max() // spe) + 1 if len(x) else 1):
        a1.axvline(k * spe, color=MUTED, linestyle=":", linewidth=1)
        a1.text(k * spe, a1.get_ylim()[1] * 0.95, f" epoch {k}", fontsize=7, color=MUTED)
    _style(a1, "Training loss (cross-entropy with label smoothing 0.1)", "optimizer step")
    a1.legend(frameon=False, fontsize=8)
    a2.plot(x, [s["lr"] for s in steps], color=ORANGE, linewidth=1.5)
    _style(a2, "Learning rate", "optimizer step")
    loss_png = save(fig, "transformer_loss.png")

    fig, ax = plt.subplots(figsize=(11, 3.4))
    es = [v["step"] for v in evals]
    ax.plot(es, [100 * v["accuracy"] for v in evals], "-o", color=BLUE, ms=4, label="accuracy")
    ax.plot(es, [100 * v.get("top3_accuracy", np.nan) for v in evals], "-^", color="#1baf7a", ms=4, label="top-3 accuracy")
    ax.plot(es, [100 * v["macro_f1"] for v in evals], "--s", color=ORANGE, ms=4, label="macro-F1")
    _style(ax, "Validation metrics at each evaluation (3,000-row validation subset)", "optimizer step", "%")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    val_png = save(fig, "transformer_validation.png")
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.plot(es, [v["loss"] for v in evals], "-o", color=BLUE, ms=4)
    _style(ax, "Validation loss (cross-entropy, no smoothing)", "optimizer step")
    vloss_png = save(fig, "transformer_validation_loss.png")
    f1_png = per_class_chart(ev, f"{info.id}: per-class F1 on the test split", "transformer_per_class_f1.png", BLUE) if ev else ""

    test = ev or {}
    body = f"""
<p>Fine-tuned transformer. A pretrained language model reads each provision in context and a classification
head maps its final representation to one of the 78 clause types. Latest version: <code>{info.id}</code>.</p>
{kpis([("Test accuracy", pct(test.get("accuracy"), 2)), ("Test top-3 accuracy", pct(test.get("top3_accuracy"), 2)),
       ("Test macro-F1", pct(test.get("macro_f1"), 2)), ("Calibration error", pct(test.get("ece"), 1)),
       ("Training time", f"{duration / 3600:.1f} h" if duration else "–"), ("Parameters", f"{(meta.get('model') or {}).get('total_params', 0) / 1e6:.0f}M")])}

<h2>Model</h2>
<div class="card">{table(["Setting", "Value"], [
    ["Base model", f"<code>{html.escape(str(mi.get('base_model') or cfg.get('model_name')))}</code> (Apache-2.0)"],
    ["Architecture", f"decoder-only transformer, {base.get('num_hidden_layers', '28')} layers, hidden size {base.get('hidden_size', 1024)}, "
                     f"MLP {base.get('intermediate_size', 3072)}, attention {base.get('num_attention_heads', 16)} query / "
                     f"{base.get('num_key_value_heads', 8)} key-value heads, vocabulary {base.get('vocab_size', 151936):,}"],
    ["Head", f"linear {base.get('hidden_size', 1024)} → {mi.get('num_labels', 78)} on the last real token"],
    ["Method", "full fine-tuning (all weights)" if mi.get("method") == "full" else html.escape(str(mi.get("method")))],
    ["Parameters trained", f"{(meta.get('model') or {}).get('trainable_params', 0):,}"],
    ["Serving size", f"{(merged / 'model.safetensors').stat().st_size / 1e9:.2f} GB (bf16 safetensors)" if (merged / 'model.safetensors').exists() else "–"],
])}</div>

<h2>Training</h2>
<div class="card">{table(["Setting", "Value"], [
    ["Optimizer", f"AdamW, learning rate {tr.get('lr')}, weight decay {tr.get('weight_decay')}, warm-up {pct(tr.get('warmup_ratio'), 0)}, cosine decay"],
    ["Epochs / steps", f"{tr.get('epochs')} epochs, {len(steps):,} optimizer steps ({spe:,} per epoch)"],
    ["Batch", f"{tr.get('batch_size')} × {tr.get('grad_accum')} gradient accumulation = {int(tr.get('batch_size', 0)) * int(tr.get('grad_accum', 0))}"],
    ["Regularisation", f"label smoothing {tr.get('label_smoothing')}, gradient clipping {tr.get('max_grad_norm')}"],
    ["Max length", f"{(cfg.get('data') or {}).get('max_len', 512)} tokens per provision"],
    ["Checkpoint selection", f"best validation {tr.get('best_metric', 'accuracy')}; early stopping patience {tr.get('early_stopping_patience')}"],
    ["Compute", f"a single laptop with {((meta.get('environment') or {}).get('memory_gb') or 48):.0f} GB of memory, built-in GPU via PyTorch "
                f"({(meta.get('environment') or {}).get('device', '')}); {duration / 3600:.2f} h; {tokens / 1e6:.0f}M tokens; peak memory {peak_mem:.0f} GB" if duration else "–"],
])}</div>
<img src="{loss_png}" alt="Training loss and learning rate">
<h3>Validation during training</h3>
<img src="{val_png}" alt="Validation accuracy, top-3 and macro-F1">
<img src="{vloss_png}" alt="Validation loss">
{table(["Step", "Epoch", "Accuracy", "Top-3", "Macro-F1", "Loss"], [[v["step"], f"{v['epoch']:.2f}", pct(v["accuracy"], 2), pct(v.get("top3_accuracy"), 2),
       pct(v["macro_f1"], 2), f"{v['loss']:.4f}"] for v in evals], {0, 1, 2, 3, 4, 5})}

<h2>Test results (full 10,000-provision test split)</h2>
<div class="card">{table(["Metric", "Value"], [
    ["Accuracy", pct(test.get("accuracy"), 2)], ["Top-3 accuracy", pct(test.get("top3_accuracy"), 2)],
    ["Micro / macro / weighted F1", f"{pct(test.get('micro_f1'), 2)} / {pct(test.get('macro_f1'), 2)} / {pct(test.get('weighted_f1'), 2)}"],
    ["Log loss (calibrated)", f"{test.get('nll', 0):.4f}"],
    ["Calibration error", f"{pct(test.get('ece_uncalibrated'), 1)} before, {pct(test.get('ece'), 1)} after temperature {cal.get('temperature', 1):.3f}"],
])}</div>
{f'<img src="{f1_png}" alt="Per-class F1">' if f1_png else ''}
<h3>Most frequent confusions</h3>
{confusions_table(test) if test else ''}

<h2>Versions</h2>
{versions_table(all_models)}
<h2>Training in motion</h2>
<p>Recorded from this run: <a href="../../demo/{info.id}_training.gif">training charts GIF</a>,
<a href="../../demo/{info.id}_terminal.gif">terminal dashboard GIF</a>, <a href="../../demo/playground_usage.gif">playground GIF</a>.</p>
"""
    data = {"id": info.id, "model_info": mi, "test": {k: test.get(k) for k in ("accuracy", "top3_accuracy", "micro_f1", "macro_f1",
            "weighted_f1", "nll", "ece", "ece_uncalibrated", "temperature")}, "evals": evals, "training_seconds": duration,
            "tokens": tokens, "peak_memory_gb": peak_mem}
    return page(f"Transformer model: {info.id}", "transformer", body, generated), data


# ----------------------------------------------------------------------------- classical page

def classical_page(info: ModelInfo, all_models: list[ModelInfo], generated: str) -> tuple[str, dict]:
    merged, tel = info.path, info.run_dir / "telemetry"
    mi, ev = info.info, jload(merged / "eval.json", {})
    cal = jload(merged / "calibration.json", {})
    cands = jload(tel / "candidates.json", [])
    curve = jload(tel / "learning_curve.json", [])
    topf = jload(merged / "top_features.json", {})
    feats = mi.get("features", {})

    labels = [f"{c['family'].replace('_', ' ')}\n{', '.join(f'{k}={v}' for k, v in c['params'].items())}" for c in cands]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    colors = [BLUE if c.get("selected") else (ORANGE if "svm" in c["family"] else GRAY) for c in cands]
    ax.bar(range(len(cands)), [100 * c["validation"]["accuracy"] for c in cands], color=colors)
    ax.set_xticks(range(len(cands)), labels, fontsize=7)
    lo = min([100 * c["validation"]["accuracy"] for c in cands] + [80]) - 2
    ax.set_ylim(lo, 100)
    for i, c in enumerate(cands):
        ax.text(i, 100 * c["validation"]["accuracy"] + 0.2, f"{100 * c['validation']['accuracy']:.2f}", ha="center", fontsize=7, color=INK)
    _style(ax, "Validation accuracy by candidate (blue: served)", "", "%")
    cand_png = save(fig, "classical_candidates.png")

    pts = [p for p in curve if p.get("validation")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.2))
    n = [p["train_examples"] for p in pts]
    a1.plot(n, [100 * p["validation"]["accuracy"] for p in pts], "-o", color=BLUE)
    _style(a1, "Learning curve: validation accuracy", "training provisions", "%")
    a2.plot(n, [p["validation"]["log_loss"] for p in pts], "-o", color=ORANGE)
    _style(a2, "Learning curve: validation log loss", "training provisions")
    curve_png = save(fig, "classical_learning_curve.png")
    f1_png = per_class_chart(ev, f"{info.id}: per-class F1 on the test split", "classical_per_class_f1.png", ORANGE) if ev else ""

    served = next((c for c in cands if c.get("selected")), {})
    test = ev or {}
    cand_rows = [[html.escape(c["family"].replace("_", " ")) + (' <span class="tag">served</span>' if c.get("selected") else ""),
                  html.escape(", ".join(f"{k}={v}" for k, v in c["params"].items())), pct(c["validation"]["accuracy"], 2),
                  pct(c["validation"]["top3_accuracy"], 2), pct(c["validation"]["macro_f1"], 2),
                  f"{c['validation']['log_loss']:.4f}" if "log_loss" in c["validation"] else "–",
                  f"{c['train']['log_loss']:.4f}" if "log_loss" in c.get("train", {}) else "–", f"{c['fit_seconds']:.0f}"] for c in cands]
    phrases = "".join(f"<tr><td>{html.escape(k)}</td><td class='terms'>{html.escape(', '.join(t['term'] for t in v[:10]))}</td></tr>"
                      for k, v in sorted(topf.items()))
    body = f"""
<p>Classical machine-learning model. Each provision becomes a sparse TF-IDF vector of word phrases and character
fragments, and a linear classifier (logistic regression) scores the 78 clause types. No neural network, CPU only,
and every decision can be traced to weighted phrases. Latest version: <code>{info.id}</code>.</p>
{kpis([("Test accuracy", pct(test.get("accuracy"), 2)), ("Test top-3 accuracy", pct(test.get("top3_accuracy"), 2)),
       ("Test macro-F1", pct(test.get("macro_f1"), 2)), ("Calibration error", pct(test.get("ece"), 1)),
       ("Training time", f"{mi.get('train_seconds', 0) / 60:.0f} min (CPU)"), ("Model size", f"{mi.get('model_size_mb', 0):.0f} MB")])}

<h2>Model</h2>
<div class="card">{table(["Setting", "Value"], [
    ["Features", f"TF-IDF, sublinear term frequency: word 1-2 grams ({feats.get('word', {}).get('n_features', 0):,} features) + "
                 f"character 3-5 grams within words ({feats.get('char', {}).get('n_features', 0):,}); {feats.get('total', 0):,} in total"],
    ["Classifier", f"multinomial logistic regression, C = {(mi.get('classifier') or {}).get('C')}, lbfgs"],
    ["Selection", "C chosen by validation accuracy among the logistic-regression candidates; SVM and Naive Bayes trained for comparison"],
    ["Calibration", f"temperature {cal.get('temperature', 1):.3f} fitted on the 10,000-row validation split"],
    ["Inference", f"{mi.get('latency_ms_per_clause_cpu', 0):.1f} ms per provision on one CPU core"],
    ["Dependencies", "scikit-learn + joblib, no GPU"],
])}</div>

<h2>Training: candidates and loss</h2>
<p>Every candidate is trained on the same 60,000 provisions and the same features, and scored on the full validation
split. Log loss is the cross-entropy of the predicted probabilities (lower is better); a large gap between training
and validation log loss means over-fitting.</p>
<img src="{cand_png}" alt="Validation accuracy by candidate">
{table(["Model", "Parameters", "Val acc", "Val top-3", "Val macro-F1", "Val log loss", "Train log loss", "Fit s"], cand_rows, {2, 3, 4, 5, 6, 7})}
<h3>Learning curve (served configuration)</h3>
<p>Accuracy keeps rising with more data: {", ".join(f"{p['train_examples']:,} → {pct(p['validation']['accuracy'], 1)}" for p in pts)}.</p>
<img src="{curve_png}" alt="Learning curve">

<h2>Test results (full 10,000-provision test split)</h2>
<div class="card">{table(["Metric", "Value"], [
    ["Accuracy", pct(test.get("accuracy"), 2)], ["Top-3 accuracy", pct(test.get("top3_accuracy"), 2)],
    ["Micro / macro / weighted F1", f"{pct(test.get('micro_f1'), 2)} / {pct(test.get('macro_f1'), 2)} / {pct(test.get('weighted_f1'), 2)}"],
    ["Log loss (calibrated)", f"{test.get('nll', 0):.4f}"],
    ["Calibration error", f"{pct(test.get('ece_uncalibrated'), 1)} before, {pct(test.get('ece'), 1)} after temperature scaling"],
])}</div>
{f'<img src="{f1_png}" alt="Per-class F1">' if f1_png else ''}
<h3>Most frequent confusions</h3>
{confusions_table(test) if test else ''}

<h2>Why it decides: top phrases per clause type</h2>
<p>The word phrases with the largest positive weight for each clause type in the served model.</p>
<details open><summary>{len(topf)} clause types</summary><table><thead><tr><th>Clause type</th><th>Strongest phrases</th></tr></thead><tbody>{phrases}</tbody></table></details>

<h2>Versions</h2>
{versions_table(all_models)}
"""
    data = {"id": info.id, "model_info": mi, "candidates": cands, "learning_curve": curve,
            "test": {k: test.get(k) for k in ("accuracy", "top3_accuracy", "micro_f1", "macro_f1", "weighted_f1", "nll", "ece",
                                                "ece_uncalibrated", "temperature")}, "served": served.get("params")}
    return page(f"Classical model: {info.id}", "classical", body, generated), data


# ----------------------------------------------------------------------------- comparison page

def compare_page(t: ModelInfo, c: ModelInfo, tdata: dict, cdata: dict, generated: str, latency_n: int) -> tuple[str, dict]:
    tev, cev = jload(t.path / "eval.json", {}), jload(c.path / "eval.json", {})
    texts, y, space = test_split(c.path / "label_space.json")
    t_space = jload(t.path / "label_space.json", {})
    same_space = (t_space.get("names") == space.names) if t_space else (jload(t.path / "labels.json", {}).get("names") == space.names)
    agree_block, adata = "<p>The two models use different label sets, so predictions cannot be compared clause by clause.</p>", {}
    if same_space:
        cp = np.load(c.run_dir / "telemetry" / "test_predictions.npz")["probs"].astype(np.float32) \
            if (c.run_dir / "telemetry" / "test_predictions.npz").exists() else None
        tp = transformer_test_predictions(t, texts)
        if cp is not None:
            tpred, cpred = tp.argmax(1), cp.argmax(1)
            agree = tpred == cpred
            t_ok, c_ok = tpred == y, cpred == y
            adata = {"n": int(len(y)), "agree_rate": float(agree.mean()), "acc_when_agree": float(t_ok[agree].mean()),
                     "disagree": int((~agree).sum()), "transformer_right_when_disagree": float(t_ok[~agree].mean()),
                     "classical_right_when_disagree": float(c_ok[~agree].mean()),
                     "both_wrong_when_disagree": float((~t_ok & ~c_ok)[~agree].mean()), "either_right": float((t_ok | c_ok).mean()),
                     "transformer_accuracy": float(t_ok.mean()), "classical_accuracy": float(c_ok.mean())}
            agree_block = f"""
<p>On the full test split (10,000 provisions) the two models give the same top-1 clause type for
<b>{pct(adata['agree_rate'])}</b> of provisions. When they agree they are right <b>{pct(adata['acc_when_agree'])}</b> of the time,
so agreement is a strong confidence signal. On the {adata['disagree']:,} provisions where they disagree, the transformer is right
{pct(adata['transformer_right_when_disagree'])}, the classical model {pct(adata['classical_right_when_disagree'])}, and both are wrong
{pct(adata['both_wrong_when_disagree'])}. At least one of the two is right for {pct(adata['either_right'])} of provisions.</p>"""

    # per-class F1 difference
    names = sorted(set(tev.get("per_class", {})) & set(cev.get("per_class", {})))
    diffs = sorted(((tev["per_class"][n]["f1-score"] - cev["per_class"][n]["f1-score"], n) for n in names))
    delta_png = ""
    if diffs:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.19 * len(diffs))))
        ax.barh(range(len(diffs)), [d for d, _ in diffs], color=[BLUE if d > 0 else ORANGE for d, _ in diffs], height=0.7)
        ax.set_yticks(range(len(diffs)), [n for _, n in diffs], fontsize=7)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        _style(ax, "Per-class F1: transformer minus classical (blue: transformer better, orange: classical better)", "F1 difference")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        delta_png = save(fig, "compare_per_class_f1_delta.png")
    t_wins = sum(d > 0.005 for d, _ in diffs)
    c_wins = sum(d < -0.005 for d, _ in diffs)

    # latency on CPU, same provisions
    from classification.classical.backend import SklearnBackend
    from classification.predict import load_predictor

    c_lat = cpu_latency_ms(SklearnBackend(c.path).predict_proba, texts, latency_n)
    t_lat = cpu_latency_ms(load_predictor(t.path, device="cpu").backend.predict_proba, texts, max(20, latency_n // 5))

    def cell(tv, cv, higher_better=True, fmt=lambda v: pct(v, 2)):
        if tv is None or cv is None:
            return [fmt(tv) if tv is not None else "–", fmt(cv) if cv is not None else "–"]
        tw = (tv > cv) if higher_better else (tv < cv)
        return [f'{fmt(tv)} <span class="tag {"win" if tw else "lose"}">{"better" if tw else ""}</span>' if tv != cv else fmt(tv),
                f'{fmt(cv)} <span class="tag {"lose" if tw else "win"}">{"" if tw else "better"}</span>' if tv != cv else fmt(cv)]

    tm, cm = t.info, c.info
    rows = [
        ["Test accuracy", *cell(tev.get("accuracy"), cev.get("accuracy"))],
        ["Test top-3 accuracy", *cell(tev.get("top3_accuracy"), cev.get("top3_accuracy"))],
        ["Test macro-F1", *cell(tev.get("macro_f1"), cev.get("macro_f1"))],
        ["Calibration error (lower is better)", *cell(tev.get("ece"), cev.get("ece"), False, lambda v: pct(v, 1))],
        ["Log loss (lower is better)", *cell(tev.get("nll"), cev.get("nll"), False, lambda v: f"{v:.4f}")],
        ["Clause types where it has higher F1", f"{t_wins}", f"{c_wins}"],
        ["Training time", f"{(tdata.get('training_seconds') or 0) / 3600:.1f} h on a laptop GPU", f"{cm.get('train_seconds', 0) / 60:.0f} min on CPU"],
        ["Inference per provision (CPU)", f"{t_lat:.0f} ms", f"{c_lat:.1f} ms"],
        ["Model size", f"{(t.path / 'model.safetensors').stat().st_size / 1e9:.1f} GB" if (t.path / 'model.safetensors').exists() else "–",
         f"{cm.get('model_size_mb', 0):.0f} MB"],
        ["Parameters / features", f"{(jload(t.run_dir / 'telemetry' / 'meta.json', {}).get('model') or {}).get('total_params', 0) / 1e6:.0f}M parameters",
         f"{(cm.get('features') or {}).get('total', 0):,} features × {cm.get('num_labels', 78)} classes"],
        ["Explainability", "no direct explanation", "weighted phrases per clause type"],
        ["Runtime needs", "PyTorch + transformers; GPU optional", "scikit-learn only; CPU"],
    ]
    body = f"""
<p>Two different approaches trained on exactly the same data: <code>{t.id}</code> (fine-tuned transformer) and
<code>{c.id}</code> (TF-IDF + logistic regression). Same 60,000 training provisions, 78 clause types, and the same
10,000-provision test split for every number below. <a href="transformer.html">Transformer details</a> ·
<a href="classical.html">Classical details</a>.</p>
{kpis([("Transformer accuracy", pct(tev.get("accuracy"), 2)), ("Classical accuracy", pct(cev.get("accuracy"), 2)),
       ("Gap", f"{100 * ((tev.get('accuracy') or 0) - (cev.get('accuracy') or 0)):+.2f} pts"),
       ("Agreement", pct(adata.get("agree_rate")) if adata else "–"), ("Either right", pct(adata.get("either_right")) if adata else "–")])}

<h2>Side by side</h2>
<div class="card">{table(["", "Transformer", "Classical"], rows)}</div>

<h2>Where they agree and disagree</h2>
<div class="card">{agree_block}</div>

<h2>Per clause type</h2>
<p>The transformer has higher F1 on {t_wins} clause types and the classical model on {c_wins} (differences under 0.5 points not counted).</p>
{f'<img src="{delta_png}" alt="Per-class F1 difference">' if delta_png else ''}
<div class="two"><div><h3>Transformer: top confusions</h3>{confusions_table(tev, 10) if tev else ''}</div>
<div><h3>Classical: top confusions</h3>{confusions_table(cev, 10) if cev else ''}</div></div>

<h2>Which to use</h2>
<div class="card"><p><b>Transformer</b> for the highest accuracy and for unusual or paraphrased wording, since it reads
meaning rather than matching words. <b>Classical</b> when speed, cost or explanations matter: it is about
{t_lat / max(c_lat, 0.01):.0f}× faster per provision on a CPU, a fraction of the size, trains in minutes, and every prediction
can be traced to phrases. <b>Both together</b> (the playground's compare mode, <code>POST /v1/compare</code>): when they
agree, the answer is very likely right; when they disagree, the provision deserves a human look.</p></div>
"""
    data = {"transformer": t.id, "classical": c.id, "agreement": adata, "per_class_f1_delta": {n: d for d, n in diffs},
            "latency_ms_cpu": {"transformer": t_lat, "classical": c_lat}, "transformer_wins": t_wins, "classical_wins": c_wins}
    return page("Transformer vs classical", "compare", body, generated), data


@app.command()
def main(
    transformer: str = typer.Option("latest", help="Transformer model id or 'latest'"),
    classical: str = typer.Option(CLASSICAL_ALIAS, help="Classical model id or 'latest-classical'"),
    latency_samples: int = typer.Option(200, help="Provisions timed per model for CPU latency"),
) -> None:
    t_models = [m for m in list_models(PROJECT_ROOT / "artifacts" / "runs") if m.family == "transformer"]
    c_models = list_models(PROJECT_ROOT / "artifacts" / "classical" / "runs")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    tinfo = resolve_model(t_models, transformer) if t_models else None
    cinfo = resolve_model(c_models, classical) if c_models else None
    tdata = cdata = None
    if tinfo:
        html_, tdata = transformer_page(tinfo, t_models, generated)
        (OUT / "transformer.html").write_text(html_)
        (OUT / "data" / "transformer.json").write_text(json.dumps(tdata, indent=2, default=str))
        print(f"wrote {display_path(OUT / 'transformer.html')}")
    if cinfo:
        html_, cdata = classical_page(cinfo, c_models, generated)
        (OUT / "classical.html").write_text(html_)
        (OUT / "data" / "classical.json").write_text(json.dumps(cdata, indent=2, default=str))
        print(f"wrote {display_path(OUT / 'classical.html')}")
    if tinfo and cinfo:
        html_, data = compare_page(tinfo, cinfo, tdata, cdata, generated, latency_samples)
        (OUT / "compare.html").write_text(html_)
        (OUT / "data" / "compare.json").write_text(json.dumps(data, indent=2, default=str))
        print(f"wrote {display_path(OUT / 'compare.html')}")
    else:
        print("compare.html needs both a transformer and a classical model (make all / make classical)")


if __name__ == "__main__":
    app()
