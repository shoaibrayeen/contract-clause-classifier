"""Render a run's telemetry into an animated GIF of the training process.

    uv run python scripts/06_render_gif.py                                   # latest run
    uv run python scripts/06_render_gif.py --run artifacts/runs/sanity --frames 80 --fps 10

Panels (animated over optimizer steps):
    1. training loss (raw + smoothed) revealed up to the current step
    2. validation accuracy / macro-F1 at each evaluation so far
    3. gradient norm per transformer layer over time (heatmap fills in)
    4. probe-set clause embeddings (PCA) moving between snapshots -> clusters forming

Output: <run>/visuals/training.gif (+ the last frame as PNG). Works while a run is still
in progress: it renders whatever telemetry has been written so far.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import typer  # noqa: E402
from PIL import Image  # noqa: E402

app = typer.Typer(add_completion=False)

HIGHLIGHT = ["Governing Laws", "Confidentiality", "Terminations", "Notices", "Indemnifications", "Taxes"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "*", "X"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a line still being written
            pass
    return out


def latest_run(root: Path) -> Path:
    runs = [p for p in root.glob("*/telemetry/steps.jsonl")]
    if not runs:
        raise typer.BadParameter(f"No runs with telemetry under {root}")
    return max(runs, key=lambda p: p.stat().st_mtime).parents[1]


def ema(values: list[float], alpha: float = 0.1) -> np.ndarray:
    out, e = [], values[0] if values else 0.0
    for v in values:
        e = (1 - alpha) * e + alpha * v
        out.append(e)
    return np.asarray(out)


def load_snapshots(tel: Path, names: list[str]):
    files = sorted(glob.glob(str(tel / "snapshots" / "step_*.npz")))
    if not files:
        return None
    snaps = [np.load(f) for f in files]
    steps = np.array([int(s["step"]) for s in snaps])
    labels = snaps[0]["labels"]
    embs = [s["embeddings"].astype(np.float32) for s in snaps]
    if embs[0].shape[1] == 0:
        return None
    accs = np.array([(s["probs"].astype(np.float32).argmax(-1) == labels).mean() for s in snaps])
    # Fit PCA on the latest snapshot and project every snapshot into that fixed 2-D basis.
    X = embs[-1] - embs[-1].mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    W = vt[:2].T
    proj = [(e - e.mean(0)) @ W for e in embs]
    # Robust per-snapshot scaling (98th percentile) so a few outliers don't squash the cloud.
    proj = [p / (np.percentile(np.abs(p), 98) or 1.0) for p in proj]
    proj = [np.clip(p, -1.1, 1.1) for p in proj]
    hl = {names.index(n): i for i, n in enumerate(HIGHLIGHT) if n in names}
    group = np.array([hl.get(int(y), -1) for y in labels])
    return {"steps": steps, "proj": proj, "group": group, "acc": accs}


def interp_snapshot(snaps, step: float) -> tuple[np.ndarray, float, int]:
    s = snaps["steps"]
    if step <= s[0]:
        return snaps["proj"][0], snaps["acc"][0], int(s[0])
    if step >= s[-1]:
        return snaps["proj"][-1], snaps["acc"][-1], int(s[-1])
    j = int(np.searchsorted(s, step))
    t = (step - s[j - 1]) / max(1, s[j] - s[j - 1])
    t = t * t * (3 - 2 * t)  # smoothstep for nicer motion
    pts = (1 - t) * snaps["proj"][j - 1] + t * snaps["proj"][j]
    return pts, snaps["acc"][j - 1] if t < 0.5 else snaps["acc"][j], int(s[j - 1] if t < 0.5 else s[j])


def style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


@app.command()
def main(
    run: Optional[Path] = typer.Option(None, help="Run dir (default: most recently updated under artifacts/runs)"),
    frames: int = typer.Option(72, help="Animation frames (plus a hold at the end)"),
    fps: int = typer.Option(8),
    out: Optional[Path] = typer.Option(None, help="Output GIF path (default <run>/visuals/training.gif)"),
    dpi: int = typer.Option(90),
) -> None:
    run = run or latest_run(Path("artifacts/runs"))
    tel = run / "telemetry"
    meta = json.loads((tel / "meta.json").read_text())
    names = meta["label_names"]
    steps = read_jsonl(tel / "steps.jsonl")
    evals = read_jsonl(tel / "evals.jsonl")
    layers = read_jsonl(tel / "layers.jsonl")
    if not steps:
        raise typer.BadParameter(f"No steps recorded yet in {tel}")
    snaps = load_snapshots(tel, names)

    x = np.array([s["step"] for s in steps])
    loss = np.array([s["loss"] for s in steps])
    smooth = ema(list(loss), alpha=0.05 if len(loss) > 500 else 0.15)
    total = meta.get("model", {}).get("total_steps", int(x[-1]))
    last = int(x[-1])

    keys = sorted({k for l in layers for k in l["grad_norm"] if k.startswith("layer_")}) + ["head"]
    G = np.array([[l["grad_norm"].get(k, np.nan) for k in keys] for l in layers]) if layers else None
    gsteps = np.array([l["step"] for l in layers]) if layers else None

    out = out or run / "visuals" / "training.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame_steps = np.linspace(x[0], last, frames)
    images: list[Image.Image] = []

    fig = plt.figure(figsize=(11, 6.6), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.18, left=0.06, right=0.98, top=0.84, bottom=0.08)
    for fi, cur in enumerate(frame_steps):
        fig.clf()
        gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.18, left=0.06, right=0.98, top=0.84, bottom=0.08)
        m = x <= cur
        i_cur = max(0, int(m.sum()) - 1)
        srec = steps[i_cur]
        fig.text(0.06, 0.95, f"{meta.get('run_name', run.name)}  ·  {meta['model']['name']}", fontsize=13, color=INK)
        fig.text(0.06, 0.915,
                 f"step {int(srec['step'])}/{total}   epoch {srec['epoch']:.2f}   loss {srec['loss']:.3f}   "
                 f"lr {srec['lr']:.1e}   {srec.get('tokens_per_s', 0):.0f} tok/s   device {meta['environment'].get('device')}",
                 fontsize=9.5, color=MUTED)

        # 1. loss
        ax = fig.add_subplot(gs[0, 0])
        style(ax, "Training loss per optimizer step")
        ax.plot(x[m], loss[m], color="#b4b2a9", linewidth=0.8)
        ax.plot(x[m], smooth[m], color="#2a78d6", linewidth=2)
        ax.set_xlim(x[0], max(last, 2))
        ax.set_ylim(0, max(loss.max() * 1.05, 0.1))
        ax.set_xlabel("step", fontsize=8, color=MUTED)

        # 2. validation
        ax = fig.add_subplot(gs[0, 1])
        style(ax, "Validation accuracy and macro-F1")
        ev = [e for e in evals if e["step"] <= cur]
        if ev:
            ax.plot([e["step"] for e in ev], [100 * e["accuracy"] for e in ev], "-o", color="#2a78d6", ms=4, lw=2,
                    label="accuracy")
            ax.plot([e["step"] for e in ev], [100 * e["macro_f1"] for e in ev], "--s", color="#eb6834", ms=4, lw=2,
                    label="macro-F1")
            ax.annotate(f"{100 * ev[-1]['accuracy']:.1f}%", (ev[-1]["step"], 100 * ev[-1]["accuracy"]),
                        textcoords="offset points", xytext=(-34, 8), fontsize=8, color=INK)
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        top = max([100 * e["accuracy"] for e in evals] + [10])
        ax.set_xlim(x[0], max(last, 2))
        ax.set_ylim(0, min(100, top * 1.2))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

        # 3. per-layer gradient heatmap
        ax = fig.add_subplot(gs[1, 0])
        style(ax, "Gradient norm per layer (log scale)")
        ax.grid(False)
        if G is not None and len(G):
            shown = G.copy()
            shown[gsteps > cur] = np.nan
            ax.imshow(np.log10(np.clip(shown.T, 1e-3, None)), aspect="auto", cmap="Blues", origin="lower",
                      extent=(gsteps[0], gsteps[-1] if len(gsteps) > 1 else gsteps[0] + 1, -0.5, len(keys) - 0.5),
                      vmin=np.log10(np.nanmin(G)), vmax=np.log10(np.nanmax(G)), interpolation="nearest")
            ticks = list(range(0, len(keys) - 1, 4)) + [len(keys) - 1]
            ax.set_yticks(ticks)
            ax.set_yticklabels([("head" if keys[t] == "head" else keys[t].replace("layer_", "L")) for t in ticks])
            ax.set_xlabel("step", fontsize=8, color=MUTED)

        # 4. embeddings
        ax = fig.add_subplot(gs[1, 1])
        style(ax, "Probe clause embeddings (PCA): clusters forming")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if snaps:
            pts, acc, snap_step = interp_snapshot(snaps, cur)
            g = snaps["group"]
            ax.scatter(pts[g < 0, 0], pts[g < 0, 1], s=7, c="#b4b2a9", alpha=0.45, linewidths=0)
            for i in range(len(HIGHLIGHT)):
                sel = g == i
                if sel.any():
                    ax.scatter(pts[sel, 0], pts[sel, 1], s=40, c=COLORS[i], marker=MARKERS[i], label=HIGHLIGHT[i],
                               edgecolors="white", linewidths=0.6)
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2, handletextpad=0.2, columnspacing=0.8)
            ax.text(0.99, 0.02, f"snapshot step {snap_step} · probe acc {100 * acc:.1f}%", transform=ax.transAxes,
                    ha="right", fontsize=8, color=MUTED)

        fig.canvas.draw()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        images.append(img.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        if fi == len(frame_steps) - 1:
            img.save(out.with_suffix(".png"))
    plt.close(fig)

    hold = [images[-1]] * (fps * 2)
    images[0].save(out, save_all=True, append_images=images[1:] + hold, duration=int(1000 / fps), loop=0,
                   optimize=True)
    print(f"Wrote {out} ({len(images)} frames, {out.stat().st_size / 1e6:.1f} MB) and {out.with_suffix('.png')}")


if __name__ == "__main__":
    app()
