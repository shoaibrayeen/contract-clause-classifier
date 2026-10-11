"""Replaying a run's terminal dashboard from telemetry, without a model, GPU or browser."""

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("render_terminal_gif", ROOT / "scripts" / "09_render_terminal_gif.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_terminal_gif"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_run(tmp_path, total=40, per_epoch=20):
    run = tmp_path / "toy-v1"
    tel = run / "telemetry"
    tel.mkdir(parents=True)
    names = [f"Clause {i}" for i in range(5)]
    (tel / "meta.json").write_text(json.dumps({"label_names": names,
                                               "model": {"total_steps": total, "steps_per_epoch": per_epoch}}))
    t0 = 1_000_000.0
    steps = [{"step": s, "epoch": s / per_epoch, "loss": 3.0 / s, "lr": 1e-5, "grad_norm": 1.0, "clipped": False,
              "step_time_s": 5.0, "tokens_per_s": 900.0, "padding_ratio": 0.01, "mem_gb": 25.0,
              "batch_accuracy": 0.5, "labels": [s % 5, (s + 1) % 5], "time": t0 + 5 * s} for s in range(1, total + 1)]
    (tel / "steps.jsonl").write_text("\n".join(json.dumps(x) for x in steps) + "\n")
    evals = [{"step": s, "accuracy": 0.8 + s / 1000, "top3_accuracy": 0.95, "macro_f1": 0.7, "loss": 0.5,
              "time": t0 + 5 * s} for s in (10, 20, 30, 40)]
    (tel / "evals.jsonl").write_text("\n".join(json.dumps(x) for x in evals) + "\n")
    (tel / "system.jsonl").write_text(json.dumps({"time": t0, "cpu_percent": 5, "ram_used_gb": 30, "ram_total_gb": 48}) + "\n")
    return run


def test_terminal_replay_holds_each_epoch(tmp_path):
    mod = _module()
    run = _fake_run(tmp_path)
    frames = mod.build_frames(run, n_frames=8, fps=6, renderer=mod.TerminalRenderer(cols=110, rows=30))
    holds = [ms for _, ms in frames if ms >= 2000]
    assert len(holds) == 2                       # end of epoch 1 and of epoch 2 (the last frame)
    out = tmp_path / "toy_terminal.gif"
    mod.save_gif(frames, out, width=600)
    gif = Image.open(out)
    assert gif.n_frames == len(frames) and gif.size[0] == 600
    durations = [f.info["duration"] for f in ImageSequence.Iterator(gif)]
    assert max(durations) >= 2000
