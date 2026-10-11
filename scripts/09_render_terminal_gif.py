"""Replay a finished (or running) training run's terminal dashboard as a GIF, epoch by epoch.

Feeds the run's telemetry (steps, evaluations, system samples) back into the same live dashboard that was
shown in the terminal during training, with a simulated clock so elapsed time and ETA match the real run.
Each frame is drawn as a terminal window (colours, box drawing) with a caption giving the epoch and the
latest validation accuracy; every epoch boundary is held so the per-epoch progress is easy to follow.

    uv run python scripts/09_render_terminal_gif.py                          # latest model -> demo/<run>_terminal.gif
    uv run python scripts/09_render_terminal_gif.py --run ledgar-qwen3-0.6b-v1 --frames 120

No GPU, browser or model needed; only the run's telemetry folder.
"""

from __future__ import annotations

import io
import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Optional

import typer
from PIL import Image, ImageDraw, ImageFont
from rich.cells import cell_len
from rich.console import Console
from rich.terminal_theme import MONOKAI

from classification.config import PROJECT_ROOT, display_path
from classification.dashboard import TrainingDashboard

app = typer.Typer(add_completion=False)
THEME = MONOKAI
COLS, ROWS = 132, 30
FONT_SIZE = 15
PAD, TITLE_H, CAPTION_H = 14, 30, 46


# ----------------------------------------------------------------------------- telemetry

def read_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def resolve_run(run: str, runs_root: Path) -> Path:
    if Path(run).exists():
        return Path(run)
    if (runs_root / run).exists():
        return runs_root / run
    from classification.registry import RegistryError, list_models, resolve_model

    models = list_models(runs_root)
    if run == "latest" and not models:
        candidates = [p.parents[1] for p in runs_root.glob("*/telemetry/steps.jsonl")]
        if not candidates:
            raise typer.BadParameter(f"No runs with telemetry under {runs_root}")
        return max(candidates, key=lambda p: (p / "telemetry" / "steps.jsonl").stat().st_mtime)
    try:
        return resolve_model(models, run).run_dir
    except RegistryError as exc:
        raise typer.BadParameter(str(exc)) from exc


# ----------------------------------------------------------------------------- terminal renderer

class TerminalRenderer:
    """Draw rich renderables as a terminal-window image (cell grid, truecolor, bold)."""

    def __init__(self, cols: int = COLS, rows: int = ROWS, size: int = FONT_SIZE):
        self.cols, self.rows = cols, rows
        self.regular, self.bold = self._fonts(size)
        self.cw = round(self.regular.getlength("M"))
        self.ch = round(size * 1.32)
        self.console = Console(file=io.StringIO(), width=cols, height=rows, force_terminal=True,
                               color_system="truecolor", legacy_windows=False)
        self.cap_font = self._ui_font(17)

    @staticmethod
    def _fonts(size: int):
        menlo = "/System/Library/Fonts/Menlo.ttc"
        if Path(menlo).exists():
            return ImageFont.truetype(menlo, size, index=0), ImageFont.truetype(menlo, size, index=1)
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",):
            if Path(path).exists():
                bold = path.replace(".ttf", "-Bold.ttf")
                return ImageFont.truetype(path, size), ImageFont.truetype(bold if Path(bold).exists() else path, size)
        f = ImageFont.load_default()
        return f, f

    @staticmethod
    def _ui_font(size: int):
        for path in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if Path(path).exists():
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    def _rgb(self, color, fg: bool) -> tuple[int, int, int]:
        if color is None or color.is_default:
            c = THEME.foreground_color if fg else THEME.background_color
        else:
            c = color.get_truecolor(THEME, foreground=fg)
        return (c.red, c.green, c.blue)

    def render(self, renderable, title: str, caption: str, accent: bool = False) -> Image.Image:
        lines = self.console.render_lines(renderable, self.console.options, pad=False)
        bg = (THEME.background_color.red, THEME.background_color.green, THEME.background_color.blue)
        width = PAD * 2 + self.cols * self.cw
        height = TITLE_H + PAD * 2 + len(lines) * self.ch + CAPTION_H
        img = Image.new("RGB", (width, height), bg)
        d = ImageDraw.Draw(img)
        # window title bar
        d.rectangle((0, 0, width, TITLE_H), fill=(40, 40, 40))
        for i, col in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
            d.ellipse((14 + i * 20, 9, 26 + i * 20, 21), fill=col)
        d.text((width / 2, TITLE_H / 2), title, font=self.cap_font, fill=(190, 190, 190), anchor="mm")
        # cells
        y = TITLE_H + PAD
        for line in lines:
            x_cell = 0
            for seg in line:
                if not seg.text or seg.control:
                    continue
                style = seg.style
                fg = self._rgb(style.color if style else None, True)
                if style and style.bgcolor and not style.bgcolor.is_default:
                    bgc = self._rgb(style.bgcolor, False)
                    d.rectangle((PAD + x_cell * self.cw, y, PAD + (x_cell + cell_len(seg.text)) * self.cw, y + self.ch), fill=bgc)
                if style and style.dim:
                    fg = tuple(int(0.55 * a + 0.45 * b) for a, b in zip(fg, bg))
                font = self.bold if (style and style.bold) else self.regular
                for ch in seg.text:  # per cell, so every glyph stays on the grid
                    if ch != " ":
                        d.text((PAD + x_cell * self.cw, y), ch, font=font, fill=fg)
                    x_cell += cell_len(ch)
            y += self.ch
        # caption strip
        top = height - CAPTION_H
        d.rectangle((0, top, width, height), fill=(42, 120, 214) if accent else (28, 28, 27))
        d.text((PAD + 4, top + CAPTION_H / 2), caption, font=self.cap_font, fill="white", anchor="lm")
        return img


# ----------------------------------------------------------------------------- replay

def build_frames(run_dir: Path, n_frames: int, fps: int, renderer: TerminalRenderer) -> list[tuple[Image.Image, int]]:
    tel = run_dir / "telemetry"
    meta = json.loads((tel / "meta.json").read_text())
    steps = read_jsonl(tel / "steps.jsonl")
    evals = read_jsonl(tel / "evals.jsonl")
    system = read_jsonl(tel / "system.jsonl")
    if not steps:
        raise typer.BadParameter(f"No steps recorded in {tel}")
    names = meta["label_names"]
    total = int(meta.get("model", {}).get("total_steps") or steps[-1]["step"])
    steps_per_epoch = int(meta.get("model", {}).get("steps_per_epoch") or total)
    epochs = max(1, math.ceil(total / steps_per_epoch))
    last = steps[-1]["step"]

    clock = [steps[0]["time"] - steps[0].get("step_time_s", 0)]
    dash = TrainingDashboard(renderer.console, total, 0, names, str(run_dir), get_time=lambda: clock[0])
    dash.progress.start_task(dash.task)
    sys_times = [s["time"] for s in system]

    # frame schedule: dense at the start (fast loss drop), even afterwards, plus every epoch boundary
    targets = set(range(1, min(last, 200) + 1, 25))
    targets |= {round(x) for x in [i * last / n_frames for i in range(1, n_frames + 1)]}
    boundaries = {e * steps_per_epoch for e in range(1, epochs + 1) if e * steps_per_epoch <= last}
    targets |= boundaries | {last}
    targets = sorted(t for t in targets if 1 <= t <= last)

    frames: list[tuple[Image.Image, int]] = []
    fed, ev_fed = 0, 0
    title = f"make all  —  {display_path(PROJECT_ROOT)}  —  training {run_dir.name}"
    base_ms = int(1000 / fps)
    for target in targets:
        while fed < len(steps) and steps[fed]["step"] <= target:
            rec = steps[fed]
            clock[0] = rec["time"]
            dash.on_step(rec["step"], rec, rec.get("labels", []))
            fed += 1
        while ev_fed < len(evals) and evals[ev_fed]["step"] <= target:
            e = evals[ev_fed]
            best = max([x["accuracy"] for x in evals[:ev_fed]] or [-1])
            dash.on_eval(e["step"], e, improved=e["accuracy"] > best)
            ev_fed += 1
        if system:
            i = max(0, bisect_right(sys_times, clock[0]) - 1)
            dash.on_system(system[i])
        cur = steps[fed - 1]
        epoch_num = min(epochs, int((cur["step"] - 1) // steps_per_epoch) + 1)
        last_eval = evals[ev_fed - 1] if ev_fed else None
        val = (f"  ·  validation accuracy {100 * last_eval['accuracy']:.1f}%  ·  top-3 "
               f"{100 * last_eval.get('top3_accuracy', 0):.1f}%  (step {last_eval['step']})") if last_eval else ""
        at_boundary = target in boundaries
        if at_boundary:
            done = target // steps_per_epoch
            caption = f"Epoch {done} of {epochs} complete  ·  step {target:,}/{total:,}{val}"
        else:
            caption = f"Epoch {epoch_num} of {epochs}  ·  step {cur['step']:,}/{total:,}  ·  loss {cur['loss']:.3f}{val}"
        img = renderer.render(dash, title, caption, accent=at_boundary)
        hold = 2600 if at_boundary else base_ms
        frames.append((img, 3800 if target == last else hold))
    return frames


def save_gif(frames: list[tuple[Image.Image, int]], out: Path, width: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    scaled = []
    for img, ms in frames:
        h = int(img.height * width / img.width)
        small = img.resize((width, h), Image.LANCZOS)
        scaled.append((small.quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE), ms))
    scaled[0][0].save(out, save_all=True, append_images=[f for f, _ in scaled[1:]],
                      duration=[ms for _, ms in scaled], loop=0, optimize=True, disposal=1)
    frames[-1][0].save(out.with_name(out.stem + "_final_frame.png"))


@app.command()
def main(
    run: str = typer.Option("latest", help="Run directory, model id (e.g. ledgar-qwen3-0.6b-v1) or 'latest'"),
    frames: int = typer.Option(60, help="Evenly spaced frames across the run (epoch boundaries are always added)"),
    fps: int = typer.Option(6, help="Playback speed for ordinary frames"),
    width: int = typer.Option(1000, help="GIF width in pixels"),
    out: Optional[Path] = typer.Option(None, help="Output GIF (default demo/<run>_terminal.gif)"),
) -> None:
    run_dir = resolve_run(run, PROJECT_ROOT / "artifacts" / "runs")
    renderer = TerminalRenderer()
    frame_list = build_frames(run_dir, frames, fps, renderer)
    out = out or PROJECT_ROOT / "demo" / f"{run_dir.name}_terminal.gif"
    save_gif(frame_list, out, width)
    print(f"Wrote {display_path(out)} ({len(frame_list)} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    app()
