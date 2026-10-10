"""Live terminal dashboard for training (rich). Falls back to plain log lines when not a TTY."""

from __future__ import annotations

from collections import Counter, deque

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values, width: int = 40) -> str:
    vals = list(values)[-width:]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(SPARK[min(len(SPARK) - 1, int((v - lo) / span * (len(SPARK) - 1)))] for v in vals)


class TrainingDashboard:
    def __init__(self, console: Console, total_steps: int, start_step: int, label_names: list[str], run_dir: str):
        self.console = console
        self.interactive = console.is_terminal
        self.label_names = label_names
        self.run_dir = run_dir
        self.progress = Progress(
            TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(),
            TextColumn("{task.fields[status]}"), TimeElapsedColumn(), TimeRemainingColumn(), console=console,
        )
        self.task = self.progress.add_task("train", total=total_steps, completed=start_step, status="")
        self.losses: deque[float] = deque(maxlen=200)
        self.ema: float | None = None
        self.stats: dict = {}
        self.system: dict = {}
        self.evals: list[dict] = []
        self.batch_labels: Counter = Counter()
        self.phase = "train"

    # ------------------------------------------------------------------ updates
    def on_step(self, step: int, record: dict, batch_labels: list[int]) -> None:
        loss = record["loss"]
        self.losses.append(loss)
        self.ema = loss if self.ema is None else 0.95 * self.ema + 0.05 * loss
        self.stats = record
        self.batch_labels = Counter(batch_labels)
        self.progress.update(self.task, completed=step, status=f"loss {loss:.4f}  ema {self.ema:.4f}")

    def on_eval(self, step: int, metrics: dict, improved: bool) -> None:
        self.evals.append({"step": step, **metrics, "improved": improved})

    def on_system(self, sample: dict) -> None:
        self.system = sample or {}

    def plain_line(self, step: int) -> str:
        s, sy = self.stats, self.system
        return (
            f"step {step} epoch {s.get('epoch', 0):.3f} loss {s.get('loss', 0):.4f} ema {self.ema or 0:.4f} "
            f"lr {s.get('lr', 0):.2e} gnorm {s.get('grad_norm', 0):.3f} tok/s {s.get('tokens_per_s', 0):.0f} "
            f"pad {s.get('padding_ratio', 0):.2f} mem {s.get('mem_gb', 0):.1f}GB cpu {sy.get('cpu_percent', 0):.0f}% "
            f"ram {sy.get('ram_percent', 0):.0f}%"
        )

    # ------------------------------------------------------------------ render
    def _stats_table(self) -> Table:
        s, sy = self.stats, self.system
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold cyan")
        t.add_column()
        t.add_column(style="bold cyan")
        t.add_column()
        t.add_row("epoch", f"{s.get('epoch', 0):.3f}", "lr", f"{s.get('lr', 0):.2e}")
        t.add_row("loss", f"{s.get('loss', 0):.4f}", "ema loss", f"{self.ema or 0:.4f}")
        t.add_row("grad norm", f"{s.get('grad_norm', 0):.3f}{' (clipped)' if s.get('clipped') else ''}",
                  "batch acc", f"{s.get('batch_accuracy', 0):.3f}")
        t.add_row("tokens/s", f"{s.get('tokens_per_s', 0):.0f}", "step time", f"{s.get('step_time_s', 0):.2f}s")
        t.add_row("padding", f"{s.get('padding_ratio', 0):.1%}", "accel mem", f"{s.get('mem_gb', 0):.1f} GB")
        t.add_row("cpu", f"{sy.get('cpu_percent', 0):.0f}%", "ram",
                  f"{sy.get('ram_used_gb', 0):.1f}/{sy.get('ram_total_gb', 0):.0f} GB")
        t.add_row("loss trend", sparkline(self.losses), "", "")
        return t

    def _batch_table(self) -> Table:
        t = Table(title="Current batch: clause types flowing in", show_header=True, header_style="bold magenta",
                  expand=True)
        t.add_column("label")
        t.add_column("n", justify="right")
        for lab, n in self.batch_labels.most_common(6):
            t.add_row(self.label_names[lab] if lab < len(self.label_names) else str(lab), str(n))
        return t

    def _eval_table(self) -> Table:
        t = Table(title="Validation", show_header=True, header_style="bold green", expand=True)
        for col in ("step", "acc", "macro-F1", "loss", ""):
            t.add_column(col, justify="right")
        for e in self.evals[-5:]:
            t.add_row(str(e["step"]), f"{e['accuracy']:.4f}", f"{e['macro_f1']:.4f}", f"{e['loss']:.4f}",
                      "[green]best[/]" if e["improved"] else "")
        return t

    def __rich__(self):
        body = Table.grid(expand=True)
        body.add_column(ratio=1)
        body.add_column(ratio=1)
        body.add_row(self._batch_table(), self._eval_table())
        return Panel(
            Group(self.progress, self._stats_table(), body),
            title=f"[bold]Training[/] · {self.run_dir}", subtitle="telemetry -> <run>/telemetry/",
        )
