"""The live dashboard must always fit the terminal, otherwise rich leaves a copy of it on every refresh."""

import io

import pytest
from rich.console import Console

from classification.dashboard import TrainingDashboard

NAMES = [f"Clause type {i}" for i in range(72)]


def render(height: int, width: int = 140) -> tuple[list[str], TrainingDashboard]:
    con = Console(file=io.StringIO(), force_terminal=True, width=width, height=height)
    d = TrainingDashboard(con, total_steps=5625, start_step=0, label_names=NAMES,
                          run_dir="/tmp/x/artifacts/runs/ledgar-qwen3-0.6b-v1")
    for step in range(1, 30):
        d.on_step(step, {"loss": 1.0 / step, "lr": 2e-5, "grad_norm": 1.0, "epoch": step / 1875,
                         "tokens_per_s": 900, "step_time_s": 4.2, "padding_ratio": 0.01, "mem_gb": 22.0,
                         "batch_accuracy": 0.9, "clipped": False}, batch_labels=list(range(32)))
    for k in range(1, 13):
        d.on_eval(k * 500, {"accuracy": 0.9, "macro_f1": 0.8, "loss": 0.4}, improved=k % 2 == 0)
    d.on_system({"cpu_percent": 10, "ram_used_gb": 30, "ram_total_gb": 48})
    con.print(d)
    return con.file.getvalue().rstrip("\n").split("\n"), d


@pytest.mark.parametrize("height", [20, 24, 30, 40, 60])
def test_dashboard_fits_terminal(height):
    lines, _ = render(height)
    assert len(lines) < height, f"{len(lines)} lines rendered for a {height}-line terminal"


def test_dashboard_shows_ten_rows_when_there_is_room():
    lines, d = render(60)
    text = "\n".join(lines)
    assert d._rows_that_fit() == 10
    assert sum(f"Clause type {i} " in text for i in range(32)) == 10   # 10 clause types of the current batch
    assert " 6000 " in text and " 1500 " in text and " 1000 " not in text   # last 10 of 12 evaluations


def test_dashboard_uses_short_path():
    lines, d = render(40)
    assert d.run_dir == "/tmp/x/artifacts/runs/ledgar-qwen3-0.6b-v1"  # outside the project: unchanged
    from classification.config import PROJECT_ROOT, display_path

    assert display_path(PROJECT_ROOT / "artifacts" / "runs" / "v1") == f"{PROJECT_ROOT.name}/artifacts/runs/v1"
