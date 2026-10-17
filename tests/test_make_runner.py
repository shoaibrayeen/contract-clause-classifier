"""The Makefile runs every step in Docker by default and with uv on RUNNER=local; transformer training in Docker
needs a GPU. Checked with `make -n` (dry run), so nothing is executed and Docker is not needed."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def dry_run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["make", "-n", "--no-print-directory", *args], cwd=ROOT, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/tmp"})


def test_docker_is_the_default_runner():
    out = dry_run("data", "classical-train", "test").stdout
    assert "docker compose --progress quiet run --rm tools python scripts/01_prepare_data.py" in out
    assert "docker compose --progress quiet run --rm tools python scripts/classical/01_train.py" in out
    assert "docker compose --progress quiet run --rm tools pytest -q" in out
    assert "uv run" not in out


def test_local_runner_uses_uv():
    out = dry_run("data", "test", "RUNNER=local").stdout
    assert "uv run python scripts/01_prepare_data.py" in out
    assert "uv run pytest -q" in out
    assert "docker" not in out


def test_classical_all_runs_in_docker():
    out = dry_run("all", "FAMILY=classical").stdout
    for script in ("01_prepare_data.py", "classical/01_train.py", "classical/02_evaluate.py", "10_model_pages.py"):
        assert f"docker compose --progress quiet run --rm tools python scripts/{script}" in out
    assert "Transformer training in Docker" not in out


def test_transformer_training_in_docker_needs_a_gpu():
    for target in ("train", "sanity", "all", "resume", "restart"):
        out = dry_run(target).stdout
        assert "Transformer training in Docker needs an NVIDIA GPU" in out, target
        assert "RUNNER=local" in out and "exit 1" in out, target


def test_gpu_and_local_training_skip_the_guard():
    gpu = dry_run("train", "GPU=1").stdout
    assert "docker compose --progress quiet run --rm tools-gpu python scripts/02_train.py" in gpu
    assert "exit 1" not in gpu
    local = dry_run("train", "RUNNER=local").stdout
    assert "uv run python scripts/02_train.py" in local and "exit 1" not in local
    forced = dry_run("train", "ALLOW_CPU_TRAIN=1").stdout
    assert "docker compose --progress quiet run --rm tools python scripts/02_train.py" in forced and "exit 1" not in forced


def test_serve_uses_the_api_container():
    assert "docker compose up --build api" in dry_run("serve").stdout
    assert "uv run python scripts/05_serve.py" in dry_run("serve", "RUNNER=local").stdout


def test_unknown_runner_is_rejected():
    result = dry_run("data", "RUNNER=podman")
    assert result.returncode != 0 and "RUNNER must be docker or local" in result.stderr
