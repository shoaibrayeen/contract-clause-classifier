"""Configuration loading and device / precision resolution.

Training is machine-independent: the same YAML runs on CUDA, Apple MPS or CPU.
`device: auto` picks cuda > mps > cpu at runtime.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# Must be set before torch initialises MPS so unsupported ops fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    data_dir: str = "data"
    dataset_repo: str = "coastalcph/lex_glue"
    dataset_config: str = "ledgar"
    dataset_revision: str = "refs/convert/parquet"
    max_len: int = 512
    # YAML mapping canonical clause type -> [original LEDGAR labels] (see configs/ledgar_label_groups.yaml).
    # None = train on the original 100 labels.
    label_groups: str | None = None


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class TrainConfig:
    run_name: str = "ledgar-qwen3-0.6b-lora"
    output_dir: str = "artifacts/runs"
    epochs: float = 2.0
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    eval_steps: int = 500
    eval_limit: int | None = None  # cap validation examples during training (None = full split)
    eval_batch_size: int = 32
    early_stopping_patience: int = 4  # evals without macro-F1 improvement
    log_steps: int = 10
    seed: int = 42
    class_weighted_loss: bool = False
    label_smoothing: float = 0.0  # 0.1 helps when some labels are near-synonyms
    best_metric: str = "accuracy"  # accuracy | macro_f1: picks checkpoint-best and drives early stopping
    gradient_checkpointing: bool = False
    limit: int | None = None  # use only N training examples (sanity runs)
    num_workers: int = 0
    empty_cache_steps: int = 25  # release cached accelerator memory every N steps (MPS fragments badly; 0 = off)


@dataclass
class TelemetryConfig:
    """What to record for later visualisation (see classification/telemetry.py)."""

    enabled: bool = True
    record_batches: bool = True     # example ids / labels / lengths / per-example loss for every step
    layer_stats_every: int = 10     # per-layer gradient norms every N optimizer steps (0 = off)
    probe_per_label: int = 5        # fixed validation probe set size per class (snapshots at each eval)
    embeddings: bool = True         # store pooled embeddings of the probe set at each eval
    layer_probe_per_label: int = 1  # of those, examples per class with full per-layer vectors ("node" data)
    system_interval_s: float = 5.0  # system monitor sampling period (CPU/RAM/accelerator memory)
    live_dashboard: bool = True     # rich live panel in the terminal while training


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B-Base"
    model_version: int = 1  # next version to train -> runs/<train.run_name>-v<N>; bumped after `make all`
    full_finetune: bool = False
    device: str = "auto"  # auto | cuda | mps | cpu
    precision: str = "auto"  # auto | bf16 | fp16 | fp32  (autocast dtype; weights stay fp32)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc: Any, values: dict[str, Any]) -> Any:
    """Recursively apply a dict onto a dataclass instance, rejecting unknown keys."""
    known = {f.name: f for f in fields(dc)}
    for key, val in (values or {}).items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {type(dc).__name__}")
        cur = getattr(dc, key)
        if is_dataclass(cur) and isinstance(val, dict):
            _merge(cur, val)
        else:
            setattr(dc, key, val)
    return dc


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path) as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        _merge(cfg, overrides)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Relative paths are resolved against the project root, not the CWD."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def display_path(p: str | Path) -> str:
    """Short path for console output: 'contract-clause-classifier/artifacts/runs/x' instead of '/Users/.../x'."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.parent.resolve()))
    except ValueError:
        return str(p)


def resolve_device(name: str = "auto") -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device=mps requested but MPS is not available")
    return torch.device(name)


def resolve_autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    """Return the autocast dtype, or None for plain fp32.

    auto: bf16 on CUDA (fp16 if bf16 unsupported); fp32 on MPS and CPU.
    On an M4 Pro, bf16 autocast measured no faster than fp32 for this model, so MPS
    stays fp32 for numerical safety. Force it with `precision: bf16` if you like.
    """
    precision = (precision or "auto").lower()
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None
