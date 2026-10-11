"""Configuration for the classical (TF-IDF + linear model) clause classifier.

Loaded from `configs/classical/ledgar_tfidf_lr.yaml`. Independent of the transformer config: its own
`model_version`, run name and output directory, but the same dataset, splits and label groups.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import DataConfig


@dataclass
class NgramConfig:
    ngram_range: list[int] = field(default_factory=lambda: [1, 2])
    min_df: int = 2
    max_features: int | None = 300_000


@dataclass
class FeaturesConfig:
    word: NgramConfig = field(default_factory=NgramConfig)
    char: NgramConfig | None = field(default_factory=lambda: NgramConfig(ngram_range=[3, 5], min_df=3))
    sublinear_tf: bool = True
    lowercase: bool = True


@dataclass
class LogisticConfig:
    C_grid: list[float] = field(default_factory=lambda: [3.0, 10.0, 30.0])
    max_iter: int = 2000


@dataclass
class SvmConfig:
    enabled: bool = True
    C_grid: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    calibration_cv: int = 3


@dataclass
class NaiveBayesConfig:
    enabled: bool = True
    alpha_grid: list[float] = field(default_factory=lambda: [0.1, 0.3, 1.0])


@dataclass
class ClassicalConfig:
    model_version: int = 1
    run_name: str = "ledgar-tfidf-lr"
    output_dir: str = "artifacts/classical/runs"
    seed: int = 42
    data: DataConfig = field(default_factory=lambda: DataConfig(label_groups="configs/ledgar_label_groups.yaml"))
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    logistic_regression: LogisticConfig = field(default_factory=LogisticConfig)
    linear_svm: SvmConfig = field(default_factory=SvmConfig)
    complement_nb: NaiveBayesConfig = field(default_factory=NaiveBayesConfig)
    learning_curve: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc: Any, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(dc)}
    for key, val in (values or {}).items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {type(dc).__name__}")
        cur = getattr(dc, key)
        if isinstance(val, dict) and (is_dataclass(cur) or cur is None):
            if cur is None:  # optional sub-config (e.g. features.char) given in the YAML
                cur = NgramConfig()
                setattr(dc, key, cur)
            _merge(cur, val)
        else:
            setattr(dc, key, val)
    return dc


def load_classical_config(path: str | Path | None = None) -> ClassicalConfig:
    cfg = ClassicalConfig()
    if path:
        _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    return cfg
