"""TF-IDF features: word n-grams (phrases) plus character n-grams (spelling and morphology variants)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .config import FeaturesConfig


def build_features(cfg: FeaturesConfig) -> FeatureUnion:
    parts = [("word", TfidfVectorizer(
        analyzer="word", ngram_range=tuple(cfg.word.ngram_range), min_df=cfg.word.min_df,
        max_features=cfg.word.max_features, sublinear_tf=cfg.sublinear_tf, lowercase=cfg.lowercase,
        token_pattern=r"(?u)\b\w[\w'-]*\b", dtype=np.float64,
    ))]
    if cfg.char is not None:
        parts.append(("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=tuple(cfg.char.ngram_range), min_df=cfg.char.min_df,
            max_features=cfg.char.max_features, sublinear_tf=cfg.sublinear_tf, lowercase=cfg.lowercase, dtype=np.float64,
        )))
    return FeatureUnion(parts)


def describe(features: FeatureUnion) -> dict:
    """Feature counts per block, for model_info.json and the model page."""
    out = {}
    for name, vec in features.transformer_list:
        out[name] = {"analyzer": vec.analyzer, "ngram_range": list(vec.ngram_range), "min_df": vec.min_df,
                     "n_features": len(vec.vocabulary_)}
    out["total"] = sum(v["n_features"] for v in out.values())
    return out
