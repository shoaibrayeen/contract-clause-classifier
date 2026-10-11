"""Explainability for linear classical models: the word n-grams that push hardest towards each clause type."""

from __future__ import annotations

import numpy as np


def top_features(pipeline, label_names: list[str], k: int = 12) -> dict[str, list[dict]]:
    """{clause type: [{"term", "weight"}]} from the linear classifier's coefficients (word block only, readable)."""
    clf = pipeline.named_steps["clf"]
    coef = getattr(clf, "coef_", None)
    if coef is None:
        return {}
    features = pipeline.named_steps["features"]
    names = features.get_feature_names_out()
    word_mask = np.array([n.startswith("word__") for n in names])
    word_idx = np.where(word_mask)[0]
    terms = np.array([n.split("__", 1)[1] for n in names])
    out = {}
    for i, label in enumerate(label_names):
        row = coef[i, word_idx]
        best = word_idx[np.argsort(-row)[:k]]
        out[label] = [{"term": str(terms[j]), "weight": round(float(coef[i, j]), 3)} for j in best]
    return out
