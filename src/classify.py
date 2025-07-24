"""
classify.py – predict heading levels from feature matrix
"""
from __future__ import annotations
import re
from typing import List

import numpy as np
from sklearn.cluster import KMeans

from .extract import Span
from .features import build_matrix

# --------------------------------------------------------------------------- #
MODEL_COEF = np.array([2.1, 1.3, 0.4, -0.5, 1.7, 2.4], dtype=np.float32)
MODEL_INT  = -2.0

SECTION_ANY = re.compile(r"\b\d+(\.\d+)+\s")   # “2.1 ” anywhere


def predict_headings(spans: List[Span]) -> List[Span]:
    """Return spans labelled with .level (H1…H6)."""
    if not spans:
        return []

    X       = build_matrix(spans)
    logits  = X @ MODEL_COEF + MODEL_INT
    probs   = 1 / (1 + np.exp(-logits))
    keep_ml = (probs >= 0.45) | (X[:, 0] >= 0.5)     # big font guard‑rail

    cand = [s for s, k in zip(spans, keep_ml) if k]

    # -------- force‑include any span containing a numbered section -------- #
    for s, k in zip(spans, keep_ml):
        if not k and SECTION_ANY.search(s.text):
            cand.append(s)
    # ---------------------------------------------------------------------- #

    if not cand:
        return []

    sizes  = np.array([s.font_size for s in cand]).reshape(-1, 1)
    k      = min(4, np.unique(sizes).size)
    labels = KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(sizes)

    means = [-sizes[labels == i].mean() for i in range(k)]
    order = np.argsort(means)

    level_map: dict[int, str] = {}
    h1_mean = None
    for idx, cl in enumerate(order):
        if idx == 0:
            h1_mean = means[cl]
        if abs(means[cl] - h1_mean) < 0.2:
            level_map[cl] = "H1"
        else:
            level_map[cl] = f"H{idx + 1}"

    for s, lab in zip(cand, labels):
        s.level = level_map[lab]

    # dotted‑number fallback
    for s in cand:
        if s.level is None:
            depth = s.text.split(" ")[0].count(".") + 1
            s.level = f"H{min(depth, 6)}"

    return cand
