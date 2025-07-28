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

MODEL_COEF = np.array([2.1, 1.3, 0.4, -0.5, 1.7, 2.4], dtype=np.float32)
MODEL_INT  = -2.0

SECTION_ANY = re.compile(r"\b\d+(\.\d+)+\s")   # “2.1 ” anywhere

def predict_headings(spans: List[Span]) -> List[Span]:
    if not spans:
        return []

    X       = build_matrix(spans)
    logits  = X @ MODEL_COEF + MODEL_INT
    probs   = 1 / (1 + np.exp(-logits))
    keep_ml = (probs >= 0.45) | (X[:, 0] >= 0.5)     # big font guard‑rail

    cand = [s for s, k in zip(spans, keep_ml) if k]

    # force‑include any span containing a numbered section
    for s, k in zip(spans, keep_ml):
        if not k and SECTION_ANY.search(s.text):
            cand.append(s)

    if not cand:
        return []

    # Cluster by raw font sizes
    fs     = np.array([float(s.font_size) for s in cand]).reshape(-1, 1)
    k      = min(4, np.unique(fs).size)
    labels = KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(fs)

    # Use raw means (pt). Anything within ~12% OR <=1.2pt of max is H1.
    mu     = [float(fs[labels == i].mean()) for i in range(k)]
    max_mu = max(mu)

    level_map = {}
    for i in range(k):
        if (mu[i] >= 0.88 * max_mu) or (abs(mu[i] - max_mu) <= 1.2):
            level_map[i] = "H1"
        else:
            # order remaining by descending size
            rank = 2 + sum(m > mu[i] for m in mu if m < 0.88 * max_mu)
            level_map[i] = f"H{min(rank, 6)}"

    for s, lab in zip(cand, labels):
        if getattr(s, "level", None) is None:
            s.level = level_map[lab]

    # dotted‑number fallback
    for s in cand:
        if s.level is None:
            depth = s.text.split(" ")[0].count(".") + 1
            s.level = f"H{min(depth, 6)}"

    return cand
