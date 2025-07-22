"""
classify.py – heading detector + level mapping
"""
from __future__ import annotations
import re, numpy as np
from typing import List
from sklearn.cluster import KMeans
from .extract import Span
from .features import build_matrix

MODEL_COEF = np.array([2.1, 1.3, 0.4, -0.5, 1.7, 2.4], dtype=np.float32)
MODEL_INT  = -2.0

SECTION_RE = re.compile(r"^\d+(\.\d+)+\s")

def predict_headings(spans: List[Span]) -> List[Span]:
    if not spans:
        return []

    X = build_matrix(spans)
    logits = X @ MODEL_COEF + MODEL_INT
    probs  = 1 / (1 + np.exp(-logits))
    keep   = (probs >= 0.45) | (X[:, 0] >= 0.5)

    cand = [s for s, k in zip(spans, keep) if k]

    # force‑include section lines that ML missed
    for s, k in zip(spans, keep):
        if not k and SECTION_RE.match(s.text):
            cand.append(s)

    if not cand:
        return []

    sizes  = np.array([s.font_size for s in cand]).reshape(-1, 1)
    k      = min(4, np.unique(sizes).size)
    labels = KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(sizes)
    order  = np.argsort([-sizes[labels == i].mean() for i in range(k)])
    level_map = {cl: f"H{idx+1}" for idx, cl in enumerate(order)}

    for s, lab in zip(cand, labels):
        s.level = level_map[lab]

    # derive level for any forced ones
    for s in cand:
        if s.level is None:
            depth = s.text.split(" ")[0].count(".") + 1
            s.level = f"H{min(depth,6)}"

    return cand
