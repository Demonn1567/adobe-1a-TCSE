import re, numpy as np
from typing import List
from sklearn.cluster import KMeans
from .extract   import Span
from .features  import build_matrix

MODEL_COEF = np.array([2.1, 1.3, 0.4, -0.5, 1.7, 2.4], dtype=np.float32)
MODEL_INT  = -2.0

SECTION_ANY = re.compile(r"\b\d+(\.\d+)+\s")   # “2.1 ” anywhere

def predict_headings(spans: List[Span]) -> List[Span]:
    if not spans:
        return []

    X       = build_matrix(spans)
    logits  = X @ MODEL_COEF + MODEL_INT
    probs   = 1 / (1 + np.exp(-logits))
    keep_ml = (probs >= 0.45) | (X[:, 0] >= 0.5)

    cand = [s for s, k in zip(spans, keep_ml) if k]

    # -------- force‑include any span containing section number ------------
    for s, k in zip(spans, keep_ml):
        if not k and SECTION_ANY.search(s.text):
            cand.append(s)
    # ----------------------------------------------------------------------

    if not cand:
        return []

    sizes = np.array([s.font_size for s in cand]).reshape(-1, 1)
    k     = min(4, np.unique(sizes).size)
    labels = KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(sizes)
    order  = np.argsort([-sizes[labels == i].mean() for i in range(k)])
    level_map = {cl: f"H{idx+1}" for idx, cl in enumerate(order)}

    for s, lab in zip(cand, labels):
        s.level = level_map[lab]

    # derive level for any forced spans that still lack one
    for s in cand:
        if s.level is None:
            depth = s.text.split(" ")[0].count(".") + 1
            s.level = f"H{min(depth, 6)}"

    return cand
