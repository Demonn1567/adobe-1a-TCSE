"""
features.py – filter noisy spans & build numeric feature matrix
"""
from __future__ import annotations
import re, numpy as np
from typing import List
from .utils import (
    norm,
    build_header_footer_stopset,
    looks_like_date,
    looks_like_dot_leader,
)
from .extract import Span

NUM_ONLY_RE     = re.compile(r"^\d+\.$")              # bare "3."
SECTION_RE_ANY  = re.compile(r"\b\d+(\.\d+)+\s")      # "2.1 " anywhere

# --------------------------------------------------------------------------- #
def filter_spans(spans: List[Span], title: str, page_cnt: int) -> List[Span]:
    """Return spans worth sending to heading classifier."""
    stop    = build_header_footer_stopset([norm(s.text) for s in spans], page_cnt)
    title_n = norm(title)
    clean: List[Span] = []

    for s in spans:
        txt, txt_n = s.text, norm(s.text)

        # quick rejects
        if (
            s.page == 1
            or txt_n == title_n
            or txt_n in stop
            or looks_like_date(txt)
            or looks_like_dot_leader(txt)
            or NUM_ONLY_RE.fullmatch(txt.strip())
        ):
            continue

        # keep numbered headings even if embedded in bigger span
        m = SECTION_RE_ANY.search(txt)
        if m:
            if m.start():                # split tail containing the heading
                from copy import copy
                tail       = copy(s)
                tail.text  = txt[m.start():].lstrip()
                clean.append(tail)
            else:
                clean.append(s)
            continue

        # ditch long sentences or bullets that clearly aren’t headings
        words = txt.split()
        if len(words) > 25 or (txt.rstrip(".").endswith(".") and len(words) > 10):
            continue
        if sum(c.isupper() for c in txt) / len(txt) > 0.9:
            continue

        clean.append(s)

    return clean


# --------------------------------------------------------------------------- #
def build_matrix(spans: List[Span]) -> np.ndarray:
    if not spans:
        return np.empty((0, 6), np.float32)

    feats = []
    for s in spans:
        y_pct = s.bbox[1] / 792.0
        caps  = sum(c.isupper() for c in s.text) / len(s.text)
        first = s.text.lstrip()[:8].split(" ")[0]
        num_prefix = 1 if first.rstrip(".").replace(".", "").isdigit() else 0
        feats.append(
            [
                s.font_size,
                int(s.is_bold),
                int(s.is_italic),
                y_pct,
                caps,
                num_prefix,
            ]
        )

    X = np.asarray(feats, np.float32)
    X[:, 0] = (X[:, 0] - X[:, 0].mean()) / (X[:, 0].std() + 1e-6)
    return X
