"""
features.py – prune noisy spans & build numeric feature matrix
"""
from __future__ import annotations
import re
from typing import List

import numpy as np

from .utils import (
    norm,
    build_header_footer_stopset,
    looks_like_date,
    looks_like_dot_leader,
)
from .extract import Span

# --------------------------------------------------------------------------- #
NUM_ONLY_RE     = re.compile(r"^\d+(\.\d+)?$")        # 2007, 3, 3.0
LEAD_NUM_BULLET = re.compile(r"^\s*\d+\)")            # "1)"
SECTION_ANY_RE  = re.compile(r"\b\d+(\.\d+)+\s")      # 2.3


def _shorten_after_colon(txt: str) -> str:
    """Long descriptive lines → keep the part before first ':'."""
    if ":" in txt and len(txt.split()) > 5:
        return txt.split(":", 1)[0] + ":"
    return txt


# --------------------------------------------------------------------------- #
def filter_spans(spans: List[Span], title: str, page_cnt: int) -> List[Span]:
    """Return spans worth sending to the heading classifier."""
    stop    = build_header_footer_stopset([norm(s.text) for s in spans], page_cnt)
    title_n = norm(title)
    kept: List[Span] = []

    for s in spans:
        txt, n = s.text, norm(s.text)

        # ----- obvious skips ------------------------------------------------ #
        if (
            (page_cnt > 1 and s.page == 1)      # keep p 1 if a single‑pager
            or n == title_n
            or n in stop
            or looks_like_date(txt)
            or looks_like_dot_leader(txt)
            or NUM_ONLY_RE.fullmatch(txt.strip())
            or LEAD_NUM_BULLET.match(txt)
        ):
            continue

        # ---------- numbered headings (keep, maybe split) ------------------ #
        m = SECTION_ANY_RE.search(txt)
        if m:
            if m.start():
                from copy import copy
                tail      = copy(s)
                tail.text = _shorten_after_colon(txt[m.start():].lstrip())
                kept.append(tail)
            else:
                s.text = _shorten_after_colon(txt)
                kept.append(s)
            continue

        # ---------- heuristics for plain‑text headings --------------------- #
        tokens = txt.split()

        # full sentences → body
        if txt.strip().endswith(".") and len(tokens) > 8:
            continue

        # unguided blobs longer than 18 words are rarely headings
        if len(tokens) > 18:
            continue

        s.text = _shorten_after_colon(txt)
        kept.append(s)

    return kept


# --------------------------------------------------------------------------- #
def build_matrix(spans: List[Span]) -> np.ndarray:
    if not spans:
        return np.empty((0, 6), np.float32)

    feats = []
    for s in spans:
        y_pct = s.bbox[1] / 792.0
        caps  = sum(c.isupper() for c in s.text) / len(s.text)
        first = s.text.lstrip()[:8].split(" ")[0]
        num   = 1 if first.rstrip(".").replace(".", "").isdigit() else 0
        feats.append(
            [s.font_size, int(s.is_bold), int(s.is_italic), y_pct, caps, num]
        )

    X = np.asarray(feats, np.float32)
    X[:, 0] = (X[:, 0] - X[:, 0].mean()) / (X[:, 0].std() + 1e-6)
    return X
