"""
utils.py – common helpers
"""
import re, unicodedata
from collections import Counter
from typing import List

WS_RE = re.compile(r"\s+")

DATE_RE = re.compile(r"\b\d{1,2}\s+[A-Z]{3,}\s+\d{4}\b")    # 18 JUNE 2013
DOT_LEADER_RE = re.compile(r"\.{4,}")                       # "......" in TOC

def norm(text: str) -> str:
    """Unicode‑normalise + compress whitespace + strip punctuation for comparisons."""
    text = unicodedata.normalize("NFKC", text)
    text = WS_RE.sub(" ", text).strip()
    return text.lower().strip("-–—:;,.")    # basic trailing punct trim


def build_header_footer_stopset(lines: List[str], page_count: int, ratio: float = 0.4):
    """Return texts that repeat on ≥ ratio of pages (probable headers/footers)."""
    counts = Counter(lines)
    cutoff = int(page_count * ratio)
    return {txt for txt, freq in counts.items() if freq >= cutoff and txt}


def looks_like_date(s: str) -> bool:
    return bool(DATE_RE.search(s))


def looks_like_dot_leader(s: str) -> bool:
    return bool(DOT_LEADER_RE.search(s))
