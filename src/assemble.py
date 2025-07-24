"""
assemble.py – build ordered outline stack
"""
from __future__ import annotations
from typing import List, Dict

from .extract import Span


def build_outline(headings: List[Span], page_cnt: int) -> List[Dict]:
    """
    Build an ordered outline.

    • ONE‑page PDFs → logical page **0**  
    • ≥ 2 pages     → physical cover 0, physical 1 → logical 1, …
    """
    headings.sort(key=lambda s: (s.page, s.bbox[1], s.bbox[0]))
    single_page = page_cnt == 1

    outline: List[Dict] = []
    for h in headings:
        logical = 0 if single_page else max(1, h.page - 1)
        outline.append(
            {
                "level": getattr(h, "level", "H3"),
                "text":  h.text.strip(),
                "page":  logical,
            }
        )
    return outline
