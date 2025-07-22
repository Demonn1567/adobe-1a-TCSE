"""
assemble.py – build ordered outline stack
"""
from __future__ import annotations
from typing import List, Dict
from .extract import Span

def build_outline(headings: List[Span]) -> List[Dict]:
    headings.sort(key=lambda s: (s.page, s.bbox[1], s.bbox[0]))
    outline = []
    for h in headings:
        logical_page = max(1, h.page - 1)   # cover = page 0
        outline.append(
            {
                "level": getattr(h, "level", "H3"),
                "text": h.text.strip(),
                "page": logical_page,
            }
        )
    return outline
