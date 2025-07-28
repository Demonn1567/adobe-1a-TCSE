from __future__ import annotations
from typing import List, Dict

from .extract import Span


def build_outline(headings: List[Span], page_cnt: int) -> List[Dict]:
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
