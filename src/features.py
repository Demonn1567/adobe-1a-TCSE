"""
features.py – filter noisy spans & build feature matrix
"""
from __future__ import annotations
import re, numpy as np
from typing import List
from .utils import norm, build_header_footer_stopset, looks_like_date, looks_like_dot_leader
from .extract import Span

NUM_ONLY_RE = re.compile(r"^\d+\.$")
SECTION_RE  = re.compile(r"^\d+(\.\d+)+\s")

def filter_spans(spans: List[Span], title: str, page_cnt: int) -> List[Span]:
    stop = build_header_footer_stopset([norm(s.text) for s in spans], page_cnt)
    title_n = norm(title); clean: List[Span] = []
    for s in spans:
        txt_n = norm(s.text)
        if s.page == 1 or txt_n == title_n or txt_n in stop: continue
        if looks_like_date(s.text) or looks_like_dot_leader(s.text): continue
        if NUM_ONLY_RE.fullmatch(s.text.strip()): continue
        if len(s.text.split()) > 25 and not SECTION_RE.match(s.text): continue
        caps_ratio = sum(c.isupper() for c in s.text)/len(s.text)
        if caps_ratio > .9: continue
        clean.append(s)
    return clean

def build_matrix(spans: List[Span]) -> np.ndarray:
    if not spans: return np.empty((0,6), np.float32)
    feats=[]
    for s in spans:
        y_pct = s.bbox[1]/792
        caps  = sum(c.isupper() for c in s.text)/len(s.text)
        first = s.text.lstrip()[:8].split(" ")[0]
        num_pref = 1 if first.rstrip(".").replace(".","").isdigit() else 0
        feats.append([s.font_size, int(s.is_bold), int(s.is_italic), y_pct, caps, num_pref])
    X = np.asarray(feats,np.float32)
    X[:,0]=(X[:,0]-X[:,0].mean())/(X[:,0].std()+1e-6)
    return X
