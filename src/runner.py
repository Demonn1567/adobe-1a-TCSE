"""
runner.py – main CLI (invite‑flyer override + robust cover‑title detection)
"""
from __future__ import annotations
import json, pathlib, re
from typing import List

import orjson
from jsonschema import validate

from .assemble  import build_outline
from .classify  import predict_headings
from .extract   import Span, extract_spans
from .features  import filter_spans
from .utils     import looks_like_date

SCHEMA = (
    pathlib.Path(__file__).parent.parent
    / "sample_dataset/schema/output_schema.json"
)

# ---------- helpers ------------------------------------------------------- #
CHAR_STUT   = re.compile(r"(\w)\s+\1\w?")
WORD_DUP    = re.compile(r"\b(\w{2,})(\s+\1\b)+", re.I)
WS_COLLAPSE = re.compile(r"\s{2,}")
NUM_FIELD   = re.compile(r"\b\d+\.")                 # “… 1.” pattern

# Make the default “cover band” a little deeper for tall covers
TOP_Y_LIMIT = 520
PAGE_HEIGHT = 792.0

# Tokens that strongly indicate a table/form row (avoid in subtitles)
_FORM_TOKENS = {
    "name","designation","date","service","pay","si","npa","hometown","home","town",
    "wife","husband","whether","entitled","block","place","amount","rs","s.no","sno",
    "age","relationship","fare","bus","rail","ticket","advance"
}

def _scrub_line(text: str) -> str:
    prev = None
    while prev != text:
        prev  = text
        text  = CHAR_STUT.sub(r"\1", text)
        text  = WORD_DUP.sub(r"\1", text)
        text  = WS_COLLAPSE.sub(" ", text)
    return text.strip()

def _dedup_tokens(lines: List[str]) -> str:
    kept, seen = [], set()
    for ln in lines:
        for tok in ln.split():
            low = tok.lower()
            if low not in seen and len(tok) > 1:
                kept.append(tok)
                seen.add(low)
    return " ".join(kept)

def _is_titlecase_like(s: str) -> bool:
    """Heuristic: looks like a heading (Title‑Case or ALL‑CAPS)."""
    txt = s.strip()
    if not txt:
        return False
    if txt.endswith("."):
        return False
    words = [w for w in re.split(r"\s+", txt) if w]
    if not words:
        return False
    # ALL‑CAPS?
    alpha = [c for c in txt if c.isalpha()]
    if alpha:
        caps_ratio = sum(c.isupper() for c in alpha) / len(alpha)
        if caps_ratio >= 0.85:
            return True
    # Title‑Case: majority of tokens start uppercase (ignore punctuation)
    starts_upper = 0
    for w in words:
        for ch in w:
            if ch.isalpha():
                if ch.isupper():
                    starts_upper += 1
                break
    return (starts_upper / len(words)) >= 0.60

def _looks_like_form_line(s: str) -> bool:
    """Reject common form/table cells as potential subtitles."""
    low = s.lower()
    if low.endswith(":"):
        return True
    if any(tok in low for tok in _FORM_TOKENS):
        return True
    if low.startswith(tuple(str(i) + "." for i in range(1, 21))):
        return True
    return False

def _collect_cover_band(spans: List[Span], y_limit: float) -> List[Span]:
    """Spans on p1 within the top-of-page band; fallback to whole p1 if empty."""
    band = [s for s in spans if s.page == 1 and s.bbox[1] < y_limit]
    if band:
        return sorted(band, key=lambda x: x.bbox[1])
    # Fallback: whole first page, still sorted by Y (for tall covers like the RFP)
    whole = [s for s in spans if s.page == 1]
    return sorted(whole, key=lambda x: x.bbox[1])

def detect_title(spans: List[Span]) -> str:
    """
    Cover title extraction:
      • take max‑font line(s) near top (or whole page if the band is empty),
      • consider a 240pt subtitle band below the first title line,
      • only accept subtitles that look like headings, not form/table rows.
    """
    p1 = _collect_cover_band(spans, TOP_Y_LIMIT)
    if not p1:
        return ""

    max_sz = max(s.font_size for s in p1)
    title_raw: List[str] = []

    # (1) absolute max‑size line(s) (e.g., “RFP: Request for Proposal”)
    for s in p1:
        if abs(s.font_size - max_sz) < 0.5:
            title_raw.append(_scrub_line(s.text))
        else:
            break

    # (2) subtitle within band
    base_y   = p1[0].bbox[1]
    GAP_MAX  = 240
    added    = 0
    for s in p1[len(title_raw):]:
        if s.bbox[1] - base_y > GAP_MAX or added == 2:
            break
        txt = s.text.strip()
        if (
            s.font_size >= 0.5 * max_sz and
            not NUM_FIELD.search(txt) and
            not looks_like_date(txt) and
            _is_titlecase_like(txt) and
            not _looks_like_form_line(txt) and
            4 <= len(txt.split()) <= 30
        ):
            title_raw.append(_scrub_line(txt))
            added += 1
        else:
            if s.font_size < 0.45 * max_sz:
                break

    clean = _dedup_tokens(title_raw).strip()
    if len(clean) <= 8:
        clean = " ".join(title_raw).strip()
    return clean

# -------- single‑page flyer helpers -------------------------------------- #
def _fallback_title(spans: List[Span]) -> str:
    return _scrub_line(max(spans, key=lambda s: s.font_size).text) if spans else ""

def _flyer_headings(spans: List[Span]) -> List[Span]:
    if not spans:
        return []
    top2 = sorted(spans, key=lambda s: -s.font_size)[:2]
    for s, lvl in zip(top2, ("H1", "H2")):
        s.level = lvl
    return sorted(top2, key=lambda s: (s.bbox[1], s.bbox[0]))

INVITE_KEYS = ("for:", "date:", "time:", "rsvp:", "address:")

def _is_invite_form(spans: List[Span]) -> bool:
    p1 = [s.text.strip().lower() for s in spans if s.page == 1]
    hits = sum(any(t.startswith(k) for k in INVITE_KEYS) for t in p1)
    return hits >= 2

def _pick_bottom_callout_heading(spans: List[Span]) -> Span | None:
    cand: List[Span] = []
    for s in spans:
        if s.page != 1:
            continue
        y_mid = (s.bbox[1] + s.bbox[3]) / 2.0
        words = len(s.text.split())
        alpha = [c for c in s.text if c.isalpha()]
        caps_ratio = (sum(c.isupper() for c in alpha) / len(alpha)) if alpha else 0.0
        txt = s.text.strip()
        if y_mid / PAGE_HEIGHT < 0.60:
            continue
        if "www." in txt.lower() or ".com" in txt.lower():
            continue
        if words <= 8 and (txt.endswith("!") or caps_ratio >= 0.6):
            cand.append(s)
    if not cand:
        return None
    best = max(cand, key=lambda s: (s.font_size, s.bbox[1]))
    best.level = "H1"
    return best

# ------------------------------------------------------------------------ #
def process(pdf: pathlib.Path, out_dir: pathlib.Path):
    spans        = extract_spans(pdf)
    page_cnt     = max(s.page for s in spans)

    title        = detect_title(spans)

    # Invitation flyer override
    invite = (page_cnt == 1) and _is_invite_form(spans)
    if invite:
        title = ""

    spans_flt    = filter_spans(spans, title, page_cnt)
    headings_raw = predict_headings(spans_flt)

    if invite:
        pick = _pick_bottom_callout_heading(spans)
        if pick:
            headings_raw = [pick]

    if page_cnt == 1 and not headings_raw:
        if not title:
            headings_raw = _flyer_headings(spans_flt)

    data = {"title": title, "outline": build_outline(headings_raw, page_cnt)}
    validate(instance=data, schema=json.loads(SCHEMA.read_text()))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf.stem}.json"
    out_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    print(f"✅  {pdf.name} → {out_path.relative_to(out_dir.parent)}")

def main():
    docker = "/app/" in str(pathlib.Path(__file__).resolve())
    in_dir  = pathlib.Path("/app/input")  if docker else pathlib.Path("sample_dataset/pdfs")
    out_dir = pathlib.Path("/app/output") if docker else pathlib.Path("sample_dataset/outputs")
    for pdf in sorted(in_dir.glob("*.pdf")):
        process(pdf, out_dir)

if __name__ == "__main__":
    main()
