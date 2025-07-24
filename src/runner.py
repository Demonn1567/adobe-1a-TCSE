"""
runner.py – main CLI   (patch v2: final title fixes for files 01 & 03)
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

SCHEMA = (
    pathlib.Path(__file__).parent.parent
    / "sample_dataset/schema/output_schema.json"
)

# ---------- helpers ------------------------------------------------------- #
CHAR_STUT   = re.compile(r"(\w)\s+\1\w?")           # Re equest → Request
WORD_DUP    = re.compile(r"\b(\w{2,})(\s+\1\b)+", re.I)
WS_COLLAPSE = re.compile(r"\s{2,}")
NUM_FIELD   = re.compile(r"\b\d+\.")                # “… 1.” pattern

TOP_Y_LIMIT = 300                                   # ignore footer banners


def _scrub_line(text: str) -> str:
    prev = None
    while prev != text:
        prev  = text
        text  = CHAR_STUT.sub(r"\1", text)
        text  = WORD_DUP.sub(r"\1", text)
        text  = WS_COLLAPSE.sub(" ", text)
    return text.strip()


def _dedup_tokens(line_list: List[str]) -> str:
    kept, seen = [], set()
    for ln in line_list:
        for tok in ln.split():
            low = tok.lower()
            if low not in seen and len(tok) > 1:
                kept.append(tok)
                seen.add(low)
    return " ".join(kept)


# ------------------------------------------------------------------------ #
def detect_title(spans: List[Span]) -> str:
    """
    Heuristic title extraction (works for all 5 PDFs):
      • collect max‑font lines on p1,
      • optionally add ≤2 subtitle lines (≤15 tokens, ≥60 % font, not numbered),
      • scrub & dedup,
      • if result still tiny -> keep raw concat,
      • if still empty -> fall back to first H1.
    """
    p1 = [s for s in spans if s.page == 1 and s.bbox[1] < TOP_Y_LIMIT]
    if not p1:
        return ""

    p1.sort(key=lambda s: s.bbox[1])
    max_sz = max(s.font_size for s in p1)

    title_raw: List[str] = []

    # (1) absolute max‑size lines
    for s in p1:
        if abs(s.font_size - max_sz) < 0.5:
            title_raw.append(_scrub_line(s.text))
        else:
            break

    # (2) optional subtitle lines (≤15 tokens, ≥60 % font, not numbered “1.”)
    base_y = p1[0].bbox[1]
    added  = 0
    for s in p1[len(title_raw):]:
        if s.bbox[1] - base_y > 60 or added == 2:
            break
        if (
            s.font_size >= 0.6 * max_sz
            and len(s.text.split()) <= 15
            and not NUM_FIELD.search(s.text)           # ← NEW guard (file 01)
        ):
            title_raw.append(_scrub_line(s.text))
            added += 1
        else:
            break

    clean = _dedup_tokens(title_raw)

    # If dedup made it too short (e.g., just “RFP:”) keep the raw concat instead
    if len(clean) <= 8:
        clean = " ".join(title_raw).strip()

    # Absolute last‑ditch: first H1 if still empty
    if not clean:
        clean = next((s.text for s in spans if getattr(s, "level", "") == "H1"), "")

    return clean


# ------------- single‑page flyer helpers (unchanged) ---------------------- #
def _fallback_title(spans: List[Span]) -> str:
    return _scrub_line(max(spans, key=lambda s: s.font_size).text) if spans else ""


def _flyer_headings(spans: List[Span]) -> List[Span]:
    if not spans:
        return []
    top2 = sorted(spans, key=lambda s: -s.font_size)[:2]
    for s, lvl in zip(top2, ("H1", "H2")):
        s.level = lvl
    return sorted(top2, key=lambda s: (s.bbox[1], s.bbox[0]))


# ------------------------------------------------------------------------ #
def process(pdf: pathlib.Path, out_dir: pathlib.Path):
    spans        = extract_spans(pdf)
    page_cnt     = max(s.page for s in spans)

    title        = detect_title(spans)
    spans_flt    = filter_spans(spans, title, page_cnt)
    headings_raw = predict_headings(spans_flt)

    if page_cnt == 1:                                # single‑page guardrails
        if not title:
            title = _fallback_title(spans)
        if not headings_raw:
            headings_raw = _flyer_headings(spans_flt)

    data = {"title": title, "outline": build_outline(headings_raw, page_cnt)}
    validate(instance=data, schema=json.loads(SCHEMA.read_text()))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf.stem}.json"
    out_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    print(f"✅  {pdf.name} → {out_path.relative_to(out_dir.parent)}")


# ------------------------------------------------------------------------ #
def main():
    docker = "/app/" in str(pathlib.Path(__file__).resolve())
    in_dir  = pathlib.Path("/app/input")  if docker else pathlib.Path("sample_dataset/pdfs")
    out_dir = pathlib.Path("/app/output") if docker else pathlib.Path("sample_dataset/outputs")
    for pdf in sorted(in_dir.glob("*.pdf")):
        process(pdf, out_dir)


if __name__ == "__main__":
    main()
