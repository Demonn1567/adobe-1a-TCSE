"""
runner.py – main CLI for Challenge 1A
"""
from __future__ import annotations
import json, pathlib, re, sys, orjson
from jsonschema import validate

from .extract   import extract_spans
from .features  import filter_spans
from .classify  import predict_headings
from .assemble  import build_outline

SCHEMA = pathlib.Path(__file__).parent.parent / "sample_dataset/schema/output_schema.json"

WORD_DUP_RE  = re.compile(r"\b(\w{2,})(\s+\1\b)+", flags=re.I)
CHAR_STUTTER = re.compile(r"(\w)\s+\1\w?")          # e.g. "Re equest" → "Request"

# --------------------------------------------------------------------------- #
def detect_title(spans):
    """Pick max‑font lines on p 1 and scrub OCR stutter / duplicates."""
    p1 = [s for s in spans if s.page == 1]
    if not p1:
        return ""
    max_sz = max(s.font_size for s in p1)
    lines  = sorted(
        (s for s in p1 if abs(s.font_size - max_sz) < 0.5),
        key=lambda s: s.bbox[1],
    )

    title = "  ".join(l.text.strip() for l in lines)

    # 1. Collapse character‑level stutter  ("Re equest" → "Request")
    while CHAR_STUTTER.search(title):
        title = CHAR_STUTTER.sub(r"\1", title)

    # 2. Token‑level dedup  ("RFP: RFP:" → "RFP:")
    title = WORD_DUP_RE.sub(r"\1", title)

    # 3. Remove stray single‑letter tokens
    title = " ".join(tok for tok in title.split() if len(tok) > 1)

    return title.strip()


# --------------------------------------------------------------------------- #
def process(pdf: pathlib.Path, out_dir: pathlib.Path):
    spans = extract_spans(pdf)
    title = detect_title(spans)

    spans = filter_spans(spans, title, page_cnt=max(s.page for s in spans))
    heads = predict_headings(spans)
    data  = {"title": title, "outline": build_outline(heads)}

    validate(instance=data, schema=json.loads(SCHEMA.read_text()))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf.stem}.json"
    out_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    print(f"✅  {pdf.name} → {out_path.relative_to(out_dir.parent)}")


# --------------------------------------------------------------------------- #
def main():
    docker = "/app/" in str(pathlib.Path(__file__).resolve())
    in_dir  = pathlib.Path("/app/input")  if docker else pathlib.Path("sample_dataset/pdfs")
    out_dir = pathlib.Path("/app/output") if docker else pathlib.Path("sample_dataset/outputs")
    for pdf in sorted(in_dir.glob("*.pdf")):
        process(pdf, out_dir)

if __name__ == "__main__":
    main()
