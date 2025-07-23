"""
extract.py – OCR fallback + robust span merging
"""
from __future__ import annotations
import pathlib, re, fitz
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image
import pytesseract
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

SECTION_NUM_RE = re.compile(r"^\d+(\.\d+)+\s?$")   # "2.1" / "2.1 "
NUM_HDR_RE     = re.compile(r"^\d+(\.\d+)+\s")      # "2.1 "

# ------------------------------------------------------------------ #
@dataclass
class Span:
    text: str
    page: int
    bbox: Tuple[float, float, float, float]
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool
    lang: str
    level: str | None = None


# ---------- helpers ------------------------------------------------ #
def _guess_lang(t: str) -> str:
    try:
        return detect(t)
    except Exception:
        return "und"


def _ocr_page(pix: fitz.Pixmap, langs="eng+jpn+hin") -> List[Span]:
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    txt = pytesseract.image_to_string(img, lang=langs).strip()
    return [
        Span(
            txt,
            0,
            (0, 0, pix.width, pix.height),
            0,
            "OCR",
            False,
            False,
            _guess_lang(txt),
        )
    ]


# ---------- merge logic -------------------------------------------- #
def _merge_line_spans(raw: List[Span]) -> List[Span]:
    if not raw:
        return []
    merged, buf = [], raw[0]

    def sec_prefix(s: Span):
        m = NUM_HDR_RE.match(s.text)
        return m.group(0).strip() if m else None

    for nxt in raw[1:]:
        if nxt.page != buf.page:
            merged.append(buf)
            buf = nxt
            continue

        # special: handle "2." + "Introduction …" split
        if SECTION_NUM_RE.fullmatch(buf.text.strip()) and not sec_prefix(nxt):
            merge = True
        else:
            diff_sec = (
                sec_prefix(buf)
                and sec_prefix(nxt)
                and sec_prefix(buf) != sec_prefix(nxt)
            )
            if diff_sec:
                merge = False
            else:
                same_baseline = abs(nxt.bbox[1] - buf.bbox[1]) < 2
                gap_ok = nxt.bbox[0] - buf.bbox[2] < 40
                same_font = abs(nxt.font_size - buf.font_size) < 0.6
                vert_ok = 0 < (nxt.bbox[1] - buf.bbox[1]) <= buf.font_size * 1.8
                merge = (same_baseline and gap_ok) or (same_font and vert_ok)

        if merge:
            buf.text = f"{buf.text} {nxt.text}"
            buf.bbox = (
                buf.bbox[0],
                buf.bbox[1],
                nxt.bbox[2],
                max(buf.bbox[3], nxt.bbox[3]),
            )
        else:
            merged.append(buf)
            buf = nxt
    merged.append(buf)
    return merged


# ---------- public API --------------------------------------------- #
def extract_spans(pdf_path: pathlib.Path, dpi: int = 150) -> List[Span]:
    doc = fitz.open(pdf_path)
    spans: List[Span] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        d = page.get_text("dict")

        # scanned page → OCR
        if not any(b["type"] == 0 for b in d["blocks"]):
            for sp in _ocr_page(page.get_pixmap(dpi=dpi)):
                sp.page = i + 1
                spans.append(sp)
            continue

        raw: List[Span] = []
        for b in d["blocks"]:
            if b["type"] != 0:
                continue
            for l in b["lines"]:
                for s in l["spans"]:
                    txt = s["text"].strip()
                    if txt:
                        raw.append(
                            Span(
                                txt,
                                i + 1,
                                tuple(s["bbox"]),
                                s["size"],
                                s["font"],
                                bool(s["flags"] & 2),
                                bool(s["flags"] & 1),
                                _guess_lang(txt),
                            )
                        )
        raw.sort(key=lambda s: (s.page, s.bbox[1], s.bbox[0]))
        spans.extend(_merge_line_spans(raw))

    doc.close()
    return spans
