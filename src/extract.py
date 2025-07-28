from __future__ import annotations
import pathlib, re, fitz
from dataclasses import dataclass
from typing import List, Tuple, Dict
from PIL import Image
import pytesseract
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

SECTION_NUM_RE = re.compile(r"^\d+(\.\d+)+\s?$")   
NUM_HDR_RE     = re.compile(r"^\d+(\.\d+)+\s")    


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


def _guess_lang(t: str) -> str:
    try:
        return detect(t)
    except Exception:
        return "und"


def _ocr_page_lines(pix: fitz.Pixmap, langs: str = "eng+jpn+hin") -> List[Span]:
    from pytesseract import Output
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    data = pytesseract.image_to_data(img, lang=langs, output_type=Output.DICT)

    n = len(data["text"])
    groups: Dict[Tuple[int, int, int], Dict] = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        try:
            conf = int(float(data["conf"][i]))
        except Exception:
            conf = -1
        if not txt or conf < 60:
            continue

        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        l, t = int(data["left"][i]), int(data["top"][i])
        w, h = int(data["width"][i]), int(data["height"][i])
        r, b = l + w, t + h

        if key not in groups:
            groups[key] = {"words": [txt], "l": l, "t": t, "r": r, "b": b, "hs": [h]}
        else:
            g = groups[key]
            g["words"].append(txt)
            g["l"] = min(g["l"], l); g["t"] = min(g["t"], t)
            g["r"] = max(g["r"], r); g["b"] = max(g["b"], b)
            g["hs"].append(h)

    spans: List[Span] = []
    for g in groups.values():
        line_text = " ".join(g["words"]).strip()
        if not line_text:
            continue
        font_size = float(sum(g["hs"]) / max(1, len(g["hs"])))
        spans.append(
            Span(
                text=line_text,
                page=0,  # filled below
                bbox=(float(g["l"]), float(g["t"]), float(g["r"]), float(g["b"])),
                font_size=font_size,
                font_name="OCR",
                is_bold=False,
                is_italic=False,
                lang=_guess_lang(line_text),
            )
        )

    spans.sort(key=lambda s: (s.bbox[1], s.bbox[0]))
    return spans



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

        if SECTION_NUM_RE.fullmatch(buf.text.strip()) and not sec_prefix(nxt):
            merge = True
        else:
            diff_sec = sec_prefix(buf) and sec_prefix(nxt) and sec_prefix(buf) != sec_prefix(nxt)
            if diff_sec:
                merge = False
            else:
                same_baseline = abs(nxt.bbox[1] - buf.bbox[1]) < 2
                gap_ok = nxt.bbox[0] - buf.bbox[2] < 40
                same_font = abs(nxt.font_size - buf.font_size) < 0.6
                vert_ok = 0 < (nxt.bbox[1] - buf.bbox[1]) <= (buf.font_size or 12) * 1.8
                merge = (same_baseline and gap_ok) or (same_font and vert_ok)

        if merge:
            buf.text = f"{buf.text} {nxt.text}"
            buf.bbox = (buf.bbox[0], buf.bbox[1], nxt.bbox[2], max(buf.bbox[3], nxt.bbox[3]))
            buf.font_size = max(buf.font_size, nxt.font_size)
        else:
            merged.append(buf)
            buf = nxt
    merged.append(buf)
    return merged


def extract_spans(pdf_path: pathlib.Path, dpi: int = 150) -> List[Span]:
    doc = fitz.open(pdf_path)
    spans: List[Span] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        d = page.get_text("dict")

        if not any(b["type"] == 0 for b in d["blocks"]):  
            for sp in _ocr_page_lines(page.get_pixmap(dpi=dpi)):
                sp.page = i + 1
                spans.append(sp)
            continue

        raw: List[Span] = []
        for b in d["blocks"]:
            if b["type"] != 0:
                continue
            for l in b["lines"]:
                for s in l["spans"]:
                    txt = (s["text"] or "").strip()
                    if not txt:
                        continue
                    raw.append(
                        Span(
                            text=txt,
                            page=i + 1,
                            bbox=tuple(s["bbox"]),
                            font_size=float(s["size"]),
                            font_name=str(s["font"]),
                            is_bold=bool(s["flags"] & 2),
                            is_italic=bool(s["flags"] & 1),
                            lang=_guess_lang(txt),
                        )
                    )
        raw.sort(key=lambda s: (s.page, s.bbox[1], s.bbox[0]))
        spans.extend(_merge_line_spans(raw))

    doc.close()
    return spans
