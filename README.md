# Challenge 1a: PDF Processing Solution

## Goal

From a PDF (≤ 50 pages), produce a clean, hierarchical outline: Title and headings (H1/H2/H3, with page numbers), as JSON.

## Constraints

Offline, CPU-only, AMD64, fast (≤ 10 s for 50 pages), and no large models.

## Approach

### High-level Pipeline

**Span Extraction (PyMuPDF):**
We read each page and collect spans (text fragments with font, size, bold/italic flags, and bounding box). If a page has no extractable text (scanned), we do a lightweight OCR fallback (pytesseract) so the pipeline remains robust.

**Candidate Filtering (rules):**
We drop obvious non-headings: page headers/footers that repeat across many pages, dates, dot-leaders ("……"), pure numbers, long sentences, and list items.
We keep numbered sections ("2.1", "3.4.2") and trim long lines to the first clause before `:` to make them heading-like.

**Heading Prediction (tiny model + clustering):**

- A logistic gate (linear coefficients baked into the repo) favors large fonts, bold text, and top-of-page positions. Anything clearly "heading-ish" passes this gate.
- For the kept spans, we run K-Means on font size (k ≤ 4) and map clusters to levels: the highest-size cluster → H1, the rest → H2/H3/… (with a small tolerance so very similar font sizes share the same level).
- We force-include items that begin with section numbers even if the gate is unsure.

**Title Detection (first page heuristics):**
On page 1, we take the max-font lines at the very top as the title and optionally append up to 2 subtitle lines (short, still large, and not list/number starts). We scrub artifacts (character stutter, duplicate words, excessive whitespace).
If no reliable title emerges, we fall back to the first H1 or the largest text on p1.

**Outline Assembly (stable ordering + page normalization):**
We sort headings by (page, y, x) and emit:

```json
{ "level": "H2", "text": "Background", "page": 3 }
```

For single-page PDFs we use logical page 0 (as required in samples). For multi-page PDFs we keep the natural page indices starting at 1.

**Schema Validation & Dump:**
Output JSON is validated against the provided schema and written per-file to `/app/output/<name>.json`.

### Why This Works

- Layout-aware signals (font size/weight and vertical position) carry most heading information; the tiny model simply turns those into a reliable gate and the clusterer assigns levels consistently between documents.
- Rules before ML reduce noise; rules after ML correct known edge cases (e.g., numbered sections, colon-trim).
- The approach is fast, deterministic, and explainable—no heavy models or internet calls.

## Models & Libraries Used

**No large ML models.** Everything runs offline and fits the constraints.

- **PyMuPDF (fitz)** – fast PDF text extraction with per-span typography and positions.
- **pytesseract + Pillow** – optional OCR fallback for image-only pages (uses system tesseract if present; not needed for text PDFs).
- **numpy / scikit-learn** – small in-process logic:
  - Logistic gate: fixed coefficients (`MODEL_COEF`, `MODEL_INT`) baked into `classify.py`; not trained at runtime.
  - K-Means (k ≤ 4): clusters font sizes to H-levels per document.
- **langdetect** – language guess for spans (helps avoid some OCR noise).
- **orjson** – fast JSON writer.
- **jsonschema** – validates outputs against the official schema.

**Footprint:** No model files; the heaviest piece is system OCR (if installed). The solution runs fine without OCR for digital PDFs.

## Build & Run (Docker)

These commands follow the organizers' "Expected Execution" (AMD64, offline). The container automatically reads all PDFs from `/app/input` and writes JSON to `/app/output`.

### 1. Prepare Folders Locally

```bash
# from challenge_1a/ (repo root for Round 1A)
mkdir -p input output
# copy your PDFs into ./input
cp sample_dataset/pdfs/*.pdf input/   # optional: for quick testing
```

### 2. Build the Image (Portable AMD64)

**On Apple Silicon / buildx users (recommended):**

```bash
docker buildx build --platform linux/amd64 -t challenge1a:latest . --load
```

**On standard Docker:**

```bash
docker build --platform linux/amd64 -t challenge1a:latest .
```

### 3. Run (As Per the Brief)

```bash
docker run --rm \
  -v "$PWD/input":/app/input \
  -v "$PWD/output":/app/output \
  --network none \
  challenge1a:latest
```

The container will process every `*.pdf` in `/app/input` and create matching `*.json` files in `/app/output`.

### 4. Example Output

```json
{
  "title": "RFP: Request for Proposal",
  "outline": [
    { "level": "H1", "text": "Summary", "page": 1 },
    { "level": "H2", "text": "Background", "page": 2 },
    { "level": "H3", "text": "Timeline:", "page": 1 }
  ]
}
```

## Project Structure (Key Files)

```
challenge_1a/
├── src/
│   ├── extract.py       # PyMuPDF spans + OCR fallback; merges broken lines
│   ├── features.py      # heading candidate filter + numeric features
│   ├── classify.py      # logistic gate + KMeans → H1/H2/H3...
│   ├── assemble.py      # stable ordering + page normalization
│   ├── runner.py        # Docker entrypoint; batch over /app/input
│   └── utils.py         # text normalization & header/footer detection
├── sample_dataset/
│   ├── pdfs/            # example PDFs (local testing)
│   └── schema/output_schema.json
├── Dockerfile
└── README.md            # this file
```

## Notes, Limits & Tips

- **Speed:** Designed to finish well under 10 s for a 50-page digital PDF on CPU.
- **Single-page flyers:** If headings are sparse, we elevate the top 1–2 largest lines to H1/H2 so you still get a useful outline.
- **Title robustness:** We avoid mistaking list fields ("RSVP: ___") for the title by rejecting short numbered/list patterns and using max-font lines near the top.
- **OCR:** If a page is image-only and tesseract isn't installed, the page is skipped from OCR, but that won't affect digital PDFs (the official samples are text).

## Troubleshooting

- **"image not found" when running:** build with `--load` (for buildx).
- **No outputs:** ensure `./input` contains PDFs and `output/` is writable.
- **Weird headings from footers:** header/footer detector uses repetition across pages; raise/lower the repetition ratio in `utils.build_header_footer_stopset` if needed.