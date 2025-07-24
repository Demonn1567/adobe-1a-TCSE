from src.extract import extract_spans
import pathlib, sys

pdf_path = pathlib.Path(sys.argv[1])
for s in extract_spans(pdf_path):
    if s.page == 1:     # first page only
        y = round(s.bbox[1])
        print(f"{s.page:>2} {s.font_size:>5.1f}  {y:>4}  {s.text}")
