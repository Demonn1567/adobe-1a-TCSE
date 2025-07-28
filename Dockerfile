# Use amd64 to match judge hardware
FROM --platform=linux/amd64 python:3.10-slim

# Prevent .pyc, get nice logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- OS deps (tesseract OCR + locales) ----
# If image size becomes a concern, you can drop jpn/hin packs and keep only -eng
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-jpn \
        tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- App code & schema ----
# We don't need PDFs or local outputs inside the image,
# but we DO need the schema used by json validation.
COPY src /app/src
COPY schema /app/sample_dataset/schema

# Default run: process /app/input -> /app/output
CMD ["python", "-m", "src.runner"]
