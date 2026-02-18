#!/bin/sh
set -eu

out_path="${1:-fixtures/scanned-1.pdf}"

mkdir -p "$(dirname "$out_path")"

# Generate an image-only (no text layer) PDF fixture for OCR testing.
# Uses the app-api container so we don't require host Python deps.

docker compose run --rm \
  -v "${PWD}:/repo" \
  -w /repo \
  -e "OUT_PATH=${out_path}" \
  app-api \
  python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

out_path = Path(os.environ.get("OUT_PATH", "fixtures/scanned-1.pdf"))
out_path.parent.mkdir(parents=True, exist_ok=True)

W, H = 1240, 1754  # ~A4 at 150dpi
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

text = "Scanned Fixture\nThis page has no text layer.\nOCR should recover this."

try:
    font = ImageFont.load_default()
except Exception:
    font = None

draw.multiline_text((80, 120), text, fill="black", font=font, spacing=12)

png_bytes = None
import io

buf = io.BytesIO()
img.save(buf, format="PNG")
png_bytes = buf.getvalue()

doc = fitz.open()
page = doc.new_page(width=595, height=842)  # A4 points
rect = fitz.Rect(0, 0, 595, 842)
page.insert_image(rect, stream=png_bytes)

doc.save(str(out_path))
doc.close()

print(f"wrote {out_path}")
PY
