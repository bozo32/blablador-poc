#!/bin/sh
set -eu

out_path="${1:-fixtures/text-1.pdf}"

mkdir -p "$(dirname "$out_path")"

# Generate a simple text-layer PDF fixture (deterministic) for end-to-end flows.

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

out_path = Path(os.environ.get("OUT_PATH", "fixtures/text-1.pdf"))
out_path.parent.mkdir(parents=True, exist_ok=True)

doc = fitz.open()
page = doc.new_page(width=595, height=842)

text = (
    "A Tiny Fixture Article\n\n"
    "This is a deterministic text-layer PDF fixture for E2E tests.\n"
    "It includes enough text to produce paragraphs and sentences.\n\n"
    "Claim: Renewable energy can reduce emissions in many scenarios.\n"
)

page.insert_text(
    (72, 72),
    text,
    fontsize=12,
    fontname="helv",
)

doc.save(str(out_path))
doc.close()
print(f"wrote {out_path}")
PY
