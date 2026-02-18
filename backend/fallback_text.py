from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader


_WS_RE = re.compile(r"[ \t\f\v]+")


def _clean_line(line: str) -> str:
    s = (line or "").replace("\u00a0", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s


def _normalize_text(raw: str) -> str:
    # Basic normalization intended to be stable and conservative.
    s = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [_clean_line(x) for x in s.split("\n")]

    # Drop repeated empty lines.
    out: List[str] = []
    prev_empty = True
    for ln in lines:
        if not ln:
            if not prev_empty:
                out.append("")
            prev_empty = True
            continue
        prev_empty = False
        out.append(ln)

    # Simple wrapped-line join heuristic:
    # - join if previous line does not end a sentence and next line starts lowercase.
    joined: List[str] = []
    for ln in out:
        if not joined:
            joined.append(ln)
            continue
        if not ln:
            joined.append("")
            continue
        prev = joined[-1]
        if prev and prev[-1] not in ".:;!?" and ln[:1].islower():
            joined[-1] = prev + " " + ln
        else:
            joined.append(ln)

    # Dehyphenate end-of-line hyphens where we joined.
    txt = "\n".join(joined)
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)
    return txt.strip()


def _split_paragraphs(text: str) -> List[str]:
    blocks = []
    cur: List[str] = []
    for ln in (text or "").split("\n"):
        if not ln.strip():
            if cur:
                blocks.append(" ".join(cur).strip())
                cur = []
            continue
        cur.append(ln.strip())
    if cur:
        blocks.append(" ".join(cur).strip())
    return [b for b in blocks if b]


@dataclass(frozen=True)
class FallbackPage:
    page_index: int
    text: str
    paragraphs: List[str]


def extract_fallback_pages(pdf_path: Path) -> List[FallbackPage]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages: List[FallbackPage] = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        norm = _normalize_text(raw)
        paras = _split_paragraphs(norm)
        pages.append(FallbackPage(page_index=i, text=norm, paragraphs=paras))
    return pages


def pages_to_jsonl(pages: List[FallbackPage]) -> bytes:
    lines: List[str] = []
    for p in pages:
        rec: Dict[str, Any] = {
            "page_index": int(p.page_index),
            "text": str(p.text or ""),
            "paragraphs": list(p.paragraphs or []),
            "provenance": {"method": "pypdf.text_layer"},
        }
        lines.append(
            json.dumps(rec, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        )
    blob = "\n".join(lines) + ("\n" if lines else "")
    return blob.encode("utf-8")


def pages_to_body_text(pages: List[FallbackPage]) -> bytes:
    parts: List[str] = []
    for p in pages:
        if p.text:
            parts.append(p.text)
    body = "\n\n".join(parts).strip() + ("\n" if parts else "")
    return body.encode("utf-8")


def fallback_quality(pages: List[FallbackPage]) -> Dict[str, Any]:
    total = len(pages)
    nonempty = sum(1 for p in pages if (p.text or "").strip())
    coverage = (float(nonempty) / float(total)) if total else 0.0
    return {
        "fallback": {
            "used": True,
            "ocr_used": False,
            "pages_total": int(total),
            "pages_ocr": 0,
            "text_layer_coverage": coverage,
        }
    }
