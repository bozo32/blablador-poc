from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pytesseract
from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel

from backend.object_store import s3 as object_store_s3
from backend.settings import settings
from backend.spine.attempts import get_attempt


app = FastAPI(title="Fallback OCR Worker")


def _require_internal_auth(authorization: Optional[str]) -> None:
    token = str(settings.INTERNAL_SERVICE_TOKEN or "").strip()
    if not token:
        return
    auth = str(authorization or "").strip()
    if auth != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _parse_dpi_steps(value: str) -> List[int]:
    raw = str(value or "").strip()
    if not raw:
        return [150, 200, 300]
    out: List[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out or [150, 200, 300]


def _normalize_text(text: str) -> str:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t\f\v]+", " ", s)
    # Keep newlines for paragraph splitting; collapse multiple blank lines.
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _split_paragraphs(text: str) -> List[str]:
    blocks = []
    cur: List[str] = []
    for ln in (text or "").split("\n"):
        if not ln.strip():
            if cur:
                blocks.append(" ".join([x.strip() for x in cur if x.strip()]).strip())
                cur = []
            continue
        cur.append(ln.strip())
    if cur:
        blocks.append(" ".join([x.strip() for x in cur if x.strip()]).strip())
    return [b for b in blocks if b]


def _attempt_cancelled(attempt_id: str) -> bool:
    try:
        a = get_attempt(str(attempt_id))
        return bool(a and str(a.get("state")) == "cancelled")
    except Exception:
        return False


def _pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    mode = "RGB"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img


def _ocr_with_steps(
    img: Image.Image,
    *,
    languages: str,
    dpi_steps: List[int],
    min_conf: int = 55,
) -> Tuple[str, int, float]:
    # We run OCR at caller-provided DPI steps by re-rendering upstream; here we
    # just score confidence and return text.
    data = pytesseract.image_to_data(
        img, lang=languages, output_type=pytesseract.Output.DICT
    )
    confs = []
    for c in data.get("conf", []) or []:
        try:
            v = float(c)
        except Exception:
            continue
        if v >= 0:
            confs.append(v)
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    txt = pytesseract.image_to_string(img, lang=languages)
    return txt, min_conf, float(mean_conf)


class FallbackRequest(BaseModel):
    work_id: str
    attempt_id: str
    pdf_object_key: str
    ocr_languages: Optional[str] = None
    text_min_chars: Optional[int] = None
    dpi_steps: Optional[List[int]] = None


class FallbackResponse(BaseModel):
    pages_object_key: str
    pages_bytes: int
    body_object_key: str
    body_bytes: int
    refs_object_key: str
    refs_bytes: int
    quality_patch: Dict[str, Any]


@app.post("/v1/fallback", response_model=FallbackResponse)
def run_fallback(
    req: FallbackRequest,
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    _require_internal_auth(authorization)

    work_id = str(req.work_id or "").strip()
    attempt_id = str(req.attempt_id or "").strip()
    pdf_key = str(req.pdf_object_key or "").strip().lstrip("/")
    if not work_id:
        raise HTTPException(status_code=400, detail="work_id is required")
    if not attempt_id:
        raise HTTPException(status_code=400, detail="attempt_id is required")
    if not pdf_key:
        raise HTTPException(status_code=400, detail="pdf_object_key is required")

    if _attempt_cancelled(attempt_id):
        raise HTTPException(status_code=409, detail="attempt_cancelled")

    languages = (
        str(req.ocr_languages or settings.FALLBACK_OCR_LANGUAGES or "eng").strip()
        or "eng"
    )
    text_min_chars = int(
        req.text_min_chars
        if req.text_min_chars is not None
        else settings.FALLBACK_TEXT_MIN_CHARS
    )
    dpi_steps = req.dpi_steps or _parse_dpi_steps(settings.FALLBACK_OCR_DPI_STEPS)

    # Download PDF to temp path.
    tmp = tempfile.NamedTemporaryFile(
        prefix="blablador_fallback_", suffix=".pdf", delete=False
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        object_store_s3.download_to_path(pdf_key, tmp_path)
        doc = fitz.open(str(tmp_path))

        pages_total = int(doc.page_count)
        pages_ocr = 0
        text_layer_nonempty_pages = 0
        lines: List[str] = []
        body_parts: List[str] = []

        for i in range(pages_total):
            if _attempt_cancelled(attempt_id):
                raise HTTPException(status_code=409, detail="attempt_cancelled")

            page = doc.load_page(i)
            raw_text = page.get_text("text") or ""
            norm = _normalize_text(raw_text)
            if norm.strip():
                text_layer_nonempty_pages += 1

            method = "text_layer"
            ocr_dpi = None
            ocr_mean_conf = None
            if len(norm) < max(0, text_min_chars):
                method = "ocr"
                pages_ocr += 1
                best_txt = ""
                best_conf = -1.0
                best_dpi = None
                for dpi in dpi_steps:
                    # Render at DPI.
                    zoom = float(dpi) / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = _pixmap_to_pil(pix)
                    txt, _minc, mean_conf = _ocr_with_steps(
                        img,
                        languages=languages,
                        dpi_steps=dpi_steps,
                    )
                    txt_norm = _normalize_text(txt)
                    if mean_conf > best_conf:
                        best_conf = mean_conf
                        best_txt = txt_norm
                        best_dpi = dpi
                    # Escalate until we exceed a confidence threshold.
                    if mean_conf >= 55:
                        break
                norm = best_txt
                ocr_dpi = best_dpi
                ocr_mean_conf = best_conf

            paras = _split_paragraphs(norm)
            rec: Dict[str, Any] = {
                "page_index": int(i),
                "text": str(norm or ""),
                "paragraphs": paras,
                "provenance": {
                    "method": method,
                    "ocr": {
                        "languages": languages,
                        "dpi": ocr_dpi,
                        "mean_conf": ocr_mean_conf,
                    }
                    if method == "ocr"
                    else None,
                },
            }
            lines.append(
                json.dumps(
                    rec, ensure_ascii=True, sort_keys=True, separators=(",", ":")
                )
            )
            if norm:
                body_parts.append(norm)

        pages_blob = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
        body_blob = (
            "\n\n".join([b for b in body_parts if b]).strip()
            + ("\n" if body_parts else "")
        ).encode("utf-8")
        refs_blob = b"[]\n"

        pages_key = f"extract/{work_id}/attempts/{attempt_id}/fallback/pages.jsonl"
        body_key = f"extract/{work_id}/attempts/{attempt_id}/fallback/body.txt"
        refs_key = f"extract/{work_id}/attempts/{attempt_id}/fallback/refs.json"

        object_store_s3.put_bytes(
            pages_key, pages_blob, content_type="application/x-ndjson"
        )
        object_store_s3.put_bytes(body_key, body_blob, content_type="text/plain")
        object_store_s3.put_bytes(refs_key, refs_blob, content_type="application/json")

        coverage = (
            (float(text_layer_nonempty_pages) / float(pages_total))
            if pages_total
            else 0.0
        )
        quality_patch = {
            "fallback": {
                "used": True,
                "ocr_used": bool(pages_ocr > 0),
                "pages_total": int(pages_total),
                "pages_ocr": int(pages_ocr),
                "text_layer_coverage": coverage,
            }
        }

        return {
            "pages_object_key": pages_key,
            "pages_bytes": int(len(pages_blob)),
            "body_object_key": body_key,
            "body_bytes": int(len(body_blob)),
            "refs_object_key": refs_key,
            "refs_bytes": int(len(refs_blob)),
            "quality_patch": quality_patch,
        }
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
