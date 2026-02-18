from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _stable_id(prefix: str, text: str) -> str:
    blob = (prefix + "::" + (text or "")).encode("utf-8", errors="ignore")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_body_from_plaintext(*, document_id: str, text: str) -> Dict[str, Any]:
    """Build a DocumentBodyResponse-compatible payload from plain text.

    This is a deterministic best-effort fallback used when TEI is unavailable.
    """
    raw = (text or "").strip()
    if not raw:
        return {"document_id": str(document_id), "paragraphs": []}

    paras = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]

    out_paras: List[Dict[str, Any]] = []
    for i, para in enumerate(paras):
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(para) if s.strip()]
        sent_out = []
        for j, s in enumerate(sentences):
            sent_out.append(
                {
                    "sentence_id": f"fb:{i}:{j}:{_stable_id('s', s)}",
                    "segments": [
                        {
                            "type": "text",
                            "text": s,
                            "citation_index": None,
                            "target_id": None,
                            "callout": None,
                            "label": None,
                            "sentence_id": None,
                        }
                    ],
                    "citation_indices": [],
                }
            )

        out_paras.append(
            {
                "paragraph_id": f"fb:p:{i}:{_stable_id('p', para)}",
                "sentences": sent_out,
            }
        )

    return {"document_id": str(document_id), "paragraphs": out_paras}
