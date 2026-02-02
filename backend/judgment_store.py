"""On-disk persistence for per-claim reviewer judgments."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from backend.schemas import JudgmentPayload, JudgmentUpsertRequest
from backend.settings import AppSettings, settings as app_settings


_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_claim_id(claim_id: str) -> str:
    value = (claim_id or "").strip() or "unknown"
    return _SAFE_ID_RE.sub("_", value)


def _claim_hash_suffix(claim_id: str) -> str:
    return sha1((claim_id or "").encode("utf-8")).hexdigest()[:8]


class JudgmentStore:
    def __init__(self, *, settings: AppSettings = app_settings) -> None:
        """Create a store for per-claim judgments."""
        self.settings = settings

    @property
    def root_dir(self) -> Path:
        base = getattr(self.settings, "EVIDENCE_STORE_DIR", None)
        if base is None:
            return Path("data") / "judgments"
        return Path(base).parent / "judgments"

    def _path_for_claim(self, claim_id: str) -> Path:
        safe = _safe_claim_id(claim_id)
        suffix = _claim_hash_suffix(claim_id)
        return self.root_dir / f"{safe}__{suffix}.json"

    def read(self, claim_id: str) -> Optional[JudgmentPayload]:
        path = self._path_for_claim(claim_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return JudgmentPayload.model_validate(payload)

    def upsert(
        self, claim_id: str, judgment: dict | JudgmentUpsertRequest
    ) -> JudgmentPayload:
        request = (
            judgment
            if isinstance(judgment, JudgmentUpsertRequest)
            else JudgmentUpsertRequest.model_validate(judgment)
        )

        stored = JudgmentPayload(
            claim_id=claim_id,
            updated_at=_now(),
            status=request.status,
            verdict=request.verdict,
            notes=request.notes,
            doc_id=request.doc_id,
            citation_index=request.citation_index,
            target_id=request.target_id,
            sentence_id=request.sentence_id,
            callout=request.callout,
            reference_id=request.reference_id,
            doi=request.doi,
            author=request.author,
            year=request.year,
            claim_text=request.claim_text,
        )

        path = self._path_for_claim(claim_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(stored.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return stored

    def validate(self, judgment: dict) -> JudgmentUpsertRequest:
        """Validate an upsert payload without persisting it."""
        try:
            return JudgmentUpsertRequest.model_validate(judgment)
        except ValidationError:
            raise


judgment_store = JudgmentStore(settings=app_settings)


__all__ = ["JudgmentStore", "judgment_store"]
