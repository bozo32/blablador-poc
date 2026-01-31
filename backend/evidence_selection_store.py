"""On-disk persistence for reviewer evidence selections."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from backend.schemas import EvidenceSelectionPayload, EvidenceSelectionUpsertRequest
from backend.settings import AppSettings, settings as app_settings


_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_claim_id(claim_id: str) -> str:
    value = (claim_id or "").strip() or "unknown"
    return _SAFE_ID_RE.sub("_", value)


class EvidenceSelectionStore:
    def __init__(self, *, settings: AppSettings = app_settings) -> None:
        """Create a store rooted next to the evidence run directory."""
        self.settings = settings

    @property
    def root_dir(self) -> Path:
        base = getattr(self.settings, "EVIDENCE_STORE_DIR", None)
        if base is None:
            return Path("data") / "evidence_selections"
        return Path(base).parent / "evidence_selections"

    def _path_for_claim(self, claim_id: str) -> Path:
        return self.root_dir / f"{_safe_claim_id(claim_id)}.json"

    def read(self, claim_id: str) -> Optional[EvidenceSelectionPayload]:
        path = self._path_for_claim(claim_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return EvidenceSelectionPayload.model_validate(payload)

    def upsert(
        self, claim_id: str, selection: dict | EvidenceSelectionUpsertRequest
    ) -> EvidenceSelectionPayload:
        request = (
            selection
            if isinstance(selection, EvidenceSelectionUpsertRequest)
            else EvidenceSelectionUpsertRequest.model_validate(selection)
        )
        stored = EvidenceSelectionPayload(
            claim_id=claim_id,
            updated_at=_now(),
            verdict=request.verdict,
            primary=request.primary,
            secondary=request.secondary,
            note=request.note,
        )

        path = self._path_for_claim(claim_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(stored.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return stored

    def validate(self, selection: dict) -> EvidenceSelectionUpsertRequest:
        """Validate a selection payload without persisting it."""
        try:
            return EvidenceSelectionUpsertRequest.model_validate(selection)
        except ValidationError:
            raise


selection_store = EvidenceSelectionStore(settings=app_settings)


__all__ = ["EvidenceSelectionStore", "selection_store"]
