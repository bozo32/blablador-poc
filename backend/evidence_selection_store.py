"""Spine-backed persistence for reviewer evidence selections.

Phase 09.3: selections persist in Postgres (no `data/evidence_selections/**`).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import ValidationError

from backend.db.pg import connect
from backend.schemas import EvidenceSelectionPayload, EvidenceSelectionUpsertRequest
from backend.settings import AppSettings, settings as app_settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EvidenceSelectionStore:
    def __init__(self, *, settings: AppSettings = app_settings) -> None:
        """Create a selection store backed by Postgres."""
        self.settings = settings

    def read(
        self, claim_id: str, reviewer_uid: str = "default"
    ) -> Optional[EvidenceSelectionPayload]:
        cid = str(claim_id or "").strip()
        rid = str(reviewer_uid or "default").strip() or "default"
        if not cid:
            return None
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT verdict, primary_json, secondary_json, note, updated_at
                    FROM evidence_selections
                    WHERE claim_id=%s AND reviewer_uid=%s
                    """,
                    (cid, rid),
                )
                row = cur.fetchone()
                if not row:
                    return None
        primary = row[1] if isinstance(row[1], dict) else json.loads(row[1] or "{}")
        raw_secondary = (
            row[2] if isinstance(row[2], (list, dict)) else json.loads(row[2] or "[]")
        )
        secondary = raw_secondary if isinstance(raw_secondary, list) else []
        updated_at = row[4].isoformat().replace("+00:00", "Z") if row[4] else _now()
        return EvidenceSelectionPayload(
            claim_id=cid,
            updated_at=updated_at,
            verdict=str(row[0] or "none"),
            primary=primary or None,
            secondary=secondary,
            note=row[3],
        )

    def upsert(
        self, claim_id: str, selection: dict | EvidenceSelectionUpsertRequest
    ) -> EvidenceSelectionPayload:
        cid = str(claim_id or "").strip()
        if not cid:
            raise ValueError("claim_id is required")
        request = (
            selection
            if isinstance(selection, EvidenceSelectionUpsertRequest)
            else EvidenceSelectionUpsertRequest.model_validate(selection)
        )
        stored = EvidenceSelectionPayload(
            claim_id=cid,
            updated_at=_now(),
            verdict=request.verdict,
            primary=request.primary,
            secondary=request.secondary,
            note=request.note,
        )
        rid = "default"
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO evidence_selections(
                      selection_id,
                      project_id,
                      updated_by_user_id,
                      claim_id,
                      reviewer_uid,
                      updated_at,
                      verdict,
                      primary_json,
                      secondary_json,
                      note
                    )
                    VALUES (
                      %s,
                      'default',
                      'local',
                      %s,
                      %s,
                      now(),
                      %s,
                      %s::jsonb,
                      %s::jsonb,
                      %s
                    )
                    ON CONFLICT(claim_id, reviewer_uid) DO UPDATE
                      SET updated_at=excluded.updated_at,
                          updated_by_user_id=excluded.updated_by_user_id,
                          verdict=excluded.verdict,
                          primary_json=excluded.primary_json,
                          secondary_json=excluded.secondary_json,
                          note=excluded.note
                    """,
                    (
                        str(uuid4()),
                        cid,
                        rid,
                        str(stored.verdict),
                        json.dumps(
                            stored.primary.model_dump(mode="json")
                            if stored.primary is not None
                            else {},
                            ensure_ascii=True,
                        ),
                        json.dumps(
                            [item.model_dump(mode="json") for item in stored.secondary]
                            if stored.secondary
                            else [],
                            ensure_ascii=True,
                        ),
                        stored.note,
                    ),
                )
        return stored

    def validate(self, selection: dict) -> EvidenceSelectionUpsertRequest:
        try:
            return EvidenceSelectionUpsertRequest.model_validate(selection)
        except ValidationError:
            raise


selection_store = EvidenceSelectionStore(settings=app_settings)


__all__ = ["EvidenceSelectionStore", "selection_store"]
