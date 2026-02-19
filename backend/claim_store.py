"""Spine-backed persistence for confirmed claim parses.

Phase 09.3 removes durable local SQLite stores (`data/claims.db`). Confirmed
claims are stored in Postgres in the `confirmed_claims` table.
"""

from __future__ import annotations

from datetime import datetime, timezone

from backend.db.pg import connect
from backend.schemas import ClaimConfirmationRequest
from backend.settings import AppSettings, settings as app_settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ClaimStore:
    def __init__(self, *, settings: AppSettings = app_settings) -> None:
        """Create a confirmed-claims store backed by Postgres."""
        self.settings = settings

    def wipe(self) -> None:
        """Delete all confirmed-claim rows (keeps schema)."""
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE confirmed_claims")

    def persist_confirmed_claims(self, payload: ClaimConfirmationRequest) -> int:
        """Upsert confirmed claims for a sentence.

        Returns the number of claim rows present in the request payload.
        """
        if not payload.confirmed_claims:
            return 0

        project_id = str(
            getattr(self.settings, "DEFAULT_PROJECT_ID", "default") or "default"
        )
        now = _now()
        rows = 0

        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                for claim in payload.confirmed_claims:
                    cur.execute(
                        """
                        INSERT INTO confirmed_claims(
                          project_id,
                          document_id,
                          sentence_id,
                          claim_index,
                          parsed_text,
                          original_text,
                          segmentation_model,
                          reviewer_uid,
                          confirmed_at,
                          confidence,
                          citation_index,
                          target_id,
                          sentence_text
                        )
                        VALUES (
                          %s,
                          %s,
                          %s,
                          %s,
                          %s,
                          %s,
                          %s,
                          %s,
                          %s,
                          %s,
                          %s,
                          %s,
                          %s
                        )
                        ON CONFLICT(document_id, sentence_id, claim_index) DO UPDATE
                          SET project_id=excluded.project_id,
                              parsed_text=excluded.parsed_text,
                              original_text=excluded.original_text,
                              segmentation_model=excluded.segmentation_model,
                              reviewer_uid=excluded.reviewer_uid,
                              confirmed_at=excluded.confirmed_at,
                              confidence=excluded.confidence,
                              citation_index=excluded.citation_index,
                              target_id=excluded.target_id,
                              sentence_text=excluded.sentence_text
                        """,
                        (
                            project_id,
                            payload.document_id,
                            payload.sentence_id,
                            int(claim.claim_index),
                            str(claim.parsed_text),
                            claim.original_text,
                            payload.segmentation_model,
                            payload.reviewer_uid,
                            now,
                            claim.confidence,
                            int(payload.citation_index),
                            payload.target_id,
                            payload.sentence_text,
                        ),
                    )
                    rows += 1

        return rows


claim_store = ClaimStore(settings=app_settings)
