"""Spine-backed persistence for evidence ranking runs and history.

Phase 09.3: evidence runs persist in Postgres; large candidate payloads live in
object storage with a Postgres pointer.
"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import uuid4

from backend.db.pg import connect
from backend.object_store import s3 as object_store_s3
from backend.settings import AppSettings, settings as app_settings

from . import serializers


class EvidenceRunStore:
    """Persist evidence ranking runs with limited history and delta metadata."""

    def __init__(self, *, settings: AppSettings | None = None) -> None:
        """Create a run store backed by Postgres + object storage."""
        self.settings = settings or app_settings

    def record_run(
        self,
        claim_id: str,
        *,
        candidates: Sequence[Any],
        metadata: dict | None = None,
    ) -> dict:
        if not claim_id:
            raise ValueError("claim_id is required")

        normalized_candidates = [self._normalize_candidate(c) for c in candidates]
        previous = self.latest_run(claim_id)
        previous_candidates = previous.get("candidates", []) if previous else []
        annotated = self._annotate_deltas(normalized_candidates, previous_candidates)

        created_at = _utcnow()
        run_id = self._run_id(created_at)
        summary = self._build_summary(annotated)
        meta = dict(metadata or {})

        candidates_key = f"evidence/{claim_id}/{run_id}.json"
        candidates_bytes = json.dumps(
            annotated,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        object_store_s3.put_bytes(
            candidates_key,
            candidates_bytes,
            content_type="application/json",
        )

        depth = max(1, int(getattr(self.settings, "EVIDENCE_HISTORY_DEPTH", 5)))
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO evidence_runs(
                      run_id,
                      project_id,
                      created_by_user_id,
                      claim_id,
                      created_at,
                      note,
                      summary_json,
                      metadata_json,
                      candidates_object_key,
                      lock_state_json
                    )
                    VALUES (
                      %s,
                      'default',
                      'local',
                      %s,
                      now(),
                      %s,
                      %s::jsonb,
                      %s::jsonb,
                      %s,
                      '{}'::jsonb
                    )
                    """,
                    (
                        run_id,
                        str(claim_id),
                        str(meta.get("note") or "") or None,
                        json.dumps(summary, ensure_ascii=True),
                        json.dumps(meta, ensure_ascii=True),
                        candidates_key,
                    ),
                )

                # Trim history rows (best-effort). Keep newest N.
                cur.execute(
                    """
                    SELECT run_id, candidates_object_key
                    FROM evidence_runs
                    WHERE claim_id=%s
                    ORDER BY created_at DESC
                    OFFSET %s
                    """,
                    (str(claim_id), int(depth)),
                )
                stale = cur.fetchall() or []
                if stale:
                    cur.execute(
                        """
                        DELETE FROM evidence_runs
                        WHERE claim_id=%s
                          AND run_id = ANY(%s)
                        """,
                        (
                            str(claim_id),
                            [str(row[0]) for row in stale if row and row[0]],
                        ),
                    )

        return {
            "claim_id": claim_id,
            "run_id": run_id,
            "created_at": created_at,
            "metadata": meta,
            "summary": summary,
            "candidates": annotated,
        }

    def latest_run(self, claim_id: str) -> dict | None:
        if not claim_id:
            return None
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      run_id,
                      created_at,
                      summary_json,
                      metadata_json,
                      candidates_object_key
                    FROM evidence_runs
                    WHERE claim_id=%s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (str(claim_id),),
                )
                row = cur.fetchone()
                if not row:
                    return None

        candidates: list[dict] = []
        key = str(row[4] or "").strip()
        if key:
            try:
                raw = object_store_s3.get_bytes(key)
                data = json.loads(raw.decode("utf-8"))
                candidates = data if isinstance(data, list) else []
            except Exception:
                candidates = []

        created_at = (
            row[1].isoformat().replace("+00:00", "Z")
            if getattr(row[1], "isoformat", None)
            else str(row[1])
        )
        summary = row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}")
        meta = row[3] if isinstance(row[3], dict) else json.loads(row[3] or "{}")
        return {
            "claim_id": str(claim_id),
            "run_id": str(row[0]),
            "created_at": created_at,
            "metadata": meta,
            "summary": summary,
            "candidates": candidates,
        }

    def history(self, claim_id: str) -> list[dict]:
        if not claim_id:
            return []
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT run_id, created_at, summary_json, metadata_json
                    FROM evidence_runs
                    WHERE claim_id=%s
                    ORDER BY created_at DESC
                    """,
                    (str(claim_id),),
                )
                rows = cur.fetchall() or []
        out: list[dict] = []
        for row in rows:
            created_at = (
                row[1].isoformat().replace("+00:00", "Z")
                if getattr(row[1], "isoformat", None)
                else str(row[1])
            )
            summary = row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}")
            meta = row[3] if isinstance(row[3], dict) else json.loads(row[3] or "{}")
            out.append(
                {
                    "claim_id": str(claim_id),
                    "run_id": str(row[0]),
                    "created_at": created_at,
                    "metadata": meta,
                    "summary": summary,
                }
            )
        return out

    def _normalize_candidate(self, candidate: Any) -> dict:
        if hasattr(candidate, "id") and hasattr(candidate, "claim_id"):
            try:
                return serializers.serialize_candidate(candidate)
            except Exception:
                pass
        if hasattr(candidate, "to_dict") and callable(candidate.to_dict):
            return candidate.to_dict()
        if isinstance(candidate, dict):
            return deepcopy(candidate)
        raise TypeError("Unsupported candidate payload")

    def _annotate_deltas(
        self, candidates: Sequence[dict], previous_candidates: Sequence[dict]
    ) -> list[dict]:
        prev_index = {
            cand.get("id"): idx for idx, cand in enumerate(previous_candidates)
        }
        prev_scores = {
            cand.get("id"): _score(cand)
            for cand in previous_candidates
            if cand.get("id") is not None
        }
        annotated: list[dict] = []
        for idx, candidate in enumerate(candidates):
            data = deepcopy(candidate)
            data.setdefault("scores", {})
            data["scores"].setdefault("position", idx + 1)
            prev_idx = prev_index.get(data.get("id"))
            prev_pos = (prev_idx + 1) if prev_idx is not None else None
            current_pos = int(data["scores"].get("position") or idx + 1)
            rank_change = None if prev_pos is None else prev_pos - current_pos
            score_delta = None
            if prev_pos is not None:
                score_delta = _round_score(
                    (_score(data) or 0.0)
                    - (prev_scores.get(data.get("id"), 0.0) or 0.0)
                )

            status: str
            demotion_reason: str | None = None
            if prev_pos is None:
                status = "new"
            elif rank_change is None or rank_change == 0:
                status = "stable"
            elif rank_change > 0:
                status = "promoted"
            else:
                status = "demoted"
                demotion_reason = "rerank-adjustment"

            diversity_note = _diversity_note(data.get("label"))
            data["delta"] = {
                "rank_change": rank_change,
                "score_delta": score_delta,
                "status": status,
                "demotion_reason": demotion_reason,
                "diversity_note": diversity_note,
            }
            annotated.append(data)
        return annotated

    def _build_summary(self, candidates: Sequence[dict]) -> dict:
        labels = Counter(
            (cand.get("label") or "unknown").lower() for cand in candidates
        )
        return {"total": len(candidates), "label_counts": dict(labels)}

    @staticmethod
    def _run_id(created_at: str) -> str:
        compact = created_at.replace("-", "").replace(":", "").replace(".", "")
        return f"{compact}-{uuid4().hex[:8]}"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _score(candidate: dict | None) -> float | None:
    if not candidate:
        return None
    scores = candidate.get("scores") or {}
    combined = scores.get("combined")
    try:
        return float(combined) if combined is not None else None
    except (TypeError, ValueError):
        return None


def _round_score(value: float) -> float:
    return round(value, 4)


def _diversity_note(label: Any) -> str | None:
    label_text = (label or "").lower()
    if label_text == "contradicts":
        return "contradiction-slot"
    if label_text == "neutral":
        return "neutral-coverage"
    return None


__all__ = ["EvidenceRunStore"]
