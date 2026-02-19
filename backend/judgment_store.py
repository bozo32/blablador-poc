"""Spine-backed persistence for per-claim reviewer judgments.

Phase 09.3: judgments persist in Postgres (no `data/judgments/**`).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Iterable, Literal, Optional
from uuid import uuid4

from pydantic import ValidationError

from backend.db.pg import connect
from backend.schemas import JudgmentPayload, JudgmentUpsertRequest
from backend.settings import AppSettings, settings as app_settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JudgmentStore:
    def __init__(self, *, settings: AppSettings = app_settings) -> None:
        """Create a judgment store backed by Postgres."""
        self.settings = settings

    def read(
        self, claim_id: str, reviewer_uid: str = "default"
    ) -> Optional[JudgmentPayload]:
        cid = str(claim_id or "").strip()
        rid = str(reviewer_uid or "default").strip() or "default"
        if not cid:
            return None
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      reviewer_uid,
                      updated_at,
                      status,
                      verdict,
                      notes_json,
                      validation_json,
                      doc_id,
                      citation_index,
                      target_id,
                      sentence_id,
                      callout,
                      reference_id,
                      doi,
                      author,
                      year,
                      claim_text,
                      cited_work_id,
                      citation_anchor,
                      span_selectors
                    FROM judgments
                    WHERE claim_id=%s AND reviewer_uid=%s
                    """,
                    (cid, rid),
                )
                row = cur.fetchone()
                if not row:
                    return None
        payload = {
            "claim_id": cid,
            "reviewer_uid": str(row[0] or "default"),
            "updated_at": row[1].isoformat().replace("+00:00", "Z")
            if row[1]
            else _now(),
            "status": row[2],
            "verdict": row[3],
            "notes": row[4]
            if isinstance(row[4], dict)
            else json.loads(row[4] or "null"),
            "validation": row[5]
            if isinstance(row[5], dict)
            else json.loads(row[5] or "null"),
            "doc_id": row[6],
            "citation_index": row[7],
            "target_id": row[8],
            "sentence_id": row[9],
            "callout": row[10],
            "reference_id": row[11],
            "doi": row[12],
            "author": row[13],
            "year": str(row[14]) if row[14] is not None else None,
            "claim_text": row[15],
            "cited_work_id": row[16],
            "citation_anchor": row[17]
            if isinstance(row[17], dict)
            else json.loads(row[17] or "null"),
            "span_selectors": row[18]
            if isinstance(row[18], dict)
            else json.loads(row[18] or "null"),
        }
        return JudgmentPayload.model_validate(payload)

    def upsert(
        self, claim_id: str, judgment: dict | JudgmentUpsertRequest
    ) -> JudgmentPayload:
        cid = str(claim_id or "").strip()
        if not cid:
            raise ValueError("claim_id is required")
        request = (
            judgment
            if isinstance(judgment, JudgmentUpsertRequest)
            else JudgmentUpsertRequest.model_validate(judgment)
        )
        reviewer = (
            str(getattr(request, "reviewer_uid", "default") or "default").strip()
            or "default"
        )
        stored = JudgmentPayload(
            claim_id=cid,
            reviewer_uid=reviewer,
            updated_at=_now(),
            status=request.status,
            verdict=request.verdict,
            notes=request.notes,
            validation=getattr(request, "validation", None),
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
            cited_work_id=getattr(request, "cited_work_id", None),
            citation_anchor=getattr(request, "citation_anchor", None),
            span_selectors=getattr(request, "span_selectors", None),
        )

        notes_json = (
            stored.notes.model_dump(mode="json") if stored.notes is not None else None
        )
        validation_json = (
            stored.validation.model_dump(mode="json")
            if getattr(stored, "validation", None) is not None
            else None
        )
        citation_anchor_json = (
            stored.citation_anchor
            if isinstance(getattr(stored, "citation_anchor", None), dict)
            else getattr(stored, "citation_anchor", None)
        )
        span_selectors_json = (
            stored.span_selectors
            if isinstance(getattr(stored, "span_selectors", None), dict)
            else getattr(stored, "span_selectors", None)
        )

        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO judgments(
                      judgment_id,
                      project_id,
                      updated_by_user_id,
                      claim_id,
                      reviewer_uid,
                      updated_at,
                      status,
                      verdict,
                      notes_json,
                      doc_id,
                      citation_index,
                      target_id,
                      sentence_id,
                      callout,
                      reference_id,
                      doi,
                      author,
                      year,
                      claim_text,
                      cited_work_id,
                      citation_anchor,
                      span_selectors,
                      validation_json
                    )
                    VALUES (
                      %s,
                      'default',
                      'local',
                      %s,
                      %s,
                      now(),
                      %s,
                      %s,
                      %s::jsonb,
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
                      %s::jsonb,
                      %s::jsonb,
                      %s::jsonb
                    )
                    ON CONFLICT(claim_id, reviewer_uid) DO UPDATE
                      SET updated_at=excluded.updated_at,
                          updated_by_user_id=excluded.updated_by_user_id,
                          status=excluded.status,
                          verdict=excluded.verdict,
                          notes_json=excluded.notes_json,
                          doc_id=excluded.doc_id,
                          citation_index=excluded.citation_index,
                          target_id=excluded.target_id,
                          sentence_id=excluded.sentence_id,
                          callout=excluded.callout,
                          reference_id=excluded.reference_id,
                          doi=excluded.doi,
                          author=excluded.author,
                          year=excluded.year,
                          claim_text=excluded.claim_text,
                          cited_work_id=excluded.cited_work_id,
                          citation_anchor=excluded.citation_anchor,
                          span_selectors=excluded.span_selectors,
                          validation_json=excluded.validation_json
                    """,
                    (
                        str(uuid4()),
                        cid,
                        reviewer,
                        str(stored.status),
                        str(stored.verdict) if stored.verdict is not None else None,
                        json.dumps(notes_json, ensure_ascii=True)
                        if notes_json is not None
                        else json.dumps(None),
                        stored.doc_id,
                        stored.citation_index,
                        stored.target_id,
                        stored.sentence_id,
                        stored.callout,
                        stored.reference_id,
                        stored.doi,
                        stored.author,
                        stored.year,
                        stored.claim_text,
                        getattr(stored, "cited_work_id", None),
                        json.dumps(citation_anchor_json, ensure_ascii=True)
                        if citation_anchor_json is not None
                        else json.dumps(None),
                        json.dumps(span_selectors_json, ensure_ascii=True)
                        if span_selectors_json is not None
                        else json.dumps(None),
                        json.dumps(validation_json, ensure_ascii=True)
                        if validation_json is not None
                        else json.dumps(None),
                    ),
                )
        return stored

    def validate(self, judgment: dict) -> JudgmentUpsertRequest:
        try:
            return JudgmentUpsertRequest.model_validate(judgment)
        except ValidationError:
            raise

    def list_all(self) -> list[JudgmentPayload]:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT claim_id, reviewer_uid
                    FROM judgments
                    ORDER BY claim_id, reviewer_uid
                    """
                )
                rows = cur.fetchall() or []
        out: list[JudgmentPayload] = []
        for cid, rid in rows:
            item = self.read(str(cid), reviewer_uid=str(rid))
            if item is not None:
                out.append(item)
        return out

    def list_for_claim(self, claim_id: str) -> list[JudgmentPayload]:
        cid = str(claim_id or "").strip()
        if not cid:
            return []
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT reviewer_uid
                    FROM judgments
                    WHERE claim_id=%s
                    ORDER BY reviewer_uid
                    """,
                    (cid,),
                )
                rows = cur.fetchall() or []
        out: list[JudgmentPayload] = []
        for (rid,) in rows:
            item = self.read(cid, reviewer_uid=str(rid))
            if item is not None:
                out.append(item)
        return out

    def list_filtered(
        self,
        *,
        status: Literal["final", "draft", "all"],
        doc_id: str | None = None,
    ) -> list[JudgmentPayload]:
        items = self.list_all()
        if doc_id is not None:
            items = [item for item in items if item.doc_id == doc_id]
        if status != "all":
            items = [item for item in items if item.status == status]
        return items

    def export_claims(
        self,
        *,
        include_drafts: bool,
        mode: Literal["core", "verbose"],
        format: Literal["json", "csv"],
    ) -> bytes:
        judgments = self.list_all()
        if not include_drafts:
            judgments = [j for j in judgments if j.status == "final"]
        judgments.sort(key=lambda item: item.claim_id)

        def core_row(j: JudgmentPayload) -> dict[str, Any]:
            return {
                "claim_id": j.claim_id,
                "reviewer_uid": j.reviewer_uid,
                "status": j.status,
                "verdict": j.verdict,
                "claim_text": j.claim_text,
            }

        def verbose_row(j: JudgmentPayload) -> dict[str, Any]:
            notes = j.notes
            validation = getattr(j, "validation", None)
            notes_payload = None if notes is None else notes.model_dump(mode="json")
            validation_payload = (
                None if validation is None else validation.model_dump(mode="json")
            )
            return {
                **core_row(j),
                "updated_at": j.updated_at,
                "notes": notes_payload,
                "validation": validation_payload,
                "sentence_id": j.sentence_id,
                "reference_id": j.reference_id,
                "doi": j.doi,
                "author": j.author,
                "year": j.year,
                "doc_id": j.doc_id,
                "citation_index": j.citation_index,
                "target_id": j.target_id,
                "callout": j.callout,
                "cited_work_id": getattr(j, "cited_work_id", None),
                "citation_anchor": getattr(j, "citation_anchor", None),
                "span_selectors": getattr(j, "span_selectors", None),
            }

        if mode == "core":
            rows = [core_row(j) for j in judgments]
        elif format == "json":
            rows = [verbose_row(j) for j in judgments]
        else:

            def verbose_csv_row(j: JudgmentPayload) -> dict[str, Any]:
                notes = j.notes
                validation = getattr(j, "validation", None)
                return {
                    **core_row(j),
                    "updated_at": j.updated_at,
                    "rationale": getattr(notes, "rationale", None),
                    "caveats": getattr(notes, "caveats", None),
                    "followups": getattr(notes, "followups", None),
                    "source_valid": getattr(validation, "source_valid", None),
                    "source_valid_comment": getattr(
                        validation, "source_valid_comment", None
                    ),
                    "source_relevant": getattr(validation, "source_relevant", None),
                    "source_relevant_comment": getattr(
                        validation, "source_relevant_comment", None
                    ),
                    "sentence_id": j.sentence_id,
                    "reference_id": j.reference_id,
                    "doi": j.doi,
                    "author": j.author,
                    "year": j.year,
                    "doc_id": j.doc_id,
                    "citation_index": j.citation_index,
                    "target_id": j.target_id,
                    "callout": j.callout,
                }

            rows = [verbose_csv_row(j) for j in judgments]

        if format == "json":
            return (
                json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
            ).encode("utf-8")

        headers = (
            list(rows[0].keys())
            if rows
            else list(core_row(JudgmentPayload(claim_id="x")).keys())
        )
        sio = StringIO()
        writer = csv.DictWriter(sio, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return sio.getvalue().encode("utf-8")

    def export_callouts(
        self,
        *,
        include_drafts: bool,
        mode: Literal["core", "verbose"],
        format: Literal["json", "csv"],
    ) -> bytes:
        judgments = self.list_all()
        if not include_drafts:
            judgments = [j for j in judgments if j.status == "final"]

        def _null_key(value: Any, *, null_value: Any) -> Any:
            return null_value if value is None else value

        def _group_key(j: JudgmentPayload) -> tuple[Any, Any, Any]:
            return (
                _null_key(j.doc_id, null_value=""),
                _null_key(j.citation_index, null_value=-1),
                _null_key(j.target_id, null_value=""),
            )

        def _claim_core(j: JudgmentPayload) -> dict[str, Any]:
            return {
                "claim_id": j.claim_id,
                "reviewer_uid": j.reviewer_uid,
                "status": j.status,
                "verdict": j.verdict,
                "claim_text": j.claim_text,
            }

        def _claim_verbose(j: JudgmentPayload) -> dict[str, Any]:
            notes = j.notes
            notes_payload = None if notes is None else notes.model_dump(mode="json")
            return {
                **_claim_core(j),
                "notes": notes_payload,
                "sentence_id": j.sentence_id,
                "reference_id": j.reference_id,
                "doi": j.doi,
                "author": j.author,
                "year": j.year,
                "updated_at": j.updated_at,
            }

        groups: dict[tuple[Any, Any, Any], list[JudgmentPayload]] = {}
        for j in judgments:
            groups.setdefault(_group_key(j), []).append(j)

        def _sorted_claims(values: Iterable[JudgmentPayload]) -> list[JudgmentPayload]:
            return sorted(values, key=lambda item: (item.claim_id, item.reviewer_uid))

        group_items: list[dict[str, Any]] = []
        for key in sorted(groups.keys()):
            claims = _sorted_claims(groups[key])
            selected_callout = None
            for claim in claims:
                if claim.callout is not None:
                    selected_callout = claim.callout
                    break
            payload_claims = [
                _claim_core(c) if mode == "core" else _claim_verbose(c) for c in claims
            ]
            sample = claims[0] if claims else None
            group_items.append(
                {
                    "doc_id": None if sample is None else sample.doc_id,
                    "citation_index": None if sample is None else sample.citation_index,
                    "target_id": None if sample is None else sample.target_id,
                    "callout": selected_callout,
                    "claims": payload_claims,
                }
            )

        if format == "json":
            return (
                json.dumps(group_items, indent=2, sort_keys=True, ensure_ascii=True)
                + "\n"
            ).encode("utf-8")

        # CSV: flatten.
        flat_rows: list[dict[str, Any]] = []
        for group in group_items:
            for claim in group.get("claims") or []:
                flat_rows.append(
                    {
                        "doc_id": group.get("doc_id"),
                        "citation_index": group.get("citation_index"),
                        "target_id": group.get("target_id"),
                        "callout": group.get("callout"),
                        **(claim if isinstance(claim, dict) else {}),
                    }
                )
        headers = (
            list(flat_rows[0].keys())
            if flat_rows
            else [
                "doc_id",
                "citation_index",
                "target_id",
                "callout",
                "claim_id",
                "reviewer_uid",
                "status",
                "verdict",
                "claim_text",
            ]
        )
        sio = StringIO()
        writer = csv.DictWriter(sio, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)
        return sio.getvalue().encode("utf-8")


judgment_store = JudgmentStore(settings=app_settings)


__all__ = ["JudgmentStore", "judgment_store"]
