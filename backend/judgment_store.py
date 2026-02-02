"""On-disk persistence for per-claim reviewer judgments."""

from __future__ import annotations

import json
import re
import csv
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

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

    def list_all(self) -> list[JudgmentPayload]:
        """List all stored judgments."""
        root = self.root_dir
        if not root.exists():
            return []
        items: list[JudgmentPayload] = []
        for path in sorted(root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            items.append(JudgmentPayload.model_validate(payload))
        items.sort(key=lambda item: item.claim_id)
        return items

    def list_filtered(
        self,
        *,
        status: Literal["final", "draft", "all"],
        doc_id: str | None = None,
    ) -> list[JudgmentPayload]:
        """List stored judgments filtered by status and optional doc_id."""
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
        """Export per-claim judgments as JSON or CSV."""
        judgments = self.list_all()
        if not include_drafts:
            judgments = [j for j in judgments if j.status == "final"]
        judgments.sort(key=lambda item: item.claim_id)

        def core_row(j: JudgmentPayload) -> dict[str, Any]:
            return {
                "claim_id": j.claim_id,
                "status": j.status,
                "verdict": j.verdict,
                "claim_text": j.claim_text,
            }

        def verbose_row(j: JudgmentPayload) -> dict[str, Any]:
            notes = j.notes
            return {
                **core_row(j),
                "updated_at": j.updated_at,
                "notes": None if notes is None else notes.model_dump(mode="json"),
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

        if mode == "core":
            rows = [core_row(j) for j in judgments]
        elif format == "json":
            rows = [verbose_row(j) for j in judgments]
        else:

            def verbose_csv_row(j: JudgmentPayload) -> dict[str, Any]:
                notes = j.notes
                return {
                    **core_row(j),
                    "updated_at": j.updated_at,
                    "rationale": getattr(notes, "rationale", None),
                    "caveats": getattr(notes, "caveats", None),
                    "followups": getattr(notes, "followups", None),
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

        if not rows:
            headers = list(core_row(JudgmentPayload(claim_id="x")).keys())
            if mode == "verbose":
                headers = [
                    "claim_id",
                    "status",
                    "verdict",
                    "claim_text",
                    "updated_at",
                    "rationale",
                    "caveats",
                    "followups",
                    "sentence_id",
                    "reference_id",
                    "doi",
                    "author",
                    "year",
                    "doc_id",
                    "citation_index",
                    "target_id",
                    "callout",
                ]
        else:
            headers = list(rows[0].keys())
        buf: list[str] = []
        from io import StringIO

        sio = StringIO()
        writer = csv.DictWriter(sio, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        buf.append(sio.getvalue())
        return "".join(buf).encode("utf-8")

    def export_callouts(
        self,
        *,
        include_drafts: bool,
        mode: Literal["core", "verbose"],
        format: Literal["json", "csv"],
    ) -> bytes:
        """Export judgments grouped per citation callout as JSON or CSV."""
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
                "status": j.status,
                "verdict": j.verdict,
                "claim_text": j.claim_text,
            }

        def _claim_verbose(j: JudgmentPayload) -> dict[str, Any]:
            notes = j.notes
            return {
                **_claim_core(j),
                "notes": None if notes is None else notes.model_dump(mode="json"),
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
            return sorted(values, key=lambda item: item.claim_id)

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

        from io import StringIO

        flat_rows: list[dict[str, Any]] = []
        for group in group_items:
            group_fields = {
                "doc_id": group.get("doc_id"),
                "citation_index": group.get("citation_index"),
                "target_id": group.get("target_id"),
                "callout": group.get("callout"),
            }
            for claim in group.get("claims") or []:
                row = {**group_fields, **(claim or {})}
                if mode == "verbose":
                    notes = row.pop("notes", None) or {}
                    if not isinstance(notes, dict):
                        notes = {}
                    row["rationale"] = notes.get("rationale")
                    row["caveats"] = notes.get("caveats")
                    row["followups"] = notes.get("followups")
                flat_rows.append(row)

        headers = [
            "doc_id",
            "citation_index",
            "target_id",
            "callout",
            "claim_id",
            "status",
            "verdict",
            "claim_text",
        ]
        if mode == "verbose":
            headers += [
                "updated_at",
                "sentence_id",
                "reference_id",
                "doi",
                "author",
                "year",
                "rationale",
                "caveats",
                "followups",
            ]

        sio = StringIO()
        writer = csv.DictWriter(sio, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)
        return sio.getvalue().encode("utf-8")


judgment_store = JudgmentStore(settings=app_settings)


__all__ = ["JudgmentStore", "judgment_store"]
