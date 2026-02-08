"""Loader utilities for attachment-backed evidence windows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from backend import attachment_store
from backend.attachment_store import AttachmentNotFound
from backend.settings import AppSettings, settings as app_settings

from .types import CandidateSpan


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class AttachmentWindow:
    """Normalized slice of attachment sentences used as evidence seeds."""

    window_id: str
    claim_id: str
    attachment_id: str
    text: str
    spans: list[CandidateSpan]
    tokens: list[str]
    tei_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "claim_id": self.claim_id,
            "attachment_id": self.attachment_id,
            "text": self.text,
            "tokens": list(self.tokens),
            "tei_ids": list(self.tei_ids),
            "metadata": dict(self.metadata),
            "spans": [span.to_dict() for span in self.spans],
        }


def load_claim_windows(
    claim_id: str,
    *,
    attachment_ids: Sequence[str] | None = None,
    settings_override: AppSettings | None = None,
    max_windows: int | None = None,
) -> list[AttachmentWindow]:
    """Load deterministic windows for the provided claim.

    When ``attachment_ids`` is ``None`` the loader inspects all attachments that
    belong to ``claim_id`` and have reached the ``matched`` state. The
    ``max_windows`` parameter defaults to ``settings.EVIDENCE_MAX_WINDOWS``.
    """
    cfg = settings_override or app_settings
    window_cap = max_windows or getattr(cfg, "EVIDENCE_MAX_WINDOWS", 500)
    records = _resolve_records(claim_id, attachment_ids)
    windows: list[AttachmentWindow] = []
    for record in records:
        windows.extend(_windows_for_attachment(record, cfg))
        if len(windows) >= window_cap:
            break
    return windows[:window_cap]


def load_attachment_windows(
    *,
    claim_id: str,
    attachment_id: str,
    settings_override: AppSettings | None = None,
) -> list[AttachmentWindow]:
    """Load windows for a single attachment id."""
    record = attachment_store.get_attachment(attachment_id)
    if record is None:
        raise AttachmentNotFound(f"Attachment {attachment_id} not found")
    if record.get("claim_id") != claim_id:
        raise ValueError(
            f"Attachment {attachment_id} does not belong to claim {claim_id}"
        )
    if not attachment_store.is_ready(record):
        return []
    cfg = settings_override or app_settings
    return _windows_for_attachment(record, cfg)


def _resolve_records(
    claim_id: str, attachment_ids: Sequence[str] | None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if attachment_ids:
        for attachment_id in attachment_ids:
            record = attachment_store.get_attachment(attachment_id)
            if record and attachment_store.is_ready(record):
                records.append(record)
        return records

    def _claim_aliases(value: str) -> list[str]:
        canonical = str(value or "").strip()
        if not canonical:
            return []
        aliases = [canonical]
        parts = canonical.split(":")
        if (
            len(parts) == 5
            and parts[0] == "cite"
            and parts[1]
            and str(parts[2]).isdigit()
            and parts[4]
        ):
            legacy = ":".join([parts[0], parts[1], parts[2], parts[4]])
            if legacy and legacy not in aliases:
                aliases.append(legacy)
        return aliases

    seen: set[str] = set()
    for cid in _claim_aliases(claim_id):
        for record in attachment_store.list_attachments(claim_id=cid):
            rid = str(record.get("id") or "")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            if attachment_store.is_ready(record):
                # Preserve the requested claim_id so downstream metadata stays
                # aligned with the evidence run key (even if the attachment was
                # originally placed on a legacy claim id).
                if record.get("claim_id") != claim_id:
                    record = dict(record)
                    record["claim_id"] = claim_id
                records.append(record)
    records.sort(key=lambda rec: rec.get("uploaded_at", ""))
    return records


def _windows_for_attachment(
    record: Mapping[str, Any], cfg: AppSettings
) -> list[AttachmentWindow]:
    attachment_id = str(record["id"])
    claim_id = str(record.get("claim_id"))
    sentences = attachment_store.load_sentences_for_attachment(attachment_id)
    sanitized = _sanitize_sentences(sentences)
    window_size = getattr(
        cfg, "EVIDENCE_WINDOW_SIZE", getattr(cfg, "HYBRID_WINDOW_SIZE", 3)
    )
    stride = getattr(cfg, "EVIDENCE_WINDOW_STRIDE", 1)
    windows: list[AttachmentWindow] = []
    if not sanitized:
        return windows
    idx = 0
    for start in range(0, len(sanitized), stride):
        chunk = sanitized[start : start + window_size]
        if not chunk:
            break
        spans = [CandidateSpan.from_sentence(sentence) for sentence in chunk]
        text = " ".join(span.text for span in spans).strip()
        if not text:
            continue
        window_id = _build_window_id(attachment_id, spans)
        tokens = _simple_tokenize(text)
        tei_ids: list[str] = []
        for span in spans:
            tei_ids.extend(span.tei_ids)
        metadata = {
            "claim_id": claim_id,
            "attachment_id": attachment_id,
            "page": spans[0].page,
            "section": spans[0].section,
            "window_index": idx,
            "attachment_filename": record.get("filename"),
        }
        windows.append(
            AttachmentWindow(
                window_id=window_id,
                claim_id=claim_id,
                attachment_id=attachment_id,
                text=text,
                spans=spans,
                tokens=tokens,
                tei_ids=tuple(tei_ids),
                metadata=metadata,
            )
        )
        idx += 1
    return windows


def _sanitize_sentences(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    clean_rows: list[Mapping[str, Any]] = []
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        payload = dict(row)
        payload["text"] = text
        clean_rows.append(payload)
    clean_rows.sort(key=lambda row: row.get("position", 0))
    return clean_rows


def _simple_tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


def _build_window_id(attachment_id: str, spans: Sequence[CandidateSpan]) -> str:
    first = spans[0].sentence_id
    last = spans[-1].sentence_id
    return f"{attachment_id}:{first}-{last}"


__all__ = [
    "AttachmentWindow",
    "load_claim_windows",
    "load_attachment_windows",
]
