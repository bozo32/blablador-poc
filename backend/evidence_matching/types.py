"""Typed data structures for evidence matching primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, MutableMapping


class EvidenceLabel(str, Enum):
    """Supported entailment labels for evidence candidates."""

    ENTAILS = "entails"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class Provenance(str, Enum):
    """How a candidate was sourced."""

    CITED = "cited"
    HEURISTIC = "heuristic"


def _ensure_label(value: EvidenceLabel | str) -> EvidenceLabel:
    if isinstance(value, EvidenceLabel):
        return value
    try:
        return EvidenceLabel(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported evidence label: {value}") from exc


def _ensure_provenance(value: Provenance | str) -> Provenance:
    if isinstance(value, Provenance):
        return value
    try:
        return Provenance(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported provenance value: {value}") from exc


def _normalize_bbox(raw: Iterable[Iterable[float]] | None) -> list[list[float]]:
    if not raw:
        return []
    box_list: list[list[float]] = []
    for box in raw:
        coords = list(box)
        if len(coords) != 4:
            continue
        box_list.append([float(value) for value in coords])
    return box_list


@dataclass
class CandidateSpan:
    """Atomic snippet pulled from attachment sentences."""

    sentence_id: str
    text: str
    page: str
    section: str
    tei_ids: tuple[str, ...] = field(default_factory=tuple)
    bbox: list[list[float]] = field(default_factory=list)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sentence(
        cls, sentence: Mapping[str, Any], *, default_section: str = "Body"
    ) -> "CandidateSpan":
        """Build a span from a persisted attachment sentence row."""
        sentence_id = str(sentence.get("sentence_id") or sentence.get("id"))
        text = (sentence.get("text") or "").strip()
        if not sentence_id:
            raise ValueError("Sentence is missing a stable identifier")
        page = cls._normalize_page(sentence.get("page"))
        section = str(sentence.get("section") or default_section)
        tei_ids_raw = sentence.get("tei_ids") or sentence.get("tei_id") or []
        if isinstance(tei_ids_raw, str):
            tei_ids = (tei_ids_raw,)
        else:
            tei_ids = tuple(str(tid) for tid in tei_ids_raw) or (sentence_id,)
        bbox = _normalize_bbox(sentence.get("bbox"))
        embedding = sentence.get("embedding")
        metadata = {
            "page": page,
            "section": section,
            "position": sentence.get("position"),
        }
        if tei_ids:
            metadata["tei_ids"] = list(tei_ids)
        return cls(
            sentence_id=sentence_id,
            text=text,
            page=page,
            section=section,
            tei_ids=tei_ids,
            bbox=bbox,
            embedding=list(embedding) if isinstance(embedding, list) else embedding,
            metadata=metadata,
        )

    @staticmethod
    def _normalize_page(value: Any) -> str:
        if value is None:
            return "Unknown"
        text = str(value).strip()
        return text or "Unknown"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "sentence_id": self.sentence_id,
            "text": self.text,
            "page": self.page,
            "section": self.section,
            "tei_ids": list(self.tei_ids),
            "bbox": self.bbox,
        }
        if self.embedding is not None:
            payload["embedding"] = list(self.embedding)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass
class RankScores:
    """Aggregated per-stage ranking scores."""

    bm25: float | None = None
    faiss: float | None = None
    sbert: float | None = None
    cross_encoder: float | None = None
    colbert: float | None = None
    nli: float | None = None
    combined: float | None = None
    position: int | None = None

    def to_dict(self) -> dict[str, float | int]:
        data: dict[str, float | int] = {}
        for field_name in (
            "bm25",
            "faiss",
            "sbert",
            "cross_encoder",
            "colbert",
            "nli",
            "combined",
            "position",
        ):
            value = getattr(self, field_name)
            if value is not None:
                data[field_name] = value
        return data

    def update(self, **scores: float | int | None) -> None:
        for key, value in scores.items():
            if not hasattr(self, key):
                raise AttributeError(f"RankScores has no field named '{key}'")
            setattr(self, key, value)


@dataclass
class EvidenceCandidate:
    """Normalized payload returned by the evidence pipeline."""

    id: str
    claim_id: str
    attachment_id: str
    text: str
    label: EvidenceLabel
    scores: RankScores
    spans: list[CandidateSpan] = field(default_factory=list)
    provenance: Provenance = Provenance.HEURISTIC
    badges: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    highlights: list[dict[str, Any]] = field(default_factory=list)
    token_saliencies: list[float] | None = None

    def __post_init__(self) -> None:
        """Normalize enums and deduplicate badge payloads."""
        self.label = _ensure_label(self.label)
        self.provenance = _ensure_provenance(self.provenance)
        self.badges = sorted(set(self.badges))

    @property
    def primary_page(self) -> str:
        return self.metadata.get("page") or (
            self.spans[0].page if self.spans else "Unknown"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "attachment_id": self.attachment_id,
            "text": self.text,
            "label": self.label.value,
            "provenance": self.provenance.value,
            "scores": self.scores.to_dict(),
            "spans": [span.to_dict() for span in self.spans],
            "badges": list(self.badges),
            "metadata": dict(self.metadata),
            "highlights": list(self.highlights),
            "token_saliencies": list(self.token_saliencies)
            if self.token_saliencies is not None
            else None,
        }

    @classmethod
    def from_window(
        cls,
        window: Mapping[str, Any],
        *,
        label: EvidenceLabel | str = EvidenceLabel.NEUTRAL,
        provenance: Provenance | str = Provenance.HEURISTIC,
        scores: RankScores | None = None,
        metadata: MutableMapping[str, Any] | None = None,
    ) -> "EvidenceCandidate":
        """Build a candidate from a window dict."""
        spans_data = window.get("spans") or []
        spans: list[CandidateSpan]
        if spans_data and isinstance(spans_data[0], CandidateSpan):
            spans = list(spans_data)
        else:
            spans = [CandidateSpan.from_sentence(row) for row in spans_data]
        candidate_metadata = dict(metadata or window.get("metadata") or {})
        if spans and "page" not in candidate_metadata:
            candidate_metadata["page"] = spans[0].page
        if spans and "section" not in candidate_metadata:
            candidate_metadata["section"] = spans[0].section
        return cls(
            id=str(window.get("window_id") or window.get("id")),
            claim_id=str(window.get("claim_id")),
            attachment_id=str(window.get("attachment_id")),
            text=str(window.get("text") or ""),
            label=label,
            provenance=provenance,
            scores=scores or RankScores(),
            spans=spans,
            metadata=candidate_metadata,
            badges=list(window.get("badges", [])),
            highlights=list(window.get("highlights", [])),
            token_saliencies=window.get("token_saliencies"),
        )


__all__ = [
    "CandidateSpan",
    "EvidenceCandidate",
    "EvidenceLabel",
    "Provenance",
    "RankScores",
]
