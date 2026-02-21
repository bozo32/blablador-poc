"""Phase 10 pipeline stage contract models (schema v1).

Every stage artifact is a single JSON envelope persisted at:
  pipeline/{run_id}/{stage}.json

The envelope is shared across stages and intentionally minimal so downstream
components can swap implementations without changing callers.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Mapping, Optional
from uuid import UUID

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

StageName = Literal[
    "extract",
    "citespans",
    "retrieval",
    "filter",
    "rerank",
    "nli",
    "assessment",
]
STAGES: tuple[str, ...] = (
    "extract",
    "citespans",
    "retrieval",
    "filter",
    "rerank",
    "nli",
    "assessment",
)


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def artifact_type_for(stage: str, schema_version: int) -> str:
    s = str(stage or "").strip()
    if not s:
        raise ValueError("stage is required")
    v = int(schema_version)
    if v <= 0:
        raise ValueError("schema_version must be >= 1")
    return f"contracts/{s}@v{v}"


def stage_object_key(run_id: str, stage: str) -> str:
    rid = str(run_id or "").strip()
    st = str(stage or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    if not st:
        raise ValueError("stage is required")
    return f"pipeline/{rid}/{st}.json"


def deterministic_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _norm_ws(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def span_id_for(
    *,
    work_id: str,
    kind: str,
    quote: Dict[str, Any],
    window_fingerprint: str | None = None,
) -> str:
    """Deterministic span id compatible with backend/span_graph_store.py.

    Source of truth is SpanGraphStore's `_span_id()` algorithm:
      sha256("|".join([anchor_id, kind, exact, prefix, suffix, fp]))

    Here we use `work_id` as `anchor_id` and expect `quote` to have
    exact/prefix/suffix.
    """
    anchor_id = _norm_ws(work_id)
    k = _norm_ws(kind)
    if not anchor_id:
        raise ValueError("work_id is required")
    if not k:
        raise ValueError("kind is required")
    sel = quote or {}
    exact = _norm_ws(sel.get("exact"))
    prefix = _norm_ws(sel.get("prefix"))
    suffix = _norm_ws(sel.get("suffix"))
    fp = _norm_ws(window_fingerprint)
    raw = "|".join([anchor_id, k, exact, prefix, suffix, fp])
    return f"span:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def candidate_id_for(
    *,
    run_id: str,
    stage: str,
    target_id: str,
    source_doc_id: str,
    selector: Dict[str, Any],
    window_fingerprint: str | None = None,
) -> str:
    """Deterministic candidate id for retrieval/filter/rerank/nli stages."""
    rid = str(run_id or "").strip()
    st = str(stage or "").strip()
    tid = str(target_id or "").strip()
    sdid = str(source_doc_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    if not st:
        raise ValueError("stage is required")
    if not tid:
        raise ValueError("target_id is required")
    if not sdid:
        raise ValueError("source_doc_id is required")
    if not isinstance(selector, dict) or not selector:
        raise ValueError("selector is required")

    canonical: dict[str, Any] = {
        "run_id": rid,
        "stage": st,
        "target_id": tid,
        "source_doc_id": sdid,
        "selector": selector,
    }
    if window_fingerprint is not None and str(window_fingerprint).strip():
        canonical["window_fingerprint"] = str(window_fingerprint).strip()
    raw = deterministic_json_bytes(canonical)
    return hashlib.sha256(raw).hexdigest()


class PipelineArtifactError(BaseModel):
    code: str
    message: str
    detail: Any | None = None
    retryable: bool = False


class PipelineArtifactComponent(BaseModel):
    id: str
    version: str | None = None
    config_hash: str | None = None


class Caps(BaseModel):
    candidates: int = 200
    rerank: int = 50
    nli: int = 12


class QuoteSelector(BaseModel):
    exact: str = ""
    prefix: str = ""
    suffix: str = ""


class CitationAnchor(BaseModel):
    citation_index: int
    reference_id: str | None = None
    target_id: str | None = None
    callout: str | None = None
    window_text: str | None = None
    sentence_id: str | None = None


class ExtractData(BaseModel):
    structured_doc: Dict[str, Any] = Field(default_factory=dict)
    citation_anchors: List[CitationAnchor] = Field(default_factory=list)


class CiteSpanAnchor(BaseModel):
    citation_index: int
    span_id: str
    selectors: Dict[str, Any] = Field(default_factory=dict)
    window_fingerprint: str | None = None


class CiteSpansTarget(BaseModel):
    anchors: List[CiteSpanAnchor] = Field(default_factory=list)


class CiteSpansData(BaseModel):
    by_target: Dict[str, CiteSpansTarget] = Field(default_factory=dict)


class Candidate(BaseModel):
    candidate_id: str
    source_doc_id: str
    selector: Dict[str, Any] = Field(default_factory=dict)
    text: str
    scores: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class CandidateTargetGroup(BaseModel):
    candidates: List[Candidate] = Field(default_factory=list)


class CandidateStageData(BaseModel):
    by_target: Dict[str, CandidateTargetGroup] = Field(default_factory=dict)


AssessmentRollupLabel = Literal[
    "supports",
    "contradicts",
    "inconsistent",
    "silent",
]


class AssessmentByTarget(BaseModel):
    label: str | None = None
    evidence_spans: List[Dict[str, Any]] = Field(default_factory=list)


class AssessmentData(BaseModel):
    reviewer_uid: str
    scope_id: str
    citing_doc_id: str
    assessed_at: str
    rollup_label: AssessmentRollupLabel | None = None
    by_target: Dict[str, AssessmentByTarget] = Field(default_factory=dict)
    judgment_snapshot: Dict[str, Any] = Field(default_factory=dict)


ArtifactStatus = Literal["complete", "partial", "error"]


class PipelineArtifactEnvelope(BaseModel):
    schema_version: int = 1
    artifact_type: str
    run_id: str
    stage: str
    work_id: str = Field(validation_alias=AliasChoices("work_id", "doc_id"))
    created_at: str = Field(default_factory=_utcnow_z)
    status: ArtifactStatus
    warnings: List[str] = Field(default_factory=list)
    error: Optional[PipelineArtifactError] = None
    caps: Caps = Field(default_factory=Caps)
    component: Optional[PipelineArtifactComponent] = None
    input_fingerprint: str
    data: Any = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, v: str) -> str:
        val = str(v or "").strip()
        if not val:
            raise ValueError("run_id is required")
        try:
            UUID(val)
        except Exception as exc:
            raise ValueError("run_id must be a UUID string") from exc
        return val

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: int) -> int:
        n = int(v)
        if n != 1:
            raise ValueError("schema_version must be 1 for v1 contracts")
        return n

    @field_validator("artifact_type")
    @classmethod
    def _validate_artifact_type_present(cls, v: str) -> str:
        at = str(v or "").strip()
        if not at:
            raise ValueError("artifact_type is required")
        return at

    @model_validator(mode="after")
    def _validate_artifact_type_matches(self):
        want = artifact_type_for(str(self.stage), int(self.schema_version))
        if str(self.artifact_type or "").strip() != want:
            raise ValueError(f"artifact_type must be '{want}'")
        return self

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, v: str) -> str:
        val = str(v or "").strip()
        if not val:
            raise ValueError("created_at is required")
        if not val.endswith("Z"):
            raise ValueError("created_at must be a Z timestamp")
        return val


class ExtractArtifact(PipelineArtifactEnvelope):
    stage: Literal["extract"]
    data: ExtractData


class CiteSpansArtifact(PipelineArtifactEnvelope):
    stage: Literal["citespans"]
    data: CiteSpansData


class RetrievalArtifact(PipelineArtifactEnvelope):
    stage: Literal["retrieval"]
    data: CandidateStageData


class FilterArtifact(PipelineArtifactEnvelope):
    stage: Literal["filter"]
    data: CandidateStageData


class RerankArtifact(PipelineArtifactEnvelope):
    stage: Literal["rerank"]
    data: CandidateStageData


class NliArtifact(PipelineArtifactEnvelope):
    stage: Literal["nli"]
    data: CandidateStageData


class AssessmentArtifact(PipelineArtifactEnvelope):
    stage: Literal["assessment"]
    data: AssessmentData


STAGE_MODEL_BY_NAME = {
    "extract": ExtractArtifact,
    "citespans": CiteSpansArtifact,
    "retrieval": RetrievalArtifact,
    "filter": FilterArtifact,
    "rerank": RerankArtifact,
    "nli": NliArtifact,
    "assessment": AssessmentArtifact,
}


def validate_v1(payload: Mapping[str, Any]) -> dict:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    stage = str(payload.get("stage") or "").strip()
    model = STAGE_MODEL_BY_NAME.get(stage)
    if model is None:
        raise ValueError(f"Unknown stage: {stage or '(missing)'}")
    try:
        parsed = model.model_validate(dict(payload))
    except ValidationError as exc:
        raise ValueError(f"Invalid v1 contract for stage '{stage}': {exc}") from exc
    return parsed.model_dump(mode="json", by_alias=False, exclude_none=True)
