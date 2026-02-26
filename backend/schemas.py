from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class Evidence(BaseModel):
    text: str
    location: str
    assessment: str
    label: str  # "entailment" or "contradiction"
    score: float  # confidence from NLI model
    chunk_id: Optional[str]
    type: Optional[str]


class SegmentResult(BaseModel):
    segment_id: str
    claim: str
    evidence: List[Evidence] = Field(default_factory=list)


class Segment(BaseModel):
    segment_id: str
    claim: str


class RequestSettings(BaseModel):
    pipeline_mode: Optional[str] = Field(
        None, description="Pipeline mode: 'classic' or 'hybrid'"
    )
    embed_model: Optional[str] = Field(
        None, description="Embedding model alias or HF path"
    )
    max_sentences: Optional[int] = Field(
        None,
        ge=1,
        le=10000,
        description="How many FAISS candidates to pull before thresholding",
    )
    faiss_min_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum FAISS similarity score threshold"
    )
    nli_model: Optional[str] = Field(
        None, description="Local NLI model for entailment/contradiction"
    )
    llm_model: Optional[str] = Field(
        None, description="LLM model alias for remote calls"
    )
    nli_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum NLI confidence threshold"
    )
    data_dir: Optional[str] = Field(
        None, description="Path to the folder containing CSV and TEI files"
    )
    api_key: Optional[str] = Field(None, description="API key for Blablador service")
    base_url: Optional[str] = Field(
        None, description="Base URL for the Blablador API endpoint"
    )
    reranker_model: Optional[str] = Field(
        None,
        description="HF path for cross-encoder reranker",
    )
    reranker_top_k: int = Field(
        10, ge=1, description="How many of the FAISS candidates to keep after reranking"
    )

    @field_validator("max_sentences", "faiss_min_score", "nli_threshold")
    def check_not_nan(cls, v):
        import math

        if v is None:
            return v
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("Field must not be NaN or infinite")
        return v


class SentenceResult(BaseModel):
    text: str
    segments: List[SegmentResult]


class CitingPaperResult(BaseModel):
    title: str
    id: str
    sentences: List[SentenceResult]


class CitedPaperResult(BaseModel):
    title: str
    id: str
    doi: str
    citing_papers: List[CitingPaperResult]


class SegmentRequest(BaseModel):
    folder: str
    row_id: str
    citing_title: str
    citing_id: str
    original_sentence: str
    segments: List[Segment]
    settings: RequestSettings


class SentencePayload(BaseModel):
    folder: str
    row_id: str
    citing_title: str
    citing_id: str
    original_sentence: str
    segments: List[Segment]
    # Make settings required so we never have to guard against None downstream
    settings: RequestSettings = Field(
        ...,  # required
        description="Runtime settings for embedding, FAISS, NLI, and LLM models",
    )


class PrebuildRequest(BaseModel):
    folder: str
    embed_model: str = "alias-embeddings"
    max_chunks: int = 256
    faiss_min_score: float = 0.2

    @field_validator("max_chunks", "faiss_min_score")
    def check_not_nan_prebuild(cls, v):
        import math

        if v is None:
            return v
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("Field must not be NaN or infinite")
        return v


class IngestionStage(BaseModel):
    status: str
    extracted_at: Optional[str] = None
    resolved_at: Optional[str] = None
    data: Optional[Any] = None
    payload: Optional[dict] = None


class ExtractionResult(BaseModel):
    status: str
    extracted_at: Optional[str] = None
    data: Optional[dict] = None


class ExtractionResponse(BaseModel):
    document_id: str
    extraction: ExtractionResult


class ResolvedReference(BaseModel):
    reference_id: str
    raw: str
    doi: Optional[str] = None
    title: Optional[str] = None
    publisher: Optional[str] = None
    year: Optional[str] = None
    source: Optional[str] = None
    confidence: Optional[float] = None
    grobid: Optional[dict] = None
    crossref: Optional[dict] = None
    openalex: Optional[dict] = None
    status: Optional[str] = None
    mismatch_reason: Optional[str] = None
    selected_source: Optional[str] = None


class ResolutionResult(BaseModel):
    status: str
    resolved_at: Optional[str] = None
    data: Optional[List[ResolvedReference]] = None


class ResolutionResponse(BaseModel):
    document_id: str
    resolution: ResolutionResult


class ResolutionSelectionRequest(BaseModel):
    selected_source: str


class CitationContextReference(BaseModel):
    id: Optional[str] = None
    raw_reference: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    grobid: Optional[dict] = None


class CitationContextPayload(BaseModel):
    citation_index: int
    target_id: Optional[str] = None
    callout: Optional[str] = None
    sentence: Optional[str] = None
    citing_sentence: Optional[str] = None
    citing_snippet: Optional[str] = None
    citing_prefix: Optional[str] = None
    citing_suffix: Optional[str] = None
    previous_sentence: Optional[str] = None
    next_sentence: Optional[str] = None
    sentence_id: Optional[str] = None
    reference: Optional[CitationContextReference] = None
    resolution: Optional[ResolvedReference] = None


class CitationContextResponse(BaseModel):
    document_id: str
    context: Optional[CitationContextPayload] = None


class CitationGraphNode(BaseModel):
    id: str
    label: str
    kind: str
    doi: Optional[str] = None
    year: Optional[int] = None


class CitationGraphEdge(BaseModel):
    source: str
    target: str
    relation: str


class CitationGraphResponse(BaseModel):
    document_id: str
    root_id: str
    nodes: List[CitationGraphNode]
    edges: List[CitationGraphEdge]


class DocumentBodySegment(BaseModel):
    type: str
    text: Optional[str] = None
    citation_index: Optional[int] = None
    target_id: Optional[str] = None
    callout: Optional[str] = None
    label: Optional[str] = None
    sentence_id: Optional[str] = None


class DocumentBodySentence(BaseModel):
    sentence_id: Optional[str] = None
    segments: List[DocumentBodySegment]
    citation_indices: List[int] = Field(default_factory=list)


class DocumentBodyParagraph(BaseModel):
    paragraph_id: Optional[str] = None
    sentences: List[DocumentBodySentence]


class DocumentBodyResponse(BaseModel):
    document_id: str
    paragraphs: List[DocumentBodyParagraph]


class ReferenceRetrievalSource(BaseModel):
    label: str
    title: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None


class ReferenceRetrievalResponse(BaseModel):
    document_id: str
    reference_id: str
    canonical_citation: str
    doi: Optional[str] = None
    primary_url: Optional[str] = None
    manual_instructions: Optional[str] = None
    resolver_status: Optional[str] = None
    resolver_confidence: Optional[float] = None
    sources: List[ReferenceRetrievalSource] = Field(default_factory=list)


class ConfirmedClaim(BaseModel):
    claim_index: int
    parsed_text: str
    original_text: Optional[str] = None
    confidence: Optional[float] = None


class ClaimConfirmationRequest(BaseModel):
    document_id: str
    sentence_id: str
    sentence_text: str
    citation_index: int
    target_id: Optional[str] = None
    segmentation_model: Optional[str] = None
    reviewer_uid: str
    cited_work_id: Optional[str] = None
    citation_anchor: Optional[Dict[str, Any]] = None
    confirmed_claims: List[ConfirmedClaim] = Field(
        ..., min_length=1, description="Confirmed claim segments for this sentence"
    )


class ClaimConfirmationResponse(BaseModel):
    inserted: int


class IngestedDocument(BaseModel):
    id: str
    project_id: Optional[str] = None
    filename: str
    size_bytes: int
    sha256: str
    uploaded_at: str
    status: str
    extraction: IngestionStage
    body_extraction: Optional[IngestionStage] = None
    resolution: IngestionStage


class IngestListResponse(BaseModel):
    documents: List[IngestedDocument]


class IngestUploadResponse(BaseModel):
    document: IngestedDocument


class AttachmentTimelineEvent(BaseModel):
    event: str
    at: str
    detail: Optional[str] = None


class AttachmentArtifactPaths(BaseModel):
    tei_xml: Optional[str] = None
    tei_json: Optional[str] = None
    sentences: Optional[str] = None


class AttachmentStatus(BaseModel):
    id: str
    claim_id: Optional[str] = None
    doc_id: Optional[str] = None
    citation_index: Optional[int] = None
    target_id: Optional[str] = None
    source_ingest_id: Optional[str] = None
    filename: str
    size: Optional[int] = None
    content_sha256: Optional[str] = None
    status: str
    error: Optional[str] = None
    uploaded_at: str
    updated_at: str
    parsed_at: Optional[str] = None
    archived: Optional[bool] = None
    timeline: List[AttachmentTimelineEvent] = Field(default_factory=list)
    history: List[AttachmentTimelineEvent] = Field(default_factory=list)
    reference_hint: Optional[dict] = None
    attempts: Optional[int] = None
    max_attempts: Optional[int] = None
    retry_available: bool = True
    artifacts: Optional[Union[AttachmentArtifactPaths, dict]] = None
    claim_text: Optional[str] = Field(
        None,
        description="Claim text supplied with this attachment",
    )


class AttachmentCreateRequest(BaseModel):
    doc_id: Optional[str] = None
    local_path: str
    filename: Optional[str] = None
    size_bytes: Optional[int] = None
    reference_hint: Optional[dict] = None
    claim_text: Optional[str] = Field(
        None,
        description="Optional claim text to persist with the attachment",
    )


class AttachmentGlobalCreateRequest(BaseModel):
    claim_id: Optional[str] = None
    doc_id: Optional[str] = None
    local_path: str
    filename: Optional[str] = None
    size_bytes: Optional[int] = None
    reference_hint: Optional[dict] = None
    claim_text: Optional[str] = Field(
        None,
        description="Optional claim text to persist with the attachment",
    )
    citation_index: Optional[int] = None
    target_id: Optional[str] = None


class AttachmentUpdateRequest(BaseModel):
    claim_id: Optional[str] = None
    doc_id: Optional[str] = None
    citation_index: Optional[int] = None
    target_id: Optional[str] = None
    archived: Optional[bool] = None


class AttachmentCloneRequest(BaseModel):
    """Create a new attachment by cloning an existing one.

    This supports reusing one cited PDF across many citing claims without
    re-uploading or mutating the original attachment.
    """

    claim_id: Optional[str] = None
    doc_id: Optional[str] = None
    citation_index: Optional[int] = None
    target_id: Optional[str] = None
    reference_hint: Optional[dict] = None
    claim_text: Optional[str] = None


class AttachmentResponse(BaseModel):
    attachment: AttachmentStatus


class AttachmentListResponse(BaseModel):
    attachments: List[AttachmentStatus]


class EvidenceCandidatePayload(BaseModel):
    id: str
    claim_id: str
    attachment_id: str
    label: str
    text: str
    scores: Dict[str, Any] = Field(default_factory=dict)
    badges: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    spans: List[Dict[str, Any]] = Field(default_factory=list)
    highlights: List[Dict[str, Any]] = Field(default_factory=list)
    token_saliencies: Optional[List[float]] = None
    delta: Optional[Dict[str, Any]] = None

    # Phase 10-04: durable reviewer decision overlays (additive)
    decision_target: Dict[str, str] = Field(default_factory=dict)
    decision_state: Dict[str, Any] = Field(default_factory=dict)


class EvidenceListResponse(BaseModel):
    claim_id: str
    pinned: List[EvidenceCandidatePayload] = Field(default_factory=list)
    candidates: List[EvidenceCandidatePayload]
    total: int
    offset: int
    limit: int
    lock_state: Dict[str, Any]
    run: Optional[Dict[str, Any]] = None

    # Phase 10-04: decision stream metadata (reviewer_uid + version)
    decisions: Dict[str, Any] = Field(default_factory=dict)


# --- Durable evidence decisions (Phase 10-04) ------------------------------

DecisionTriage = Literal["none", "accepted", "rejected"]
EvidenceDecisionAction = Literal["pin", "unpin", "accept", "reject", "clear"]


class EvidenceDecisionTargetInput(BaseModel):
    attachment_id: str
    span_id: str


class EvidenceDecisionTarget(BaseModel):
    attachment_id: str
    span_id: str
    target_key: str


class EvidenceDecisionState(BaseModel):
    pinned: bool = False
    triage: DecisionTriage = "none"
    updated_at: Optional[str] = None


class EvidenceDecisionAppendRequest(BaseModel):
    idempotency_key: str
    expected_version: int = Field(..., ge=0)
    action: EvidenceDecisionAction
    target: Optional[EvidenceDecisionTargetInput] = None
    set: Optional[bool] = None
    payload: Optional[Dict[str, Any]] = None

    @field_validator("idempotency_key")
    def normalize_idempotency_key(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("idempotency_key is required")
        return text

    @field_validator("action", mode="before")
    def normalize_action(cls, value: Any) -> str:
        return str(value or "").strip().lower() or "clear"

    @model_validator(mode="after")
    def validate_rules(self) -> "EvidenceDecisionAppendRequest":
        if self.action in {"pin", "unpin", "accept", "reject"}:
            if self.target is None:
                raise ValueError("target is required for this action")
        if self.action == "clear":
            if self.target is not None:
                raise ValueError("target must be omitted for clear")
        if self.action in {"accept", "reject"} and self.set is None:
            self.set = True
        return self


class EvidenceDecisionAppendResponse(BaseModel):
    ok: bool = True
    event_id: Optional[int] = None
    event_uid: Optional[str] = None
    version: int = 0
    no_op: bool = False
    state: EvidenceDecisionState = Field(default_factory=EvidenceDecisionState)
    created_at: Optional[str] = None


class EvidenceDecisionEventPayload(BaseModel):
    event_id: int
    event_uid: str
    action: EvidenceDecisionAction
    target: Optional[EvidenceDecisionTarget] = None
    set: Optional[bool] = None
    created_at: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class EvidenceDecisionReadResponse(BaseModel):
    claim_id: str
    reviewer_uid: str
    version: int = 0
    targets: Dict[str, EvidenceDecisionState] = Field(default_factory=dict)
    pinned_targets: List[EvidenceDecisionTarget] = Field(default_factory=list)
    events: List[EvidenceDecisionEventPayload] = Field(default_factory=list)


class EvidenceHistoryEntry(BaseModel):
    run_id: str
    created_at: str
    summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceHistoryResponse(BaseModel):
    claim_id: str
    runs: List[EvidenceHistoryEntry]


class EvidenceRerunRequest(BaseModel):
    claim_text: Optional[str] = None
    note: Optional[str] = None
    advanced_settings: Optional[Dict[str, Any]] = None

    @field_validator("advanced_settings")
    def validate_advanced_settings(cls, value: Optional[Dict[str, Any]]):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("advanced_settings must be an object")
        profile = value.get("profile")
        if profile is not None and not isinstance(profile, str):
            raise ValueError("advanced_settings.profile must be a string")
        return value


class EvidenceRerunResponse(BaseModel):
    job_id: str
    status: str
    position: int
    locked: bool


class AttachmentSpanJumpResponse(BaseModel):
    attachment_id: str
    span_id: str
    page: str
    section_path: str
    paragraph_id: str
    sentence_id: str
    sentence_index: int


class AttachmentSpanExcerptSentence(BaseModel):
    sentence_id: str
    text: str
    is_highlight: bool
    page: str
    section_path: str
    paragraph_id: str


class AttachmentSpanExcerptResponse(BaseModel):
    attachment_id: str
    span_id: str
    before: int
    after: int
    sentences: List[AttachmentSpanExcerptSentence]


EvidenceSelectionVerdict = Literal["support", "contradict", "uncertain", "none"]


class EvidenceSelectionPrimary(BaseModel):
    candidate_id: str
    attachment_id: str
    span_id: str


class EvidenceSelectionSecondary(BaseModel):
    candidate_id: str
    attachment_id: str
    span_id: str
    rationale: str

    @field_validator("rationale")
    def validate_rationale(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("Secondary rationale is required")
        return text


class EvidenceSelectionUpsertRequest(BaseModel):
    verdict: EvidenceSelectionVerdict
    primary: Optional[EvidenceSelectionPrimary] = None
    secondary: List[EvidenceSelectionSecondary] = Field(default_factory=list)
    note: Optional[str] = None

    @field_validator("note")
    def normalize_note(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def validate_rules(self) -> "EvidenceSelectionUpsertRequest":
        if self.verdict == "uncertain" and not self.note:
            raise ValueError("note is required when verdict is uncertain")
        if self.verdict in ("support", "contradict") and self.primary is None:
            raise ValueError("primary selection is required for support/contradict")
        if self.verdict == "none" and self.primary is not None:
            raise ValueError("primary must be null when verdict is none")
        return self


class EvidenceSelectionPayload(BaseModel):
    claim_id: str
    updated_at: Optional[str] = None
    verdict: EvidenceSelectionVerdict = "none"
    primary: Optional[EvidenceSelectionPrimary] = None
    secondary: List[EvidenceSelectionSecondary] = Field(default_factory=list)
    note: Optional[str] = None

    @field_validator("note")
    def normalize_note(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def validate_rules(self) -> "EvidenceSelectionPayload":
        if self.verdict == "uncertain" and not self.note:
            raise ValueError("note is required when verdict is uncertain")
        if self.verdict in ("support", "contradict") and self.primary is None:
            raise ValueError("primary selection is required for support/contradict")
        if self.verdict == "none" and self.primary is not None:
            raise ValueError("primary must be null when verdict is none")
        return self


JudgmentVerdict = Literal["support", "contradict", "uncertain"]
JudgmentStatus = Literal["draft", "final"]


class JudgmentNotes(BaseModel):
    rationale: Optional[str] = None
    caveats: Optional[str] = None
    followups: Optional[str] = None

    @field_validator("rationale", "caveats", "followups")
    def normalize_note_field(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


ValidationRating = Literal[
    "strongly_agree",
    "agree",
    "neutral",
    "disagree",
    "strongly_disagree",
]


class JudgmentValidation(BaseModel):
    source_valid: Optional[ValidationRating] = None
    source_valid_comment: Optional[str] = None
    source_relevant: Optional[ValidationRating] = None
    source_relevant_comment: Optional[str] = None

    @field_validator(
        "source_valid_comment",
        "source_relevant_comment",
    )
    def normalize_validation_comment(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class JudgmentUpsertRequest(BaseModel):
    reviewer_uid: str = Field(
        "default",
        description=(
            "Local reviewer identity. Defaults to 'default' " "for legacy clients."
        ),
    )
    status: JudgmentStatus = "draft"
    verdict: Optional[JudgmentVerdict] = None
    notes: Optional[JudgmentNotes] = None
    validation: Optional[JudgmentValidation] = None

    doc_id: Optional[str] = None
    citation_index: Optional[int] = None
    target_id: Optional[str] = None
    sentence_id: Optional[str] = None
    callout: Optional[str] = None
    reference_id: Optional[str] = None
    doi: Optional[str] = None
    author: Optional[str] = None
    year: Optional[str] = None
    claim_text: Optional[str] = None

    # --- Phase 09+: stable citation/span anchoring ------------------------
    #
    # Motivation:
    # - `target_id` / `citation_index` are derived from PDF parsing and can drift
    #   across reprocessing (e.g., GROBID updates).
    # - Reviewers may segment the citing text differently, so segment IDs are not
    #   reliable for cross-user agreement.
    #
    # We therefore allow clients to attach a stable cited-work identifier and a
    # small bundle of text features (quote selectors, fingerprints) so a future
    # resolver can re-locate the intended in-text citation occurrence and spans.
    cited_work_id: Optional[str] = None
    citation_anchor: Optional[Dict[str, Any]] = None
    span_selectors: Optional[List[Dict[str, Any]]] = None

    @field_validator("reviewer_uid", mode="before")
    def normalize_reviewer_uid(cls, value: Any) -> str:
        text = str(value or "").strip()
        return text or "default"

    @model_validator(mode="after")
    def validate_rules(self) -> "JudgmentUpsertRequest":
        if self.status == "final" and self.verdict is None:
            raise ValueError("verdict is required when status is final")
        return self


class JudgmentPayload(BaseModel):
    claim_id: str
    reviewer_uid: str = "default"
    updated_at: Optional[str] = None
    status: JudgmentStatus = "draft"
    verdict: Optional[JudgmentVerdict] = None
    notes: Optional[JudgmentNotes] = None
    validation: Optional[JudgmentValidation] = None

    doc_id: Optional[str] = None
    citation_index: Optional[int] = None
    target_id: Optional[str] = None
    sentence_id: Optional[str] = None
    callout: Optional[str] = None
    reference_id: Optional[str] = None
    doi: Optional[str] = None
    author: Optional[str] = None
    year: Optional[str] = None
    claim_text: Optional[str] = None

    # Stable citation/span anchoring (see JudgmentUpsertRequest for rationale).
    cited_work_id: Optional[str] = None
    citation_anchor: Optional[Dict[str, Any]] = None
    span_selectors: Optional[List[Dict[str, Any]]] = None

    @field_validator("reviewer_uid", mode="before")
    def normalize_reviewer_uid(cls, value: Any) -> str:
        text = str(value or "").strip()
        return text or "default"

    @field_validator("claim_id")
    def validate_claim_id(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("claim_id is required")
        return text

    @model_validator(mode="after")
    def validate_rules(self) -> "JudgmentPayload":
        if self.status == "final" and self.verdict is None:
            raise ValueError("verdict is required when status is final")
        return self


class JudgmentListResponse(BaseModel):
    judgments: List[JudgmentPayload] = Field(default_factory=list)


class JudgmentByReviewerResponse(BaseModel):
    judgments: List[JudgmentPayload] = Field(default_factory=list)


# --- Document ledger / graph (Phase 09) ------------------------------------


class LedgerOption(BaseModel):
    num: int
    short: str
    apa: str
    status: Literal["green", "orange"]


class LedgerRow(BaseModel):
    num: int
    node_id: str
    status: Literal["green", "orange"]
    short: str
    title: Optional[str] = None
    apa: str
    incoming: List[int] = Field(default_factory=list)
    outgoing: List[int] = Field(default_factory=list)
    incoming_live: List[int] = Field(default_factory=list)
    outgoing_live: List[int] = Field(default_factory=list)
    assigned: bool = False
    anchored: bool = False
    extracted: bool = False
    resolved: bool = False
    ingest_id: Optional[str] = None


class LedgerResponse(BaseModel):
    rows: List[LedgerRow] = Field(default_factory=list)
    options: List[LedgerOption] = Field(default_factory=list)


class LedgerLinksUpdateRequest(BaseModel):
    targets: List[int] = Field(default_factory=list)


class LedgerAssignRequest(BaseModel):
    assigned: bool


class ProjectMeta(BaseModel):
    version: int = 1
    name: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Phase 09: local reviewer identities + graph view persistence.
    reviewers: List[str] = Field(default_factory=list)
    active_reviewer_uid: Optional[str] = None
    compare_reviewer_a: Optional[str] = None
    compare_reviewer_b: Optional[str] = None
    graph_settings: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def _normalize_reviewer_name(value: str) -> str:
        # Trim + collapse internal whitespace.
        text = str(value or "").strip()
        text = " ".join(text.split())
        return text

    @field_validator(
        "active_reviewer_uid", "compare_reviewer_a", "compare_reviewer_b", mode="before"
    )
    def normalize_reviewer_uid_fields(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = ProjectMeta._normalize_reviewer_name(value)
        return text or None

    @field_validator("reviewers", mode="before")
    def normalize_reviewers(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise ValueError("reviewers must be a list")

        normalized: List[str] = []
        seen: set[str] = set()
        for raw in value:
            text = ProjectMeta._normalize_reviewer_name(str(raw or ""))
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return normalized

    @field_validator("graph_settings", mode="before")
    def normalize_graph_settings(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("graph_settings must be an object")
        return value


class ProjectImportResponse(BaseModel):
    ok: bool
    backup_zip: str
    project: ProjectMeta


class AutoPlaceRequest(BaseModel):
    doc_id: str
    citation_index: Optional[int] = None
    target_id: Optional[str] = None


class AutoPlaceResponse(BaseModel):
    attachment: AttachmentStatus
    reused: bool = False


# --- Claim graph (Phase 09) -------------------------------------------------

ClaimGraphVoteVerdict = Literal["support", "contradict", "neutral", "uncertain"]


# --- Span-first graph (Rebuild) ----------------------------------------------

SpanKind = Literal["citation_window", "evidence_excerpt", "other"]
SpanCiteRole = Literal["evidentiary", "background", "reputational", "unknown"]
SpanStatus = Literal[
    "unknown",
    "supported",
    "contradicted",
    "contested",
    "not_supported",
]


class QuoteSelector(BaseModel):
    exact: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None


class WorkPayload(BaseModel):
    work_id: str
    doi: Optional[str] = None
    openalex_id: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[str] = None
    abstract: Optional[str] = None
    abstract_source: Optional[str] = None


class SpanPayload(BaseModel):
    span_id: str
    work_id: Optional[str] = None
    ingest_id: Optional[str] = None
    kind: SpanKind
    selector: Dict[str, Any] = Field(default_factory=dict)
    window_fingerprint: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SpanUpsertRequest(BaseModel):
    kind: SpanKind
    selector: QuoteSelector
    window_fingerprint: Optional[str] = None
    ingest_id: Optional[str] = None
    work_id: Optional[str] = None


class SpanUpsertResponse(BaseModel):
    span: SpanPayload


class SpanCiteUpsert(BaseModel):
    cited_work_id: str
    reference_id: Optional[str] = None
    citation_index: Optional[int] = None


class SpanCitesUpsertRequest(BaseModel):
    cites: List[SpanCiteUpsert] = Field(default_factory=list)


class SpanCitesUpsertResponse(BaseModel):
    span_id: str
    inserted: int


class SpanCiteRoleUpsertRequest(BaseModel):
    reviewer_uid: str
    role: SpanCiteRole


class ClaimSpanUpsert(BaseModel):
    order_index: int
    selector: Optional[QuoteSelector] = None


class ClaimSpansUpsertRequest(BaseModel):
    claim_spans: List[ClaimSpanUpsert] = Field(default_factory=list)


class ClaimSpanPayload(BaseModel):
    claim_span_id: str
    span_id: str
    order_index: int
    selector: Optional[Dict[str, Any]] = None


class ClaimSpansUpsertResponse(BaseModel):
    span_id: str
    claim_spans: List[ClaimSpanPayload] = Field(default_factory=list)


class ClaimAtomCreateRequest(BaseModel):
    reviewer_uid: str
    text: str
    supersedes_id: Optional[str] = None


class ClaimAtomPayload(BaseModel):
    claim_atom_id: str
    text: str
    created_by: str
    created_at: str
    updated_at: str
    supersedes_id: Optional[str] = None


class ClaimAtomCreateResponse(BaseModel):
    atom: ClaimAtomPayload


class ClaimSpanAtomLinkRequest(BaseModel):
    reviewer_uid: str
    claim_atom_id: str
    source: str = "manual"


class AtomAlignClaimRequest(BaseModel):
    reviewer_uid: str
    claim_id: str
    source: str = "manual"


AssertionVerdict = ClaimGraphVoteVerdict


class AssertionCreateRequest(BaseModel):
    reviewer_uid: str
    verdict: AssertionVerdict
    confidence: Optional[float] = None
    comment: Optional[str] = None
    claim_atom_id: Optional[str] = None
    claim_span_id: Optional[str] = None
    evidence_span_id: Optional[str] = None
    evidence_work_id: Optional[str] = None


class AssertionPayload(BaseModel):
    assertion_id: str
    reviewer_uid: str
    verdict: AssertionVerdict
    confidence: Optional[float] = None
    comment: Optional[str] = None
    claim_atom_id: Optional[str] = None
    claim_span_id: Optional[str] = None
    evidence_span_id: Optional[str] = None
    evidence_work_id: Optional[str] = None
    created_at: str


class AssertionCreateResponse(BaseModel):
    assertion: AssertionPayload


class ClaimSpanAssertionsResponse(BaseModel):
    claim_span_id: str
    assertions: List[AssertionPayload] = Field(default_factory=list)


class ClaimSpanStatusResponse(BaseModel):
    claim_span_id: str
    reviewer_uid: str
    status: SpanStatus
    checked: bool = False
    n_support: int = 0
    n_contradict: int = 0


class ClaimSpanContextResponse(BaseModel):
    claim_id: str
    reviewer_uid: str
    ingest_id: str
    citation_index: int
    target_id: Optional[str] = None
    span_id: str
    claim_span_id: str
    order_index: int
    cited_work_id: Optional[str] = None


class SpanStatusResponse(BaseModel):
    span_id: str
    reviewer_uid: str
    status: SpanStatus
    n_claim_spans: int = 0
    n_supported: int = 0
    n_contradicted: int = 0
    n_contested: int = 0
    n_not_supported: int = 0
    n_unknown: int = 0


class ClaimStatusResponse(BaseModel):
    claim_id: str
    reviewer_uid: str
    span_id: str
    claim_span_id: str
    status: SpanStatus
    checked: bool = False


class SpanCitePayload(BaseModel):
    cited_work_id: str
    reference_id: Optional[str] = None
    citation_index: Optional[int] = None
    role: SpanCiteRole = "unknown"


class ClaimSpanSummaryPayload(BaseModel):
    claim_span_id: str
    span_id: str
    order_index: int
    status: SpanStatus
    checked: bool = False
    n_support: int = 0
    n_contradict: int = 0
    current: Optional[AssertionPayload] = None
    history_n_total: int = 0


class SpanBundleResponse(BaseModel):
    span: SpanPayload
    reviewer_uid: str
    span_status: SpanStatusResponse
    cites: List[SpanCitePayload] = Field(default_factory=list)
    claim_spans: List[ClaimSpanSummaryPayload] = Field(default_factory=list)
    include_history: bool = False


class OkResponse(BaseModel):
    ok: bool = True


class SpanGraphCompactRequest(BaseModel):
    dry_run: bool = False
    aggressive: bool = False


class SpanGraphCompactResponse(BaseModel):
    dry_run: bool
    aggressive: bool
    before: int
    after: int
    dedupe_candidates: int
    legacy_candidates: int
    selection_dupe_candidates: int = 0
    deleted: int


class NeighborhoodSearchRequest(BaseModel):
    span_id: str
    reviewer_uid: str = "default"
    max_per_seed: int = Field(default=25, ge=1, le=200)
    min_bib_intersection: int = Field(default=1, ge=1, le=1000)
    query_text: Optional[str] = None
    min_abstract_score: float = Field(default=0.0, ge=0.0, le=1.0)


class NeighborhoodCandidatePayload(BaseModel):
    work_id: str
    bib_intersection: int
    abstract_score: Optional[float] = None
    rank: int
    title: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[str] = None


class NeighborhoodSearchResponse(BaseModel):
    run_id: str
    span_id: str
    reviewer_uid: str
    candidates: List[NeighborhoodCandidatePayload] = Field(default_factory=list)


class NeighborhoodRunResponse(BaseModel):
    run: Dict[str, Any]
    candidates: List[NeighborhoodCandidatePayload] = Field(default_factory=list)


class ClaimGraphNode(BaseModel):
    id: str
    kind: str
    label: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class ClaimGraphEdgeAggregates(BaseModel):
    n_support: int = 0
    n_contradict: int = 0
    n_neutral: int = 0
    n_uncertain: int = 0
    n_total: int = 0


class ClaimGraphEdge(BaseModel):
    edge_id: int
    source_id: str
    target_id: str
    kind: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    aggregates: ClaimGraphEdgeAggregates = Field(
        default_factory=ClaimGraphEdgeAggregates
    )


class TopologyEdgePayload(BaseModel):
    edge_id: int
    kind: str
    source_id: str
    target_id: str
    enabled: bool = True
    properties: Dict[str, Any] = Field(default_factory=dict)
    aggregates: ClaimGraphEdgeAggregates = Field(
        default_factory=ClaimGraphEdgeAggregates
    )


class ClaimSpanAtomsResponse(BaseModel):
    claim_span_id: str
    atoms: List[ClaimAtomPayload] = Field(default_factory=list)
    edges: List[TopologyEdgePayload] = Field(default_factory=list)


class TopologySettlePolicy(BaseModel):
    min_total_votes: int = Field(default=1, ge=1, le=1000)
    min_support: int = Field(default=1, ge=0, le=1000)
    min_contradict: int = Field(default=1, ge=0, le=1000)
    support_margin: int = Field(default=1, ge=0, le=1000)
    contradict_veto: bool = True
    manual_lock: bool = True


class TopologySettleRequest(BaseModel):
    kind: str
    policy: TopologySettlePolicy = Field(default_factory=TopologySettlePolicy)
    dry_run: bool = False


class TopologySettleResponse(BaseModel):
    kind: str
    dry_run: bool = False
    evaluated: int = 0
    enabled: int = 0
    disabled: int = 0
    unchanged: int = 0


class WorkCitesWorkRequest(BaseModel):
    citing_ingest_id: str
    cited_ingest_id: str
    reviewer_uid: str
    enabled: bool = True


class WorkCitesWorkResponse(BaseModel):
    edge: TopologyEdgePayload


class ClaimSubgraphResponse(BaseModel):
    center_claim_id: str
    nodes: List[ClaimGraphNode] = Field(default_factory=list)
    edges: List[ClaimGraphEdge] = Field(default_factory=list)


class ClaimGraphVote(BaseModel):
    reviewer_uid: str
    verdict: ClaimGraphVoteVerdict
    confidence: Optional[float] = None
    comment: Optional[str] = None
    updated_at: Optional[str] = None


class ClaimGraphVoteUpsertRequest(BaseModel):
    verdict: ClaimGraphVoteVerdict
    confidence: Optional[float] = None
    comment: Optional[str] = None


class ClaimGraphVoteListResponse(BaseModel):
    edge_id: int
    votes: List[ClaimGraphVote] = Field(default_factory=list)


class ClaimGraphVoteUpsertResponse(BaseModel):
    edge_id: int
    vote: ClaimGraphVote
    aggregates: ClaimGraphEdgeAggregates


class ClaimNodeResponse(BaseModel):
    node: ClaimGraphNode


class ClaimNodeListResponse(BaseModel):
    nodes: List[ClaimGraphNode] = Field(default_factory=list)
    count: int = 0


class ReferenceResolveRequest(BaseModel):
    citing_doc_id: str
    reference_ids: List[str] = Field(default_factory=list)


class ReferenceResolveResponse(BaseModel):
    citing_doc_id: str
    mapping: Dict[str, Optional[str]] = Field(default_factory=dict)


class CitationSpanLookupResponse(BaseModel):
    ingest_id: str
    citation_index: int
    target_id: Optional[str] = None
    span_id: Optional[str] = None


class ClaimLinkCreateRequest(BaseModel):
    source_claim_id: str
    target_claim_id: str


class ClaimLinkCreateResponse(BaseModel):
    edge: ClaimGraphEdge


class ClaimLinkDeleteResponse(BaseModel):
    ok: bool = True


class ClaimCandidate(BaseModel):
    target_claim_id: str
    score: float
    node: ClaimGraphNode


class ClaimCandidatesResponse(BaseModel):
    claim_id: str
    candidates: List[ClaimCandidate] = Field(default_factory=list)
