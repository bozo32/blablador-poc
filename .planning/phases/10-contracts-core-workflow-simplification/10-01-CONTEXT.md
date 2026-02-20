# Phase 10-01: Contracts Arc (Stage Boundaries) - Context

**Gathered:** 2026-02-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Define and adopt a minimal, versioned set of stage contracts (typed artifacts) so pipeline components can swap (e.g. ColBERT reranker) without changing callers.

Scope: contracts from extraction output through citespans, retrieval, filter, rerank, NLI, and an assessment-ready bundle (without changing the existing selection/judgment persistence endpoints).

</domain>

<decisions>
## Implementation Decisions

### Stage Boundaries
- Contract arc starts at extraction output (structured_doc).
- Citespans are anchored to citation windows (not claim segments) for this arc.
- Standardize stage names: `extract`, `citespans`, `retrieval`, `filter`, `rerank`, `nli`.
- Filter and rerank are separate contract stages.
- Unit of work is per citation anchor.
  - Citation anchor identity: `(document_id/work_id, citation_index, target_id)`.
  - Multi-citation spans: one run can include multiple `target_id` values; results are grouped by `target_id`.
- Primary vs fallback extraction: unified structured_doc contract with quality flags (no separate contract types).
- Partial results: contracts include `status` and best-effort `data` plus `warnings`.
- Failures: structured error object in-contract: `error={code,message,detail,retryable}` plus `warnings[]`.
- Rerank and NLI are optional stages (pipeline can stop after filter; later stages can run when requested).

### Artifact Storage
- Persist contract outputs as JSON in object store (S3/MinIO) with Postgres artifacts pointers.
- Granularity: one artifact per stage per run (stage payload may include grouped `target_id` results).
- Storage is append-only: keep old artifacts when inputs/settings change (keyed by `input_fingerprint`).
- Persist all stages by default, but cap list sizes; allow caller override.
  - Default caps: `candidates=200`, `rerank=50`, `nli=12`.
- Object key layout: `pipeline/{run_id}/{stage}.json` in the existing `works` bucket.
- No compression; no dedupe for paragraph text.
- Candidate payload stores full paragraph text (plus ids).
- Retrieval stage stores ranked ids + scores (normalized; no raw opaque blobs).
- No raw external service response artifacts by default.
- Optional run bundle artifact (zip of stage JSONs) is allowed (only when requested).
- `POST /dev/wipe` must delete pipeline contract artifacts too.
- Artifact access: API returns JSON bodies (clients do not fetch from S3 directly).

### IDs + Anchors
- Treat `doc_id` and `work_id` as the same stable id for now.
  - Canonical field name in contracts should be `work_id`; accept `doc_id` as alias for compatibility.
- `span_id` is a deterministic hash (stable across re-extraction).
- Evidence span anchoring uses quote selectors as canonical (`exact/prefix/suffix`), with optional fast-path offsets/attempt-local ids.
- `target_id` is OpenAlex work id when available (and should have fallbacks later).
- Run identity: `run_id` is a UUID.
- `candidate_id` is stable per-run (hash of `{run_id,target_id,span_id}` or equivalent).
- Candidate references for cited text use `attachment_id + span_id`.
- Include `paragraph_id` and `section_path` in contracts when available.

### Versioning + Evolution
- Versioning: integer `schema_version`.
- Encode version in both payload and `artifact_type` (artifact_type includes `@vN`).
- Compatibility: consumers must read older versions and upgrade in code.
- Evolution rule (POC): additive-only changes (no rename/remove in-place).

### Claude's Discretion
- Exact artifact_type naming scheme (default to hierarchical `contracts/{stage}@vN`).
- Exact contents of the assessment-ready bundle (default: ranked spans + NLI + highlight/snippet fields).
- The precise fallback chain for `target_id` when OpenAlex id is missing (policy to be documented during planning).

</decisions>

<specifics>
## Specific Ideas

- Treat ColBERT as the canonical example of a swappable stage implementation; swaps should be visible via per-stage component metadata (implementation id + version).
- Fixed defaults are fine, but callers can override caps when they need more results.

</specifics>

<deferred>
## Deferred Ideas

None. Discussion stayed within Phase 10-01 scope.

</deferred>

---

*Phase: 10-contracts-core-workflow-simplification*
*Context gathered: 2026-02-20*
