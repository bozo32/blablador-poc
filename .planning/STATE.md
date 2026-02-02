# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-23)

**Core value:** Readers can validate a citation by linking a claim to supporting (or contradicting) evidence from the cited source, with the judgment captured and reusable.
**Current focus:** Phase 8 - Workspace Organization

## Current Position

Phase: 7 of 8 (Validation + Export)
Plan: 3 of 3 in current phase
Status: Phase 7 verified
Last activity: 2026-02-02 — Phase 7 verification passed

Progress: ███████████░ 90%

## Performance Metrics

**Velocity:**
- Total plans completed: 27
- Average duration: 14 min
- Total execution time: 3.6 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 4 | 4 | 4 min |
| 2 | 6 | 6 | 4 min |
| 3 | 3 | 3 | 6 min |
| 4 | 6 | 6 | 45 min |
| 5 | 4 | 4 | 20 min |

**Recent Trend:**
- Last 5 plans: 05-04 (23 min), 05-03 (15 min), 05-02 (22 min), 05-01 (20 min), 04-03 (20 min)
- Trend: Building

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

 - Evidence pipeline caches attachment sentences centrally so deterministic windows reuse a single source of truth (05-01).
 - BM25 seeding now carries provenance/badge metadata with spaCy tokenization and regex fallback for deterministic output (05-01).
 - EvidencePipeline serializes ranked candidates (scores + highlights) for downstream API consumers (05-01).
 - Evidence runs now persist to disk with history + delta metadata to power audits (05-02).
 - EvidenceMatchingService orchestrates reruns, auto-refresh after attachments, and exposes list/history helpers (05-02).
 - `/claims/{claim_id}/evidence*` FastAPI routes deliver candidates, rerun status, and run history for the frontend (05-02).
 - Evidence store auto-refreshes when attachment timeline events reach matched so reviewers see fresh candidates without rerunning manually (05-03).
 - Streamlit evidence panel keeps claim selection, rerun/load-more controls, and history in sync through the session-backed EvidenceStore (05-03).
 - Evidence cards now render via a reusable component with keyboard navigation, inline actions, and shared styling (05-04).
 - Inline share/pin/Open PDF actions are handled inside the session-backed store until backend endpoints land, with rationale sidebar + export helpers covering reviewer context (05-04).
 - Store attachment metadata + timelines as JSON on disk to guarantee resumable processing (04-03).
 - Streamlit attachment queue now polls backend statuses (no client-side simulation) and surfaces retry/diagnostics affordances (04-03).
 - Retrieval instructions copy control uses a reusable clipboard helper with manual fallback so reviewers can trust the copy action (04-04).
 - Attachment lifecycle emits pending → converting → parsing → matched on both backend and UI, with queue summary chips reflecting the counts (04-05).
-  - Queue auto-matches dropped files via claim heuristics and highlights ambiguous attachments with manual reassignment controls (04-06).
-  - Persist attachment claim_text so evidence reruns inherit the UI-provided text (05-05).
-  - EvidenceMatchingService resolves claim_text from attachment metadata before raising 409 so new claims succeed (05-05).
  - Cache claim_text metadata in the EvidenceStore via claim_queue registry updates so background refreshes and reruns always have the text (05-06).
  - Surface an explicit remediation path whenever the backend rejects evidence fetches for missing claim_text and reuse the cached or edited text to recover (05-06).
 - Send claim_text from attachment uploads so backend auto reruns triggered after parsing have the data they need (05-06).

 - Span excerpt/jump anchors use TEI sentence_id as span_id so the UI can request stable jump metadata (06-01).
 - Evidence selections are persisted on disk under data/evidence_selections keyed by claim_id (06-01).

 - Judgments are persisted on disk under data/judgments keyed by claim_id with collision-safe filenames, and exports default to final-only (07-01).

- Use local citation graph data when DOI resolution is unavailable to avoid OpenAlex 404s (02-02).
- Prefer GROBID consolidation when resolver sources disagree; warn that OpenAlex may return citing articles (02-06).

### Pending Todos

- Run full cleanup protocol pass (see `.planning/CLEANUP_PROTOCOL.md`) and re-verify UI flows.
- Ensure chase queue preserves document order (no reordering on activation).

### Blockers/Concerns

- OpenAlex resolution can return citing articles; consolidation remains inconsistent for some references.
- Non-DOI link formatting in callout metadata remains messy.

## Session Continuity

Last session: 2026-02-02T18:52:45Z
Stopped at: Phase 7 verification passed
Resume file: None
