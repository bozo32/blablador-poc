---
phase: 05-evidence-matching-ranking
verified: 2026-01-28T22:33:14Z
status: human_needed
score: 12/12 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 10/12
  gaps_closed:
    - "Backend claim_text plumbing now seeds the first evidence sync so /claims/{claim_id}/evidence no longer 409s new claims."
    - "Attachment lifecycle auto reruns now pass cached or persisted claim_text so deterministic windows can be built without manual seeding."
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "Streamlit evidence panel claim_text recovery"
    expected: "When the backend reports HTTP 409 for missing claim_text on a fresh claim, the evidence panel shows the warning, lets the reviewer resend the cached text, and the subsequent sync succeeds."
    why_human: "Requires running the Streamlit UI because the warning banner, button interactions, and rerun toast depend on live UI behavior and backend timing."
  - test: "Attachment upload claim_text payload"
    expected: "Uploading a PDF for a claim stores claim_text in the attachment metadata and the attachment pipeline sends that text to the auto rerun job so the rerun completes without manual seeding."
    why_human: "Needs a manual upload and backend log/metadata inspection because the claim_text travel through the UI and pipeline cannot be observed purely from the code."
---

# Phase 05: Evidence Matching + Ranking Verification Report

**Phase Goal:** Users receive ranked evidence candidates that link claims to cited text.
**Verified:** 2026-01-28T22:33:14Z
**Status:** human_needed
**Re-verification:** Yes — after gap closure (previous claim_text blockers resolved).

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Attachment-backed sentences are normalized into deterministic windows for a claim. | ✓ VERIFIED | `backend/evidence_matching/loaders.py` builds deterministic windows with stride controls and metadata normalization; `tests/test_evidence_matching_pipeline.py::test_loader_*` covers the loader output. |
| 2 | Evidence pipeline produces ranked `EvidenceCandidate` objects with combined FAISS/BM25/NLI scoring. | ✓ VERIFIED | `backend/evidence_matching/pipeline.py` sorts seeds via `_score_candidates`, trims to caps, and annotates NLI labels; pipeline ordering tests assert the rerank behaves as expected. |
| 3 | Each candidate carries page/section/bbox metadata so the UI can deep-link into PDFs. | ✓ VERIFIED | `backend/evidence_matching/types.py` and `serializers.py` embed page/section/bbox fields; serialization tests validate metadata presence in API responses. |
| 4 | Claim-focused ranking runs are persisted with metadata, score deltas, and history snapshots. | ✓ VERIFIED | `backend/evidence_matching/store.py` records per-run metadata/deltas/history with delta annotations verified via `tests/test_evidence_matching_api.py`. |
| 5 | FastAPI exposes evidence endpoints for listing candidates, triggering reruns, and downloading rank history. | ✓ VERIFIED | `backend/main.py` registers `/claims/{claim_id}/evidence`, `/claims/{claim_id}/evidence/rerun`, and `/claims/{claim_id}/evidence/history` using Pydantic schemas; `tests/test_evidence_matching_api.py` asserts each route’s response structure. |
| 6 | Attachment lifecycle changes automatically enqueue evidence reruns so claims always have fresh candidates. | ✓ VERIFIED | `backend/attachment_pipeline.py` reloads the processed attachment record, reads `claim_text`, and calls `evidence_service.trigger_auto_rerun(claim_id, claim_text=claim_text)`; `EvidenceMatchingService.trigger_auto_rerun` resolves text from attachments/metadata before requesting the rerun, and `tests/test_evidence_matching_api.py::test_service_auto_rerun_triggered_by_attachment_pipeline` confirms the text arrives. |
| 7 | Streamlit session state tracks evidence candidates, filters, pinned cards, and load-more counts per claim. | ✓ VERIFIED | `frontend/evidence_store.py` keeps per-claim metadata, load-more counters, and rerun state, and `tests/test_evidence_store.py` covers paging, filters, and toast guardrails. |
| 8 | Selecting a claim or completing attachment parsing automatically fetches ranked evidence and rerun status. | ✓ VERIFIED | `frontend/ui.py` passes `active_claim_text_payload` to `EvidenceStore` for sync/load-more/reruns, `frontend/claim_queue.py` seeds the metadata via `_sync_evidence_store_metadata`, and `tests/test_evidence_store.py::test_store_claim_metadata_fallbacks_to_cached_text` ensures every fetch/rerun includes cached claim_text. |
| 9 | Manual rerun + load-more actions enforce concurrency limits (max two outstanding fetches) and surface toast feedback. | ✓ VERIFIED | `frontend/evidence_api.py` enforces `_reserve_slot`/`_release_slot` per claim, and `EvidenceStore` shows toast when reruns are inflight; `tests/test_evidence_store.py::test_api_stub_request_limit` and `test_request_rerun_queue_limit` cover the behavior. |
| 10 | Evidence cards display snippets (~600 chars), badges, sparkline, inline actions (accept/reject/open/pin/share), plus keyboard navigation. | ✓ VERIFIED | `frontend/components/evidence_card.py` renders snippets, badges, and keyboard handlers plus CSS; coverage in `tests/test_evidence_components.py`. |
| 11 | Rationale sidebar stays synced with hovered/pinned cards, exposes rank scores/deltas, and offers JSON export plus advanced toggle. | ✓ VERIFIED | `frontend/components/rationale_sidebar.py` selects focused/pinned entries, displays rank delta badges, and wires export/advanced toggle markup for the sidebar view. |
| 12 | UI renders filter chips, load-more batches of five, global entail/contrad progress bar, and rerun/status toasts. | ✓ VERIFIED | `frontend/ui.py` renders chips, load-more/resync buttons, progress bar markup, rerun status toasts, and ties them to store state so the UI stays responsive. |

**Score:** 12/12 truths verified

## Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `backend/schemas.py` | Models that carry claim_text across attachment/evidence API boundaries | ✓ VERIFIED | `AttachmentCreateRequest`, `AttachmentStatus`, and rerun/list schemas expose `claim_text`, allowing metadata persistence and downstream validation. |
| `backend/main.py` | Attachment/evidence endpoints forwarding claim_text | ✓ VERIFIED | `create_claim_attachment` forwards `payload.claim_text` to `attachment_store`, and `list_claim_evidence` passes the optional `claim_text` query parameter into `evidence_service.ensure_current_run`. |
| `backend/attachment_store.py` | Persistence of claim_text alongside attachment metadata | ✓ VERIFIED | `create_attachment` writes `claim_text` into the record, and `_public_view` exposes it for UI and pipeline consumers, enabling `_claim_text_from_attachments` to find it later. |
| `backend/attachment_pipeline.py` | Auto rerun wiring that includes claim_text | ✓ VERIFIED | After matching and persisting artifacts, `process_attachment` reloads the record, extracts `claim_id`/`claim_text`, and calls `evidence_service.trigger_auto_rerun` with the text so auto reruns never start without missing context. |
| `backend/evidence_matching/service.py` | Claim_text resolution and rerun orchestration | ✓ VERIFIED | `_resolve_claim_text` prefers supplied text, falls back to the latest run’s metadata or attachment metadata, and both `request_rerun` and the rerun worker reuse the resolved text before `_execute_run`. |
| `tests/test_evidence_matching_api.py` | Regression coverage for first-run syncs and reruns | ✓ VERIFIED | `test_service_ensure_current_run_tracks_snapshot`, `test_service_auto_rerun_triggered_by_attachment_pipeline`, and `test_service_api_accepts_claim_text` prove fresh claims accept text and auto reruns receive it. |
| `tests/test_attachment_pipeline.py` | Attachment pipeline preserves claim_text | ✓ VERIFIED | `test_process_attachment_creates_artifacts` asserts both the internal record and public view keep the supplied `claim_text`. |
| `frontend/evidence_store.py` | Metadata-backed claim_text propagation for sync + rerun calls | ✓ VERIFIED | `_resolve_claim_text` caches normalized text and every `sync_for_claim`, `load_more`, `apply_filter`, and `queue_rerun` resolves the text before touching `evidence_api`. |
| `frontend/ui.py` | Claim header + rerun controls wired to send claim_text and show remediation | ✓ VERIFIED | `render_evidence_panel` computes `active_claim_text_payload`, passes it to store calls, and surfaces a warning/button when the backend flags a missing `claim_text`. |
| `frontend/claim_queue.py` | Claim registry metadata kept in sync with evidence store | ✓ VERIFIED | `_sync_evidence_store_metadata` writes claim_text/callouts/flags into `EvidenceStore`, and `_notify_evidence_refresh` triggers claim_text-backed syncs after timeline events. |
| `frontend/attachment_queue.py` | Attachment POST payload includes claim_text | ✓ VERIFIED | `_upload_to_backend` adds the normalized claim text (when present) to the JSON body so backend uploads never miss it. |
| `tests/test_evidence_store.py` | Claim_text fallback + rerun coverage | ✓ VERIFIED | `test_store_claim_metadata_fallbacks_to_cached_text` and `test_store_queue_rerun_includes_cached_claim_text` prove cached text is reused across syncs and rerun requests. |

## Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `frontend/ui.render_evidence_panel` | `frontend.evidence_store.sync_for_claim` | Claim_text parameter derived from the active claim metadata | ✓ VERIFIED | The panel computes `active_claim_text_payload` from claim data and passes it for initial syncs, filters, reruns, load-more, and refresh actions. |
| `frontend/claim_queue.record_timeline_event` | `frontend.evidence_store.sync_for_claim` | Metadata updates with claim_text before background refresh | ✓ VERIFIED | Timeline events call `_notify_evidence_refresh`, which marks the claim stale and re-syncs using `_claim_text_for`. |
| `frontend/attachment_queue._upload_to_backend` | `/claims/{claim_id}/attachments` | JSON payload containing claim_text | ✓ VERIFIED | `_upload_to_backend` reads the associated claim record, strips whitespace, and includes `claim_text` when posting so backend auto reruns have the field. |
| `backend/main.create_claim_attachment` | `backend/attachment_store.create_attachment` | Claim_text argument from the request payload | ✓ VERIFIED | The route forwards `payload.claim_text` directly to the store, ensuring the persisted attachment metadata contains the text. |
| `backend/main.list_claim_evidence` | `EvidenceMatchingService.ensure_current_run` | Claim_text query parameter | ✓ VERIFIED | The evidence listing endpoint forwards its optional `claim_text` argument to the service before listing candidates. |
| `backend/attachment_pipeline.process_attachment` | `EvidenceMatchingService.trigger_auto_rerun` | Claim_text read from the attachment record | ✓ VERIFIED | After marking the attachment matched, the pipeline reads `claim_text` from the stored record and passes it into `trigger_auto_rerun` so reruns never start without text. |

## Requirements Coverage

| Requirement | Status | Blocking Issue |
| --- | --- | --- |
| EVD-04 – Deterministic matching finds closest cited spans | ✓ SATISFIED | Claim_text plumbing guarantees `/claims/{claim_id}/evidence` runs for fresh claims, so deterministic matching can build windows immediately. |
| EVD-05 – System reranks candidates and surfaces top options | ✓ SATISFIED | With claim_text provided on the first sync, the reranking pipeline executes and returns ranked candidates to the UI. |
| EVD-06 – Present top-N candidates with entail/contradict labels | ✓ SATISFIED | UI load-more/rerun flows now receive candidates annotated with entail/contrad labels because the backend can build deterministic windows on every run. |

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| --- | --- | --- | --- | --- |
| `frontend/evidence_api.py` | ~191 | Helper calls `/claims/{claim_id}/evidence/export`, but no backend route implements that endpoint. | ⚠️ Warning | `export_rank_json` will 404 until a backend export route exists or the helper is removed; the phase never invoked this path, but it remains unattached. |

## Human Verification Required

1. **Streamlit evidence panel claim_text recovery**  
   **Test:** Run `streamlit run app.py`, pick a claim without prior evidence runs, trigger a fetch to produce a missing `claim_text` 409, press the warning’s “Send claim text” button, and confirm the evidence list loads.  
   **Expected:** The warning appears, the button resubmits the cached claim text, and the subsequent evidence fetch succeeds without 409.  
   **Why human:** Only the running UI can show the banner, interactive button, and rerun toast together after a live 409. 

2. **Attachment upload claim_text payload**  
   **Test:** Upload a PDF for an existing claim (via the attachment queue), then inspect the attachment metadata or logs to confirm `claim_text` is stored and sent to the auto rerun job.  
   **Expected:** The attachment metadata reflects the uploaded claim text, and the auto rerun job receives that text so the rerun completes immediately.  
   **Why human:** Observing the upload UI plus attachment pipeline propagation requires a manual upload and metadata/log check. 

## Gaps Summary

No open gaps remain; the claim_text plumbing resolved the two previous blockers so evidence runs and reruns now succeed for new claims.

---

_Verified: 2026-01-28T22:33:14Z_  
_Verifier: Claude (gsd-verifier)_
