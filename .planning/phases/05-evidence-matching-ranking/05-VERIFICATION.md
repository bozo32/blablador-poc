---
phase: 05-evidence-matching-ranking
verified: 2026-01-28T10:15:00Z
status: gaps_found
score: 10/12 must-haves verified
gaps:
  - truth: "Selecting a claim or completing attachment parsing automatically fetches ranked evidence and rerun status."
    status: failed
    reason: "The FastAPI `ensure_current_run` call requires a `claim_text` parameter for the first run, but no UI or store path ever supplies it, so `/claims/{claim_id}/evidence` returns HTTP 409 for every new claim and no candidates ever load."
    artifacts:
      - path: "frontend/ui.py"
        issue: "`render_evidence_panel` calls `store.sync_for_claim(selected_claim)` (L948-L964) without passing the claim text."
      - path: "frontend/evidence_store.py"
        issue: "`sync_for_claim` forwards its `claim_text` argument to `evidence_api.list_evidence`, but callers never provide one, so the query params never include `claim_text` (L69-L105)."
      - path: "frontend/evidence_api.py"
        issue: "`list_evidence` only attaches the `claim_text` param when a value is provided (L103-L135), which never happens."
      - path: "backend/main.py"
        issue: "`list_claim_evidence` invokes `evidence_service.ensure_current_run(claim_id, claim_text=claim_text)` (L424-L443) and raises HTTP 409 when no claim_text is supplied and no prior run exists."
      - path: "backend/evidence_matching/service.py"
        issue: "`_resolve_claim_text` raises a ValueError unless a claim_text was supplied previously (L289-L305)."
    missing:
      - "Plumb each claim's text from the claim registry/record into `EvidenceStore.sync_for_claim` and `EvidenceStore.queue_rerun`, so `/claims/{claim_id}/evidence` receives `claim_text` on the initial request."
      - "Persist the resolved claim text in metadata during the first successful run so subsequent auto-reruns can reuse it."
      - "UI feedback when a claim lacks stored evidence should prompt the user to submit claim text rather than silently failing."
  - truth: "Attachment lifecycle changes automatically enqueue evidence reruns so claims always have fresh candidates."
    status: failed
    reason: "The attachment pipeline queues auto reruns without a claim_text payload, so `_resolve_claim_text` fails before the pipeline executes unless a previous manual run already stored claim_text. Because the UI never performs that first run, auto reruns never succeed and claims never gain fresh candidates."
    artifacts:
      - path: "backend/attachment_pipeline.py"
        issue: "`process_attachment` calls `evidence_service.trigger_auto_rerun(claim_id)` with no claim_text after marking an attachment matched (L122-L134)."
      - path: "backend/evidence_matching/service.py"
        issue: "`trigger_auto_rerun` simply proxies to `request_rerun` without adding claim text (L168-L174), and `_run_job` immediately calls `_resolve_claim_text`, which raises when no stored text exists (L198-L213 & L289-L305)."
      - path: "frontend/evidence_store.py"
        issue: "`queue_rerun` never passes `claim_text` when requesting reruns (L139-L170), so even manual reruns cannot seed the metadata the auto path depends on."
    missing:
      - "Include the active claim's text when calling `EvidenceStore.queue_rerun` so manual reruns can store it in run metadata."
      - "Propagate claim text (or fetch it server-side from `claim_store`) inside `trigger_auto_rerun` so attachment-based jobs can execute without relying on a prior manual run."
      - "Add regression tests that assert `ensure_current_run` succeeds for a fresh claim when the UI supplies claim text."
---

# Phase 05: Evidence Matching + Ranking Verification Report

**Phase Goal:** Users receive ranked evidence candidates that link claims to cited text.

**Verified:** 2026-01-28T10:15:00Z

**Status:** gaps_found

**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Attachment-backed sentences are normalized into deterministic windows for a claim. | ✓ VERIFIED | `backend/evidence_matching/loaders.py` builds deterministic windows from attachment sentences with stride/size controls and metadata (L45-L155), covered by `tests/test_evidence_matching_pipeline.py::test_loader_*`. |
| 2 | Evidence pipeline produces ranked `EvidenceCandidate` objects with combined FAISS/BM25/NLI scoring. | ✓ VERIFIED | `backend/evidence_matching/pipeline.py` sorts seeds via `_score_candidates` and applies NLI labels before returning capped lists (L32-L117); regression in `tests/test_evidence_matching_pipeline.py::test_pipeline_orders_candidates`. |
| 3 | Each candidate carries page/section/bbox metadata so the UI can deep-link into PDFs. | ✓ VERIFIED | `backend/evidence_matching/types.py` ensures spans carry page/section/bbox metadata (L55-L125) and `serializers.py` injects page/section/attachment/bbox_count fields (L12-L50). |
| 4 | Claim-focused ranking runs are persisted with run metadata, score deltas, and history snapshots. | ✓ VERIFIED | `backend/evidence_matching/store.py` records each run with metadata, delta annotations, and trimmed history (L29-L190); `tests/test_evidence_matching_api.py::test_store_*` verify behavior. |
| 5 | FastAPI exposes evidence endpoints for listing candidates, triggering reruns, and downloading rank history. | ✓ VERIFIED | `backend/main.py` registers `/claims/{claim_id}/evidence`, `/evidence/rerun`, and `/evidence/history` using the new schemas (L424-L499) validated by `tests/test_evidence_matching_api.py`. |
| 6 | Attachment lifecycle changes automatically enqueue evidence reruns so claims always have fresh candidates. | ✗ FAILED | Auto reruns are queued without claim text (`backend/attachment_pipeline.py` L122-L134) and `_resolve_claim_text` rejects runs without a stored claim_text (`backend/evidence_matching/service.py` L289-L305), so no rerun can execute for a new claim. |
| 7 | Streamlit session state tracks evidence candidates, filters, pinned cards, and load-more counts per claim. | ✓ VERIFIED | `frontend/evidence_store.py` maintains per-claim state (filters, pins, load-more pages, rerun queue) and exposes helpers used by the UI; covered by `tests/test_evidence_store.py`. |
| 8 | Selecting a claim or completing attachment parsing automatically fetches ranked evidence and rerun status. | ✗ FAILED | The UI never supplies `claim_text`, so `/claims/{claim_id}/evidence` responds 409 for every fresh claim (`frontend/ui.py` L948-L964, `frontend/evidence_store.py` L69-L105, `frontend/evidence_api.py` L103-L135, `backend/main.py` L424-L443). |
| 9 | Manual rerun + load-more actions enforce concurrency limits (max two outstanding fetches) and surface toast feedback. | ✓ VERIFIED | `frontend/evidence_api.py` enforces `MAX_LIST_REQUESTS` with `_reserve_slot`/`_release_slot` (L77-L135) and `EvidenceStore.load_more/queue_rerun` supply user toasts; verified by `tests/test_evidence_store.py::test_api_stub_request_limit` & `test_request_rerun_queue_limit`. |
|10 | Evidence cards display snippets (~600 chars), badges, sparkline, inline actions (accept/reject/open/pin/share), plus keyboard navigation. | ✓ VERIFIED | `frontend/components/evidence_card.py` truncates snippets, renders badges/actions, injects keyboard JS, and loads shared CSS (L17-L345, L368-L465); styling in `frontend/assets/evidence.css`. |
|11 | Rationale sidebar stays synced with hovered/pinned cards, exposes rank scores/deltas, and offers JSON export plus advanced toggle. | ✓ VERIFIED | `frontend/components/rationale_sidebar.py` selects the focused/pinned candidate, shows progress, rank deltas, advanced mode, and export downloads (L17-L290). |
|12 | UI renders filter chips, load-more batches of five, global entail/contrad progress bar, and rerun/status toasts. | ✓ VERIFIED | `frontend/ui.py` `render_evidence_panel` renders claim headers, warning toasts, filter chips, load-more/resync buttons, and progress bar markup (L995-L1250). |

**Score:** 10/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `backend/evidence_matching/types.py` | Data classes for candidates/spans/scores | ✓ VERIFIED | 255-line module exporting enums/dataclasses with metadata normalization. |
| `backend/evidence_matching/loaders.py` | Deterministic attachment windows | ✓ VERIFIED | Loads matched attachments via `attachment_store`, sanitizes sentences, and builds rolling windows. |
| `backend/evidence_matching/deterministic_matcher.py` | BM25 seeding utilities | ✓ VERIFIED | Scores windows with BM25, tags provenance/badges, honors settings caps. |
| `backend/evidence_matching/pipeline.py` | Retrieve → rerank → NLI pipeline | ✓ VERIFIED | Combines hybrid retrieval metadata, weighted scores, trimming, and NLI labeling. |
| `backend/evidence_matching/store.py` | Run persistence + deltas/history | ✓ VERIFIED | Atomic writes + history pruning with delta annotations used by service. |
| `backend/evidence_matching/service.py` | Orchestration + rerun queue | ✓ VERIFIED | Manages ensure/list/history/rerun with locking; however, lacks fallback to fetch claim text (see gaps). |
| `backend/main.py` | Evidence endpoints | ✓ VERIFIED | FastAPI routes delegate to service and schemas. |
| `backend/attachment_pipeline.py` | Auto trigger reruns after match | ⚠️ PARTIAL | Calls `trigger_auto_rerun` but omits claim_text, causing first-time reruns to fail. |
| `backend/schemas.py` | Evidence payload models | ✓ VERIFIED | Defines candidate/list/history/rerun schemas used by API + tests. |
| `tests/test_evidence_matching_pipeline.py` | Loader/matcher/pipeline tests | ✓ VERIFIED | Covers loader sanitization, matcher scoring, pipeline ordering, serialization. |
| `tests/test_evidence_matching_api.py` | Store/service/API tests | ✓ VERIFIED | Validates run persistence, rerun queue, FastAPI responses. |
| `frontend/evidence_api.py` | Streamlit-aware HTTP helpers | ✓ VERIFIED | Wraps evidence endpoints with concurrency guard + toast errors. |
| `frontend/evidence_store.py` | Session store for candidates/filters/reruns | ✓ VERIFIED | Centralizes state transitions; missing claim_text plumbing noted in gaps. |
| `frontend/ui.py` | Evidence board UI wiring | ✓ VERIFIED | Renders selectors, controls, progress, cards, sidebar. |
| `frontend/components/evidence_card.py` | Evidence card renderer | ✓ VERIFIED | Outputs snippet, badges, actions, keyboard nav, CSS injection. |
| `frontend/components/rationale_sidebar.py` | Sidebar + rationale view | ✓ VERIFIED | Shows progress, selected candidate rationale, exports. |
| `frontend/assets/evidence.css` | Evidence board styles | ✓ VERIFIED | Styles cards, progress bar, keyboard focus, sidebar. |
| `tests/test_evidence_store.py` | Store/API helper tests | ✓ VERIFIED | Confirms load-more persistence, filter toggles, rerun queue guard. |
| `tests/test_evidence_components.py` | Component helper tests | ✓ VERIFIED | Covers snippet truncation, highlight merging, progress summaries, chip configs. |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `loaders.py` | `backend.attachment_store` | `load_sentences_for_attachment` | ✓ VERIFIED | Loader directly imports `attachment_store` to read matched sentences. |
| `pipeline.py` | `backend.hybrid` | `DefaultHybridPipeline` | ✓ VERIFIED | Imports `HybridPipeline` when available for retrieval priming. |
| `serializers.py` | `backend.settings` | `settings.EVIDENCE*` caps | ⚠️ NOT WIRED | Serializers use a hardcoded 600-char limit and never consult settings; candidate cap currently enforced earlier in `pipeline.run`. |
| `service.py` | `pipeline.py` | `EvidencePipeline.run` | ✓ VERIFIED | Injects pipeline dependency and calls `run` when executing reruns. |
| `attachment_pipeline.py` | `evidence_service` | `trigger_auto_rerun` | ✓ VERIFIED | After `mark_matched`, auto-enqueues reruns (but missing claim_text payload). |
| `backend/main.py` | `backend.schemas` | Pydantic response models | ✓ VERIFIED | Routes instantiate schema responses before returning JSON. |
| `frontend/ui.py` | `frontend.evidence_store` | Store initialization | ✓ VERIFIED | UI instantiates `EvidenceStore` and calls its sync/pin/rerun helpers. |
| `frontend/evidence_store.py` | `frontend.evidence_api` | HTTP helpers | ✓ VERIFIED | Store imports API helpers to fetch evidence/history/rerun data. |
| `frontend/claim_queue.py` | `frontend.evidence_store` | `mark_claim_stale/sync_for_claim` | ✓ VERIFIED | Claim timeline events notify the evidence store to refresh. |
| `frontend/ui.py` | `frontend.components.evidence_card` | `EvidenceCardRenderer` | ✓ VERIFIED | UI renders cards using the shared component/callback bundle. |
| `frontend/ui.py` | `frontend.components.rationale_sidebar` | `render_rationale_sidebar` | ✓ VERIFIED | Sidebar column stays synced with store focus order. |
| `frontend/components/evidence_card.py` | `frontend/assets/evidence.css` | `ASSET_PATH` injection | ✓ VERIFIED | Component loads shared CSS via `st.markdown(<style/>)`. |

### Requirements Coverage

| Requirement | Status | Blocking Issue |
| --- | --- | --- |
| EVD-04 – Deterministic matching finds closest cited spans | ✗ BLOCKED | Evidence pipeline cannot run for a new claim because no request provides `claim_text`. |
| EVD-05 – System reranks candidates and surfaces top options | ✗ BLOCKED | `/claims/{claim_id}/evidence` responds 409 on first load, so reranked candidates never reach the UI. |
| EVD-06 – Present top-N candidates with entail/contrad labels | ✗ BLOCKED | UI rendering exists, but zero evidence ever loads without the missing `claim_text` plumbing. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| --- | --- | --- | --- | --- |
| `frontend/evidence_api.py` | ~172 | Helper calls `/claims/{claim_id}/evidence/export`, but no backend route implements that path | ⚠️ Warning | Any future use of `export_rank_json` will 404 immediately; consider adding the endpoint or removing the helper. |

### Gaps Summary

Both failing truths share the same root cause: the backend requires a `claim_text` payload to build deterministic windows, but neither the UI nor the attachment pipeline ever sends it. As a result, the very first call to `/claims/{claim_id}/evidence` or `/evidence/rerun` raises HTTP 409, no run metadata is persisted, and auto reruns triggered from attachment processing immediately fail as well. Until the claim text is plumbed from the claim registry into the evidence endpoints (or retrieved server-side), users cannot receive any ranked evidence, so Phase 05’s goal remains unmet.

---

Verified: 2026-01-28T10:15:00Z  
Verifier: Claude (gsd-verifier)
