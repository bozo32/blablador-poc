---
phase: 06-evidence-review-selection
verified: 2026-02-01T22:35:54Z
status: gaps_found
score: 5/9 must-haves verified
gaps:
  - truth: "User sees evidence spans highlighted in the cited PDF"
    status: failed
    reason: "Frontend does not render a PDF viewer or highlight spans inside a PDF; it only stores jump metadata and shows an informational hint."
    artifacts:
      - path: "frontend/ui.py"
        issue: "No PDF viewer/highlight rendering; `_render_pdf_notice()` only displays page/fragment text."
      - path: "frontend/evidence_store.py"
        issue: "`open_candidate_pdf()` only stores `pdf_jump` metadata; it does not open/render a PDF with highlights."
    missing:
      - "A PDF rendering surface in the Streamlit app (or integrated viewer)"
      - "Apply highlight/anchor rendering in the PDF based on span metadata (at minimum jump-to-page; ideally bbox highlights)"
  - truth: "User can choose support/contradict/uncertain/none per claim-segment and the selection persists"
    status: failed
    reason: "UI does not let the user pick a primary/secondary candidate and does not load saved selections; current save path omits required primary fields so backend validation will 422 for support/contradict."
    artifacts:
      - path: "frontend/ui.py"
        issue: "Calls `store.save_selection(... verdict=...)` without `primary_candidate_id`; no `sync_selection()` usage; no primary/secondary pickers."
      - path: "backend/schemas.py"
        issue: "Enforces `primary` required for support/contradict and `note` required for uncertain."
    missing:
      - "UI control to select a primary candidate (candidate_id + attachment_id + span_id)"
      - "Optional secondary selections with required per-entry rationale"
      - "Load persisted selection on claim change (call `store.sync_selection`) and render it"
      - "Display/handle selection API validation errors (e.g., show `selection_error`)"
  - truth: "When verdict is uncertain, UI requires a note before saving"
    status: failed
    reason: "UI labels the note as optional and does not block save; backend requires `note` when `verdict=uncertain`."
    artifacts:
      - path: "frontend/ui.py"
        issue: "Renders 'Optional note' for the verdict flow and allows save without enforcing the uncertain-note rule."
      - path: "backend/schemas.py"
        issue: "`EvidenceSelectionUpsertRequest` rejects uncertain without note."
    missing:
      - "Client-side validation: disable Save (or show inline error) when verdict=uncertain and note is empty"
  - truth: "When attachments are not parsed or no candidates exist, UI clearly communicates status and disables selection"
    status: partial
    reason: "UI shows empty/excerpt-unavailable messaging, but there is no explicit disabled selection state tied to attachment readiness (save controls remain available)."
    artifacts:
      - path: "frontend/ui.py"
        issue: "`Source` view shows excerpt-unavailable info, but selection controls are not disabled based on attachment parse/match status."
    missing:
      - "Detect attachment readiness / excerpt availability and disable selection save when evidence cannot be reviewed"
      - "Clearer empty state when attachments are still parsing vs truly no candidates"
---

# Phase 6: Evidence Review + Selection Verification Report

**Phase Goal:** Users can inspect evidence in-context and choose the best match.
**Verified:** 2026-02-01T22:35:54Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | API can return an excerpt around an attachment sentence_id without crossing paragraph boundaries | ✓ VERIFIED | `backend/attachment_spans.py` implements paragraph-bounded `excerpt()`; enforced by `tests/test_attachment_spans.py` (`test_excerpt_does_not_cross_paragraph_boundaries`). |
| 2 | API can return stable jump metadata (page + section path) for a given attachment span | ✓ VERIFIED | `backend/attachment_spans.py#jump` returns `page` + `section_path`; FastAPI endpoint `GET /attachments/{attachment_id}/spans/{span_id}/jump` in `backend/main.py`. |
| 3 | API persists and returns an evidence selection for a claim_id (primary + optional secondary + notes) | ✓ VERIFIED | `backend/main.py` exposes `GET/PUT /claims/{claim_id}/evidence/selection`; validation rules in `backend/schemas.py`; persistence via `backend/evidence_selection_store.py`; tested in `tests/test_evidence_selection_store.py`. |
| 4 | User can expand a candidate to see an in-source excerpt with highlighted span context | ✓ VERIFIED | `frontend/ui.py` `Source` view uses `store.preview_excerpt(...)` and renders `is_highlight` with `.evidence-review__sentence--highlight`; CSS in `frontend/assets/evidence_review.css`. |
| 5 | User can browse candidates grouped by TEI section path (fallback Body) with top hits shown first | ✓ VERIFIED | `frontend/ui.py` takes top 5 as “Top hits”, groups remainder by `section_path` with `Body` fallback and renders under “By section”. |
| 6 | User sees evidence spans highlighted in the cited PDF | ✗ FAILED | No in-app PDF viewer/highlighting; UI only shows a “PDF span ready…” hint from stored jump metadata (`frontend/ui.py`). |
| 7 | User can choose support/contradict/uncertain/none per claim-segment and the selection persists | ✗ FAILED | `frontend/ui.py` only saves a verdict+note (no primary candidate); backend requires `primary` for support/contradict (`backend/schemas.py`), and UI never loads saved selection (`frontend/ui.py` has no `sync_selection` call). |
| 8 | When verdict is uncertain, UI requires a note before saving | ✗ FAILED | UI note is optional; backend rejects uncertain without note (`backend/schemas.py`). |
| 9 | When attachments are not parsed or no candidates exist, UI clearly communicates status and disables selection | ◐ PARTIAL | UI shows empty/excerpt-unavailable messages, but does not disable selection save based on attachment readiness. |

**Score:** 5/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|---------|----------|--------|---------|
| `backend/attachment_spans.py` | TEI parsing + excerpt/jump helpers | ✓ VERIFIED | Exists (263 lines); implements `AttachmentSpanIndex.for_attachment/jump/excerpt`. |
| `backend/evidence_selection_store.py` | On-disk selection persistence keyed by claim_id | ✓ VERIFIED | Exists (87 lines); JSON read/write; root dir derived from settings. |
| `backend/main.py` | Excerpt/jump + selection endpoints | ✓ VERIFIED | Exists (904 lines); endpoints for `.../excerpt`, `.../jump`, and `.../evidence/selection`. |
| `backend/schemas.py` | Pydantic models for excerpt/jump/selection | ✓ VERIFIED | Exists (531 lines); includes `AttachmentSpan*` and `EvidenceSelection*` models + validators. |
| `frontend/evidence_api.py` | HTTP helpers for excerpt/jump/selection | ✓ VERIFIED | Exists (242 lines); has `fetch_span_excerpt`, `fetch_span_jump`, `get_evidence_selection`, `put_evidence_selection`. |
| `frontend/evidence_store.py` | Client-side selection state + API wiring | ✓ VERIFIED (but unused features) | Exists (648 lines); implements `sync_selection/save_selection/preview_excerpt`, but `sync_selection` is not used by UI. |
| `frontend/ui.py` | Evidence review UI + selection controls | ⚠️ PARTIAL | Exists (3612 lines); Source view + excerpt highlights exist; selection UI is incomplete/broken vs backend contract. |
| `frontend/assets/evidence_review.css` | Styling for excerpt highlights + muted candidates | ✓ VERIFIED | Exists (87 lines); loaded by `frontend/ui.py#inject_evidence_review_styles`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `backend/main.py` | `backend/attachment_spans.py` | `AttachmentSpanIndex.for_attachment()` | ✓ WIRED | Endpoints call `attachment_spans.AttachmentSpanIndex...`. |
| `backend/main.py` | `backend/evidence_selection_store.py` | `selection_store.read/upsert` | ✓ WIRED | `GET/PUT /claims/{claim_id}/evidence/selection` uses `selection_store`. |
| `backend/evidence_matching/serializers.py` | UI “open PDF” flow | `metadata.span_id/anchor_id` | ✓ WIRED | Serializer sets `metadata.span_id`/`anchor_id`; UI reads `metadata.get('span_id') or ...`. |
| `frontend/evidence_api.py` | `/attachments/{attachment_id}/spans/{span_id}/excerpt` | `requests.request` | ✓ WIRED | `fetch_span_excerpt()` calls the endpoint with query params. |
| `frontend/evidence_api.py` | `/claims/{claim_id}/evidence/selection` | `requests.request` | ◐ PARTIAL | API helper exists and store calls it, but UI does not provide required payload fields (primary) or load saved selection. |

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|------------|--------|----------------|
| EVD-07 | ✗ BLOCKED | No in-PDF highlight/anchor viewer implemented (only excerpt highlighting; UI provides at most a “PDF hint” message). |
| EVD-08 | ✗ BLOCKED | UI cannot select a primary evidence span/candidate and persist it; save flow omits required `primary` fields. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `frontend/ui.py` | 1827 | Save selection without required primary | 🛑 Blocker | Support/contradict saves will fail backend validation; persistence goal not achievable. |

### Human Verification Required

1. Evidence excerpt rendering

**Test:** In `Evidence view → Source`, expand a top hit for a claim with matched attachments.
**Expected:** Excerpt renders 2-before/1-after within the same paragraph; highlight is applied to the anchor sentence; section label shows.
**Why human:** Rendering + UX clarity can’t be verified structurally.

2. Selection UX (after gaps are fixed)

**Test:** Pick a primary candidate, set verdict support/contradict/uncertain/none, save, refresh.
**Expected:** Saved selection reloads and is displayed; uncertain requires note; secondaries require rationale.
**Why human:** Needs end-to-end interaction and reruns.

### Gaps Summary

Backend APIs for span excerpt/jump and selection persistence are present and test-covered, and the Streamlit Source view can render highlighted excerpt windows.

The phase goal is not achieved because the UI does not implement (or enforce) the selection contract required to actually choose and persist the “best match” evidence span per claim: there is no primary/secondary selection UI, no loading of saved selection, and the current save path can generate payloads the backend rejects.

---

_Verified: 2026-02-01T22:35:54Z_
_Verifier: Claude (gsd-verifier)_
