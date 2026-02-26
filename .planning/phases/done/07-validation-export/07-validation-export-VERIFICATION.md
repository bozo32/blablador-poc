---
phase: 07-validation-export
verified: 2026-02-02T18:52:45Z
status: passed
score: 4/4 must-haves verified
---

# Phase 7: Validation + Export Verification Report

**Phase Goal:** Users can record a verdict (support/contradict/uncertain) for a claim, add notes, see validated vs unvalidated callout status, and export judgments/metadata as CSV or JSON.
**Verified:** 2026-02-02T18:52:45Z
**Status:** passed
**Re-verification:** No (initial verification)

## Goal Achievement

### Must-Haves (ROADMAP: VAL-01..VAL-04)

| ID | Must-have (observable truth) | Status | Evidence (code) |
| --- | --- | --- | --- |
| VAL-01 | User can record a verdict for a claim (support/contradict/uncertain) | ✓ VERIFIED | `frontend/ui.py:1383` renders verdict UI + save; `backend/schemas.py:567` enforces final requires verdict; `backend/main.py:607` persists via `/claims/{claim_id}/judgment` |
| VAL-02 | User can add annotations/notes to a judgment | ✓ VERIFIED | Notes UI + persistence via `frontend/ui.py:1512`; schema `backend/schemas.py:538`; persistence `backend/judgment_store.py:59` |
| VAL-03 | Citation shows validation status (validated vs unvalidated) | ✓ VERIFIED | Callout status computed from stored judgments (`frontend/judgment_store.py:152`) and applied to chips (`frontend/ui.py:3424`) with styling (`frontend/assets/judgment.css:38`) |
| VAL-04 | User can export judgments and metadata (CSV/JSON) | ✓ VERIFIED | Export endpoint returns raw payload (`backend/main.py:651`); store builders (`backend/judgment_store.py:127`, `backend/judgment_store.py:232`); Streamlit download buttons (`frontend/ui.py:2314`) using raw downloader (`frontend/judgment_api.py:114`) |

**Automated check:** `pytest -q` -> 72 passed, 4 skipped (run during verification).

## Required Artifacts (Existence + Substantive + Wired)

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `backend/judgment_store.py` | Persist per-claim judgments; build claim/callout exports (JSON/CSV) | ✓ VERIFIED | Implements on-disk store + deterministic exporters; filtered final-only when `include_drafts=false` (`backend/judgment_store.py:127`, `backend/judgment_store.py:232`) |
| `backend/schemas.py` | Judgment models + validation rules | ✓ VERIFIED | `JudgmentUpsertRequest` and `JudgmentPayload` enforce final requires verdict (`backend/schemas.py:551`, `backend/schemas.py:574`) |
| `backend/main.py` | Judgment CRUD/list/export endpoints | ✓ VERIFIED | `/claims/{claim_id}/judgment`, `/judgments`, `/judgments/export` wired to store and returns raw Response for exports (`backend/main.py:607`) |
| `frontend/judgment_api.py` | Frontend HTTP helpers incl. raw export download | ✓ VERIFIED | `download_export()` returns `st.download_button`-ready `{content,mime,filename}` without JSON-decoding (`frontend/judgment_api.py:114`) |
| `frontend/judgment_store.py` | Session-backed caching + callout aggregation | ✓ VERIFIED | `callout_status()` marks validated only when any matching claim has `status==final` and `verdict` present (`frontend/judgment_store.py:152`) |
| `frontend/ui.py` | UI: verdict+notes controls, callout indicators, export buttons | ✓ VERIFIED | Inline judgment controls + export sidebar + callout chip classes and navigation (`frontend/ui.py:1231`, `frontend/ui.py:2314`, `frontend/ui.py:3424`) |
| `frontend/assets/judgment.css` | Callout and judgment UI styling | ✓ VERIFIED | Validated/unvalidated + outcome styles present (`frontend/assets/judgment.css:38`) |
| `tests/test_judgment_store.py` | Backend unit tests for rules + exports | ✓ VERIFIED | Covers final/draft rules, export filtering/order, CSV headers (`tests/test_judgment_store.py:20`) |
| `tests/test_judgment_frontend_store.py` | Frontend store tests for caching + callout status | ✓ VERIFIED | Covers validated/outcome rules + target_id fallback (`tests/test_judgment_frontend_store.py:76`) |

## Key Link Verification (Critical Wiring)

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `frontend/ui.py` | `frontend/judgment_store.py` | `JudgmentStore().sync_judgment/save_judgment/sync_doc/callout_status` | WIRED | Judgment controls call `save_judgment` (`frontend/ui.py:1470`); callout chips call `callout_status` (`frontend/ui.py:3424`) |
| `frontend/judgment_store.py` | `frontend/judgment_api.py` | `get_judgment/put_judgment/list_judgments` | WIRED | Store delegates to API and caches results (`frontend/judgment_store.py:49`) |
| `frontend/judgment_api.py` | Backend endpoints | `requests` to `/claims/{claim_id}/judgment`, `/judgments`, `/judgments/export` | WIRED | Raw export download uses `requests.get(.../judgments/export)` (`frontend/judgment_api.py:132`) |
| `backend/main.py` | `backend/judgment_store.py` | `judgment_store.judgment_store.*` | WIRED | Endpoints call store read/upsert/list/export (`backend/main.py:612`) |
| `backend/judgment_store.py` | Export payloads | JSON/CSV bytes returned directly | WIRED | Export functions return `bytes`; API returns `Response(content=payload, media_type=...)` (`backend/main.py:675`) |

## Requirements Coverage (.planning/REQUIREMENTS.md)

| Requirement | Status | Blocking Issue |
| --- | --- | --- |
| VAL-01 | ✓ SATISFIED | - |
| VAL-02 | ✓ SATISFIED | - |
| VAL-03 | ✓ SATISFIED | - |
| VAL-04 | ✓ SATISFIED | - |

## Anti-Patterns Found

No phase-blocking stub patterns found in the Phase 7 artifacts (judgment API/store/UI/export). Existing `FIXME` comments were detected in unrelated backend modules (`backend/parser.py`, `backend/hybrid.py`) and do not affect VAL-01..VAL-04.

## Human Verification Checklist (Recommended)

### 1) Record Draft + Notes

**Test:** In Streamlit, pick an existing claim in the evidence panel; set Status=Draft, Verdict=No verdict; add a Rationale note; click `Save judgment`.
**Expected:** Save succeeds; Notes collapse; refreshing/reloading keeps the saved draft + notes.
**Why human:** Confirms Streamlit state keys + backend connectivity in a real session.

### 2) Final Verdict Enforced

**Test:** Set Status=Final with Verdict=No verdict and attempt to save.
**Expected:** Save button disabled (or error shown); setting a verdict enables save.
**Why human:** Confirms UI guardrails match schema rule and error messaging is clear.

### 3) Callout Validated/Unvalidated Indicators

**Test:** After saving a Final verdict, view Document text and find the corresponding citation chip.
**Expected:** Chip indicator switches from unvalidated to validated styling; outcome style matches Support/Contradict/Uncertain.
**Why human:** Confirms visual affordance and that provenance mapping matches the active callout selection.

### 4) Callout "Open judgment" Navigation

**Test:** Click the small indicator button next to a citation in Document text.
**Expected:** Evidence panel focuses a related claim (judgment controls visible); if no claim exists, view switches toward chasing flow.
**Why human:** Confirms cross-panel navigation and session state coherence.

### 5) Export Downloads (JSON/CSV)

**Test:** In the Export sidebar, download all four files: claims/callouts x JSON/CSV; toggle `Include drafts` on/off.
**Expected:** JSON parses; CSV opens; defaults to final-only when drafts toggle is off; filenames reflect params.
**Why human:** Confirms browser download behavior and file contents usability.

---

_Verified: 2026-02-02T18:52:45Z_
_Verifier: Claude (gsd-verifier)_
