---
phase: 04-evidence-attachment
plan: 03
subsystem: attachments
tags: [fastapi, streamlit, grobid, pytest]

# Dependency graph
requires:
  - phase: 04-02
    provides: Attachment queue UI scaffolding and local drop handling
provides:
  - Durable attachment persistence with TEI/embedding artifacts
  - FastAPI endpoints for upload, status polling, and retries
  - Streamlit queue integration backed by live backend polling
  - CLI smoke runner plus sample PDF for scripted verification
affects: [05-evidence-matching, 06-evidence-review]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Attachment timeline logging persisted on disk (last five events)"

key-files:
  created:
    - backend/attachment_store.py
    - backend/attachment_pipeline.py
    - scripts/attachment_smoke.py
    - fixtures/sample.pdf
  modified:
    - backend/main.py
    - backend/schemas.py
    - frontend/attachment_queue.py
    - frontend/ui.py

key-decisions:
  - "Store attachment metadata as JSON on disk for resilience and easy timeline replay"
  - "Provide --local-app smoke mode that runs pipeline without external services"

patterns-established:
  - "Attachment statuses are polled server-side and mirrored in Streamlit with auto-refresh"
  - "CLI-based verification must emit status transitions plus backend timeline details"

# Metrics
duration: 20m
completed: 2026-01-27
---

# Phase 04 Plan 03: Attachment persistence and queue Summary

**Backend attachment store + parsing pipeline with frontend polling/retry integration and CLI smoke coverage**

## Performance

- **Duration:** 20m (approx)
- **Started:** 2026-01-27T18:38:17Z
- **Completed:** 2026-01-27T18:58:44Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Implemented `backend/attachment_store.py` and `backend/attachment_pipeline.py` to copy uploaded PDFs into deterministic folders, capture TEI/NDJSON/embedding artifacts, and log resumable timeline events with single automatic retry.
- Extended FastAPI (`POST /claims/{claim_id}/attachments`, `GET /claims/{claim_id}/attachments/status`, `GET /attachments/{id}`, `POST /attachments/{id}/retry`) plus schemas/settings so statuses survive restarts and background jobs resume automatically.
- Reworked Streamlit queue to upload through the backend, poll server-derived counts, surface the last five backend timeline entries per attachment, and expose a Retry + diagnostics drawer when parsing fails.
- Added `scripts/attachment_smoke.py` (with a sample PDF fixture) to upload a document and print status transitions; supports `--local-app` mode for offline verification and now guards union typing for Python 3.9 compatibility.

## Task Commits

Each task was committed atomically:

1. **Task 1: Build attachment persistence + parsing pipeline** - `e24bdf0` (feat)
2. **Task 2: Integrate queue polling + CLI smoke test** - `6c874e7` (feat)

## Files Created/Modified

- `backend/attachment_store.py` – Disk persistence helpers, timeline logging, resumable metadata
- `backend/attachment_pipeline.py` – TEI parsing, sentence extraction, embedding generation, background dispatcher
- `backend/main.py` – Attachment upload/status/retry endpoints plus resume-on-startup logic
- `backend/schemas.py` – Pydantic models for attachment payloads and responses
- `frontend/attachment_queue.py` – Backend uploads, status polling, retry orchestration, diagnostics caching
- `frontend/ui.py` – Queue panel auto-refresh, backend-driven timeline display, retry button wiring
- `scripts/attachment_smoke.py` – CLI to upload a PDF and print transitions (supports local in-process mode)
- `fixtures/sample.pdf` – Tiny PDF used by the smoke runner and manual testing

## Decisions Made

- Attachment metadata stays in JSON under `data/attachments/{id}` to guarantee restart resilience without additional DB migrations.
- Streamlit queue defers to backend status polling (no more simulated client timers) and signals background processing via auto-refresh + resume banners.
- Smoke testing leverages a `--local-app` flag that runs the attachment pipeline directly with stubbed TEI/extraction so verification is possible without a full FastAPI stack or FAISS installation.

## Deviations from Plan

None – plan executed exactly as written.

## Issues Encountered

- Python 3.9 lacks native union syntax for some Pydantic models; adjusted new annotations to use `typing.Optional/Union` so the CLI and backend imports stay compatible.

## Authentication Gates

None – all operations ran locally without external authentication.

## User Setup Required

None – all services run locally.

## Next Phase Readiness

- Backend now outputs TEI/NDJSON/embedding artifacts per attachment, enabling Phase 05 evidence matching to consume ready-to-index attachment data.
- Frontend queue exposes polling + diagnostic hooks; remaining work for next phases involves wiring retrieval/matching to these ready attachments.

---
*Phase: 04-evidence-attachment*
*Completed: 2026-01-27*
