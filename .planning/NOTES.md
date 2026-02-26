# Working Notes (Casual Mode)

<!-- SNAPSHOT:START -->
Updated: `2026-02-16T12:20:00Z`
Branch: `feature/09-1-ingestion-replumbing-solid-spine`
HEAD: `7d4fc6f`

```text
## feature/09-1-ingestion-replumbing-solid-spine
working tree: planning docs in-progress
stash:
  stash@{0}: On main: chore: park opencode plugin bump
```

```text
Recent commits:
  7d4fc6f docs: plan phase 9.2 and spine-everywhere follow-on
  bd0d932 refactor(09.1): remove remaining legacy ingest reads
  13fdddf feat(09.1): default spine-only ingest mode
  c2c10b3 feat(09.1): list ingests via project membership
  22420ea feat(09.1): spine-backed PDF and resolution reads
```

```text
Phase 09.1 status: complete (plans 01..15)
Phase 09.2 status: planned (plans 01..07)
Phase 09.3 status: planned (plans 01..06)

Key runtime configuration:
  - `SPINE_MODE=spine` (default in compose)
  - Postgres + object store are the system of record

Key new spine APIs (additive):
      GET/POST /spine/settings
      GET/POST /spine/workflows
      POST /spine/workflows/{workflow_id}/versions
      POST /spine/locators
      GET /spine/locators/{locator_id}
      GET /spine/document-versions/{document_version_id}/locators

Key dev endpoints:
  - POST /dev/wipe (guarded; resets Postgres/MinIO/local stores)

Compose quickstart:
  - bash scripts/dev/up.sh
  - bash scripts/dev/smoke_ingest.sh fixtures/sample.pdf
```
<!-- SNAPSHOT:END -->

Use this as a lightweight, resumable scratchpad when you're not running a full
GSD plan.

Update it at the start/end of a session so context loss is cheap.

## Goal

- Execute Phase 09.2 (fallback extraction robustness) on top of the spine-only backend.
- Keep configuration Ockham-clean and prevent stacking-error (small sequential edits, verify after each plan).

## Current State

- Branch: `feature/09-1-ingestion-replumbing-solid-spine`
- HEAD: `7d4fc6f`
- Last known green: `bash scripts/dev/up.sh` + `bash scripts/dev/smoke_ingest.sh fixtures/sample.pdf`
- Current behavior:
  - Ingest/extract/resolve are spine-backed (Postgres + object store) with attempts/jobs/artifacts
  - `/dev/wipe` resets local POC state
  - No runtime dependence on `data/ingestion/**`

## What Changed (So Far)

- New plans added:
  - `.planning/phases/09.2-ingestion-automation-robustness-fallback-extraction/*`
  - `.planning/phases/done/09.3-spine-everywhere-legacy-removal/*`
- 09.2 locked decisions:
  - OCR engine: tesseract (worker container)
  - OCR languages: default English, settings-backed/togglable
  - Graph store strategy: hybrid (durable Postgres truth + optional caches)
  - Spine persistence: Postgres metadata/state + object store artifacts; functional dedupe later

## Known Risks / Loose Ends

- Fixture gap: 09.2 needs a reliable "GROBID choker" PDF and a scanned-ish PDF.
- Rights-domain note: global sha256 dedupe is planned, but future rights ponds may require separate identical storage.

## Next 3 Actions

1) Commit remaining planning doc tweaks (PROJECT/V2-PLANNING/09.2 plan edits).
2) Execute `09.2-01-PLAN.md` (fixtures + corpus manifest).
3) Execute `09.2-02-PLAN.md` (fallback contract) once fixtures are in place.

## How To Verify

- Commands:
  - `bash scripts/dev/up.sh`
  - `bash scripts/dev/smoke_ingest.sh fixtures/sample.pdf`
  - `curl -4 -sf -X POST http://127.0.0.1:8000/dev/wipe -H 'Content-Type: application/json' -d '{"confirm":"WIPE"}'`
- Manual checks:
  - API docs load: `http://localhost:8000/docs`
  - UI loads: `http://localhost:8501`

## Rollback Plan

- If this goes sideways, revert commit(s) on this branch.
- Keep runtime simple: `SPINE_MODE=spine` only.
