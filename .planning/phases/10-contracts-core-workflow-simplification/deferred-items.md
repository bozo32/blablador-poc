## Deferred Items

### 2026-03-02 — 10-04.5-04 verification environment mismatch

- `bash scripts/dev/verify_10_04_5_01_intake.sh` fails in this workspace for reasons not introduced by 10-04.5-04 UI policy changes:
  1. default corpus paths in the script are absent (`corpus/workflow/*.pdf` not present);
  2. after overriding with local fixtures, strict API scope now requires `X-User-Id` on `/ingest` uploads and the verifier does not provide it (HTTP 422).
- Action deferred: align verifier inputs/headers with current strict scope contract in a dedicated plan.

### 2026-03-03 — citation-context missing `X-User-Id` symptom classification

- Symptom: `/ingest/{doc_id}/citation-context` responds with missing `X-User-Id` even though strict scope headers are required in current code/tests.
- Classification: operational stale `app-ui` image/bundle (frontend still running older request-header logic), not a scope-contract rollback signal.
- Mitigation: rebuild/restart UI image (`docker compose build app-ui && docker compose up -d app-ui`), then re-test before opening scope-contract changes.

### 2026-03-03 — scope bootstrap status semantics are still string-based

- `frontend/scope_lock.py` now exposes `scope_sync_error` for bootstrap failures, but the status surface remains free-form string text in Streamlit session state.
- Deferred follow-up: move bootstrap sync outcomes to a small typed status contract (`ok` / `missing-user` / `backend-error`) so UI warnings are stable and testable without string matching.
