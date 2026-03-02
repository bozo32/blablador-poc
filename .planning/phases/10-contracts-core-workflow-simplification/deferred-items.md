## Deferred Items

### 2026-03-02 — 10-04.5-04 verification environment mismatch

- `bash scripts/dev/verify_10_04_5_01_intake.sh` fails in this workspace for reasons not introduced by 10-04.5-04 UI policy changes:
  1. default corpus paths in the script are absent (`corpus/workflow/*.pdf` not present);
  2. after overriding with local fixtures, strict API scope now requires `X-User-Id` on `/ingest` uploads and the verifier does not provide it (HTTP 422).
- Action deferred: align verifier inputs/headers with current strict scope contract in a dedicated plan.
