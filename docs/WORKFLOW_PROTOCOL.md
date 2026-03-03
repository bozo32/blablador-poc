# Workflow Protocol (Stage A-F)

This protocol is the operational definition of the "spine-only" workflow: all durable state lives in Postgres + object store; the UI is never the source of truth.

Primary verifier runner:

```bash
bash scripts/dev/verify_10_04_5_spine_workflow.sh
```

## Preconditions

- You can reach the API at `API_URL` (default `http://127.0.0.1:8000`).
- You have a minimal corpus: 1 citing PDF + 1 source (cited) PDF.

Env overrides used by the verifiers:

- `API_URL` (default `http://127.0.0.1:8000`)
- `CITING_PDF_PATH` (defaults to a repo fixture if `corpus/workflow` is absent)
- `SOURCE_PDF_PATH` (defaults to a repo fixture if `corpus/workflow` is absent)
- `P1`, `P2` project ids (defaults: `proj-a`, `proj-b`)

## Non-Negotiables

- No required local disk state for workflow correctness (no `local_path` contracts).
- Project isolation is enforced via `X-Project-Id` on project-sensitive endpoints.
- "Global" source PDFs are reusable; assigning a source to a claim creates a clone.

## Stage A: Ingestion (Upload + Document Record)

Expected:

- `POST /ingest` accepts multipart bytes and persists the PDF to object storage.
- `GET /ingest` and `GET /ingest/{doc_id}` require `X-Project-Id` and deny cross-project reads.

Verifier coverage:

- `bash scripts/dev/verify_10_04_5_01_intake.sh`

Common failures and remediation:

- `400` missing `X-Project-Id`: ensure clients send the header.
- `422 Empty upload`: confirm PDF path and curl `-F file=@...` usage.

## Stage B: Extraction (Text + References)

Expected:

- `POST /ingest/{doc_id}/extract` reaches `complete` and stores artifacts in the spine.
- `force=true` re-extract is available for diagnosis.

Verifier coverage:

- Stage smoke is exercised indirectly by the intake verifier when `auto_process=true` is used.

Common failures and remediation:

- extraction stuck `running`: check worker container health and queue pause state.
- extraction `error`: inspect extraction artifacts in object store and API logs.

## Stage C: Resolution (Reference Targets)

Expected:

- `POST /ingest/{doc_id}/resolve` produces reference targets with explicit unresolved representations.
- Project-scoped resolution selection requires `X-Project-Id`.

Verifier coverage:

- Intake verifier validates project-scoped ingest isolation; resolution selection scoping is covered in `10-04.5-01` verification.

Common failures and remediation:

- cited targets missing: confirm extraction produced references and resolution ran.
- cross-project resolution bleed: ensure all spine reads/writes are keyed by `project_id`.

## Stage D: Attachments (Cited PDF Pipeline)

Expected:

- `POST /attachments/upload` is bytes-only and project-scoped.
- Attachments dedupe is possible within a project via `content_sha256`.
- Assigning a global source uses `POST /attachments/{id}/clone` (non-mutating).

Verifier coverage:

- `bash scripts/dev/verify_10_04_5_01_intake.sh`

Common failures and remediation:

- attachment never reaches `matched`: check attachment pipeline worker, GROBID availability, and retry limits.
- legacy `local_path` endpoints: should be dev-gated; do not use in web flow.

## Stage E: Evidence (Retrieval + Durable Decisions)

Expected:

- Evidence decisions are recorded as durable events and survive export/import.
- Idempotency and OCC semantics are correct on the decisions event append endpoint.

Verifier coverage:

- `bash scripts/dev/verify_10_04_decisions.sh`

Common failures and remediation:

- `409` version conflicts: client should refresh decisions and retry with latest version.
- export/import does not retain decisions: verify archive includes event log and is imported into spine.

## Stage F: Graph / Surfing (Navigation)

Expected:

- Navigation graph returns non-empty elements with typed nodes.
- Work contexts and reference retrieval dossier are present for at least one cited work.

Verifier coverage:

- `bash scripts/dev/verify_10_03_graph_nav.sh`

Common failures and remediation:

- empty graph: confirm extraction succeeded and graph indexing ran.
- missing contexts: verify reference->ingest linking and ledger edge construction.
- `GET /ingest/{doc_id}/citation-context` reports missing `X-User-Id` even though strict scope code/tests expect it: classify this as stale `app-ui` image/bundle, rebuild and restart UI, then re-test.

```bash
docker compose build app-ui && docker compose up -d app-ui
```

If the runtime stamp also indicates stale API bits, refresh both services:

```bash
docker compose build app-api app-ui && docker compose up -d app-api app-ui
```

## Pruning Discipline (10-04.5)

Prune backlog:

- `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-PRUNE-CANDIDATES.md`

Rule of thumb:

- If something reintroduces UI-local state (especially `local_path`), it is either deleted or explicitly dev-gated and recorded in the prune backlog.
