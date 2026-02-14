# V2 Replumbing Plan (Full Spec)

This document turns the V2 replumbing discussion into an executable plan.
It is written to be falsifiable: each step has concrete checks that can fail,
and fallback/rollback options when assumptions break.

## Objectives

- Make the repo run cleanly on other machines with minimal setup.
- Support remote placement of heavy/external services (GROBID, fallback extraction/OCR/docling, model backends) without changing app logic.
- Preserve current POC behavior while introducing a robust operations spine (jobs/attempts/provenance/artifacts).

## Non-Goals (for this plan)

- Permissions/roles (POC). Do not block adding them later.
- Kubernetes-first.
- GPU requirements (CPU-only).

## Current Baseline (Observed in Repo)

- Backend is FastAPI in `backend/main.py`.
- Frontend is Streamlit in `frontend/ui.py`.
- Ingestion and attachments are persisted on local filesystem (`backend/ingestion_store.py`, `backend/attachment_store.py`).
- GROBID is accessed over HTTP via `backend/grobid_client.py` and `settings.GROBID_URL`.
- ColBERT already supports external API mode (`settings.COLBERT_MODE=external`, `settings.COLBERT_API_URL`).

## High-Level Target Architecture

Run the system as containers with explicit service boundaries.

### Services (Compose)

- `app-api` (FastAPI, port 8000)
- `app-ui` (Streamlit, port 8501)
- `postgres` (metadata/jobs/attempts)
- `minio` (S3-compatible object store for PDFs/artifacts)
- `grobid` (GROBID service)
- `fallback-worker` (HTTP job API for fallback extraction/OCR/docling)
- `model-gateway` (HTTP gateway to LLM/embedding/ColBERT services; can point at HPC)

### Communication

- All services talk over internal HTTP.
- No shared filesystem is required between app and workers.
- PDFs and artifacts live in MinIO/S3; metadata and job state live in Postgres.

### Auth (inter-service)

- POC-safe bearer token between internal services.
- No user auth.

## Contracts (Fully Specified)

### Identifier Mapping (Legacy `doc_id` vs V2 `work_id`)

The current API and UI use `doc_id` (a UUID string) as the ingested document identifier.

V2 introduces the term `work_id` to reflect that this id is the stable identifier for a Work.

During migration:
- Treat `doc_id` == `work_id` (same string) to avoid breaking the existing UI and endpoints.
- New services and DB tables should use `work_id` as the primary key, but API payloads may accept either name for backward compatibility.

### Object Store Layout (Bucket: `works`)

- `pdf/{work_id}/{sha256}.pdf`
- `extract/{work_id}/attempts/{attempt_id}/primary/tei.xml`
- `extract/{work_id}/attempts/{attempt_id}/primary/extraction.json`
- `extract/{work_id}/attempts/{attempt_id}/fallback/pages.jsonl`
- `extract/{work_id}/attempts/{attempt_id}/fallback/body.txt`
- `extract/{work_id}/attempts/{attempt_id}/fallback/refs.json`
- `logs/{work_id}/attempts/{attempt_id}/worker.jsonl`
- `bundles/{work_id}/attempts/{attempt_id}/debug.zip`

Notes:
- JSONL page format keeps writes streaming-friendly.
- `body.txt` exists for consumers that want a single string.

### Postgres Data Model (Minimal)

Tables (names can be prefixed later; columns are the contract):

1) `works`
- `work_id` (text, pk)
- `created_at` (timestamptz)
- `filename` (text)
- `sha256` (text)
- `size_bytes` (bigint)
- `pdf_object_key` (text)
- `active_attempt_id` (text, nullable)
- `tags` (jsonb, default '{}')

2) `attempts`
- `attempt_id` (text, pk)
- `work_id` (text, fk works)
- `kind` (text enum-ish: `primary`, `fallback`, `merge`)
- `state` (text: `queued`, `running`, `succeeded`, `partial`, `failed`, `cancelled`)
- `created_at` (timestamptz)
- `started_at` (timestamptz, nullable)
- `finished_at` (timestamptz, nullable)
- `schema_version` (int)
- `settings_hash` (text)  # idempotency key component
- `settings_json` (jsonb)
- `provenance_json` (jsonb)
- `quality_json` (jsonb)
- `failure_reason` (text, nullable)
- `failure_detail` (text, nullable)

3) `jobs`
- `job_id` (text, pk)
- `attempt_id` (text, fk attempts)
- `worker` (text: `grobid`, `fallback`, `model`)
- `state` (text)
- `progress_json` (jsonb)
- `heartbeat_at` (timestamptz, nullable)
- `created_at` (timestamptz)

4) `artifacts`
- `artifact_id` (text, pk)
- `attempt_id` (text, fk attempts)
- `artifact_type` (text)
- `object_key` (text)
- `bytes` (bigint, nullable)
- `content_type` (text, nullable)
- `created_at` (timestamptz)

#### Invariants

- Attempts are append-only (versioned). A work has one `active_attempt_id`.
- Idempotency: `settings_hash` is derived from `{work_id, kind, settings_json}`.
- Consistency pattern:
  1) create attempt + job row
  2) write artifacts to object store
  3) write artifact rows
  4) finalize attempt state

### Failure Taxonomy (Small Fixed Set)

- `grobid_unavailable`
- `grobid_error`
- `encrypted_pdf`
- `needs_ocr`
- `timeout`
- `limits_hit`
- `parse_error`
- `unknown`

### Fallback Extraction Defaults (carried from 9.1 decisions)

- Hybrid per-page OCR based on text coverage.
- Adaptive DPI escalation: 150 -> 200 -> 300 on low-confidence.
- Index/search immediately with warning badge for OCR/partial.
- Refs gating: prompt at resolution time; default is text-only; offer "replace with higher-quality PDF".

## API Surface (v1)

### `fallback-worker` HTTP API

Auth: `Authorization: Bearer ${INTERNAL_SERVICE_TOKEN}`

1) Create job
- `POST /v1/jobs`
- Body:
  - `work_id` (string)
  - `pdf_object_key` (string)
  - `attempt_settings` (object)
- Response:
  - `job_id` (string)
  - `attempt_id` (string)
  - `state` (string)

2) Poll job
- `GET /v1/jobs/{job_id}`
- Response:
  - `job_id`, `attempt_id`, `state`
  - `progress` (object: phase, pages_total, pages_done, ocr_pages_done, message)
  - `result` (optional pointers + quality flags when terminal)

3) Cancel job
- `POST /v1/jobs/{job_id}/cancel`

### `model-gateway` HTTP API (initial)

Auth: internal bearer token.

- `POST /v1/embeddings`
- `POST /v1/rerank`
- `POST /v1/llm/chat`

The gateway can initially proxy to existing endpoints (Blablador API, ColBERT API) and later add caching/rate limiting.

## Build/Run: "Trigger the Complete Job"

### One-command local run (Docker)

Target end-state command:

- `docker compose up --build`

Expected:
- UI at `http://localhost:8501`
- API at `http://localhost:8000/docs`
- GROBID at `http://localhost:8070`
- MinIO console at `http://localhost:9001`

### Smoke: ingest -> extraction -> fallback

1) Upload PDF
- `curl -F "file=@/path/to/doc.pdf" http://localhost:8000/ingest`

2) Confirm metadata exists
- `curl http://localhost:8000/ingest/{doc_id}`

3) Trigger extraction/resolution (if not auto)
- `curl -X POST http://localhost:8000/ingest/{doc_id}/extract`
- `curl -X POST http://localhost:8000/ingest/{doc_id}/resolve`

4) Force fallback (manual override)
- `curl -X POST http://localhost:8000/ingest/{doc_id}/fallback-extract`
- Poll `GET /ingest/{doc_id}` for new stage/attempt metadata.

## Implementation Plan (Sequenced, With Bullshit Checks)

### Repository Changes (Expected Files/Paths)

This plan assumes we will add the following new files/directories:

- `docker-compose.yml`
- `docker/app-api/Dockerfile`
- `docker/app-ui/Dockerfile`
- `docker/fallback-worker/Dockerfile`
- `docker/model-gateway/Dockerfile`
- `docker/grobid/` (optional; if we just use upstream image, no Dockerfile needed)
- `requirements/app.in`, `requirements/app.lock.txt`
- `requirements/worker.in`, `requirements/worker.lock.txt`
- `backend/db/` (Postgres access + migrations)
- `backend/object_store/` (S3 client wrapper)
- `services/fallback_worker/` (FastAPI app)
- `services/model_gateway/` (FastAPI app)
- `scripts/dev/up.sh` and `scripts/dev/down.sh` (thin wrappers)

And the following existing files will be modified:

- `backend/main.py` (wire new endpoints + worker integration)
- `backend/settings.py` (add envs for Postgres/S3/worker URLs/tokens)
- `backend/ingestion_store.py` (transition strategy: dual-write or migration)
- `frontend/ingestion_api.py` (add fallback endpoints)
- `frontend/ui.py` (UI affordances: force fallback + job progress)

### Milestone 1: Containerize the existing app (no behavior changes)

Deliverables:
- `docker-compose.yml` with `app-api` and `app-ui` running the current code.
- A reproducible dependency install (no conda required to run).

Dependency constraints (to avoid known breakage):
- Pin `numpy<2` in the app containers initially (Streamlit imports pandas; docling imports pandas; the current dev env already demonstrated NumPy 2 ABI breakage).

Bullshit checks:
- `docker compose up --build` starts API+UI.
- `GET /docs` returns 200.
- Existing tests still pass on host.

Rollback:
- Keep the existing host-run workflow (`application.py`) working.

### Milestone 2: Add GROBID as a service (and keep gradle option)

Deliverables:
- Compose adds a `grobid` container.
- App uses `GROBID_URL=http://grobid:8070` in containers.

Gradle option (dev):
- If you run GROBID via gradle on your host, you can set `GROBID_URL` to point to it.

Bullshit checks:
- `POST /api/processFulltextDocument` works from inside the `app-api` container.
- The existing fallback in `backend/ingest_pipeline.py` (fulltext -> header+refs) still works.

### Milestone 3: Introduce Postgres + MinIO (do not migrate data yet)

Deliverables:
- Compose adds `postgres` and `minio`.
- Add a small internal library to talk to S3 and Postgres.
- Add a migration runner (simple startup migrations).

Bullshit checks:
- On fresh start, migrations apply and the app starts.
- MinIO bucket creation is idempotent.

### Milestone 4: Implement the fallback-worker service (job API + S3 writes)

Deliverables:
- New service `fallback-worker` exposing `/v1/jobs`.
- Worker reads PDF from S3 and writes fallback artifacts back to S3.
- Worker updates Postgres attempt/job state.
- Concurrency limits: 1 global OCR worker by default.

Docling/OCR packaging note:
- Keep docling (and its NumPy/Pandas constraints) isolated inside `fallback-worker`.
- Do not add docling to `app-api`/`app-ui` images.

Bullshit checks:
- Create job -> poll -> terminal state.
- Job cancellation results in `cancelled` and stops work.
- Idempotency: submitting the same `{work_id, settings}` returns existing running/recent job.

### Milestone 5: Connect app-api to the worker (without breaking legacy)

Deliverables:
- App persists uploaded PDFs to S3 *in addition to* current filesystem store (transition period).
- Add API endpoints:
  - `POST /ingest/{doc_id}/fallback-extract`
  - `GET /ingest/{doc_id}/fallback-status`
- Add a new ingest stage on the document record for fallback results + quality flags.

Bullshit checks:
- Upload/extract/resolve still works end-to-end with legacy filesystem pipeline.
- When primary extraction fails, fallback can be triggered and produces body text + refs.

### Milestone 6: Tighten ingestion robustness (9.1 behavior)

Deliverables:
- Always attempt GROBID `processReferences` even if header/fulltext fails.
- Persist quality flags on header failure.
- Auto-run fallback (including OCR) on primary failure (as previously decided).
- Keep "never silently replace TEI" invariant.

Bullshit checks:
- Regression test for `processHeaderDocument` 500: references-only still succeeds.
- A representative "GROBID choker" PDF triggers fallback and yields usable body text.

### Milestone 7: Model gateway (remote-ready)

Deliverables:
- `model-gateway` service that the app calls for embeddings/rerank/LLM.
- Gateway routes to:
  - local CPU implementations (optional)
  - external APIs (Blablador)
  - remote HPC services (ColBERT)

Bullshit checks:
- App can run on a laptop with gateway pointing to remote services.
- Evidence reruns still function with ColBERT external mode.

## Open Questions (To Resolve During Execution)

- Dependency locking: do we standardize on `uv` lockfiles or `pip-tools`?
- Where do Postgres/MinIO live in "single VPS" deployment (same box vs remote box)?
- Do we keep filesystem ingestion store as a dev-only mode, or fully migrate to S3+Postgres?

## Recommendation on GROBID (Gradle vs Docker)

For distribution: use GROBID Docker in Compose.

For dev: keep `GROBID_URL` configurable so you can point at a gradle-run GROBID during local iteration.
