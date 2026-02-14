# V2 Planning: Robust Plumbing + Operations UI

## Why V2 Exists

V2 is the shift from a single-machine POC to a system that runs cleanly on other machines (student laptops, a single VPS) and can offload heavy extraction/model workloads to remote Linux hosts (HPC/beefy box) with minimal friction.

Phase 9.2 focuses on ingestion robustness and fallback extraction behavior. V2 adds the distribution + operational surface area (storage/job management) that makes 9.2 practical at scale.

## V2 Goals

- Deterministic, repeatable installs across platforms (macOS Apple Silicon, Windows, Linux x86_64).
- Remote-capable architecture: services can move to a beefy Linux box/HPC edge node without changing application logic.
- Observable ingestion: job attempts, progress, failure reasons, provenance, and metrics are first-class.
- Operations UI: a traditional interface shell (top menus) with a Works Manager to inspect and correct ingestion/storage issues.

## Non-Goals (V2)

- Full permissions / multi-tenant auth. This remains a POC. Design should not block adding roles later, but we will not implement them in V2.
- Kubernetes-first deployment.

## Proposed System Shape (Robust Defaults)

### Services

- `app` (API + UI orchestrator)
- `grobid-svc` (primary TEI/refs extraction)
- `fallback-extract-svc` (PyMuPDF fallback + optional OCR; docling isolated here if used)
- `model-gateway` (one internal API that routes to LLM/embedding/ColBERT services, local or remote)
- `postgres` (system of record: docs, attempts, stages, provenance, quality flags)
- `object-store` (MinIO locally; S3-compatible in prod)

### Communication Pattern

- Internal HTTP between services (no shared filesystem assumptions).
- App stores/reads state in Postgres; large artifacts in object store.
- Workers are stateless and can run anywhere that can reach Postgres + object store.

### Storage Model

- PDFs stored in object store.
- Extraction outputs stored in object store (e.g., fallback pages text, debug bundles).
- Postgres stores pointers (object keys), attempt metadata, and quality/provenance.

### Inter-service Auth (POC-safe)

- Private network + service tokens.
- All service-to-service calls include a shared bearer token (scoped per service if easy).

### Versioning/Compatibility Requirements

- Worker HTTP endpoints are versioned (`/v1/...`).
- Stored outputs include `schema_version` and extractor versions/config.

## Workstreams (Developmentally Separable) + Sequence

V2 is intentionally split into separable tracks so we can ship robustness before redesigning UI.

1) Platform + Deployment Plumbing (containers + remoting)
2) Ingestion Execution Spine (jobs/attempts/provenance/artifacts)
3) Operations UI (Works Manager + top-menu shell)
4) UX Polish Backlog (attachment/workspace ergonomics)

5) Legacy Hygiene (deprecate filesystem ingestion store)

Goal: make Postgres + object store the source of truth so no shared filesystem is required.

Scope:

- Move ingest listing + dedupe to Postgres (no directory scans)
- Run extraction from S3 PDF objects (not local `source.pdf`)
- Serve extraction artifacts from S3 via the artifacts table (not local `extraction/tei.xml`)
- Update UI to read spine-backed status/artifacts
- Remove dual-write and `data/ingestion/**` dependence once compatibility is no longer needed

## Deployment + Development Strategy

### Distribution (Primary)

- Docker + Docker Compose is the official "it runs" path.
- Compose supports two modes:
  - Local: everything on one machine (app + workers + Postgres + MinIO)
  - Remote: app local (or VPS), workers remote; all share Postgres + object store

### Development (Fast Inner Loop)

- Default dev: run everything in Compose with bind-mounted source for fast reload.
- Optional dev: run only infra/workers in Docker; run app locally on host, pointing env vars at Docker services.

## UI Direction: Traditional Shell + Works Manager

Streamlit is productive for POCs, but a "traditional" top-menu interface (dropdown menus at top: `Program`, `Project`, `Works`) tends to fight Streamlit defaults.

V2 introduces a UI shell that supports operations.

### Proposed IA (Information Architecture)

- `Program`
  - Program picker/summary
  - Defaults (limits, model routing profile)
  - Metrics overview (ingestion success, failure reasons, storage usage)

- `Project`
  - Project settings (model profile, ingestion limits, reference resolution policy prompts)
  - Project corpora / tags

- `Works` (Works Manager)
  - Works list with filters: extraction status, OCR/partial, failure reason, date, project/program
  - Work detail:
    - Attempt timeline (versioned attempts; active attempt selection)
    - Provenance + quality flags
    - Links to stored artifacts
    - Preview of extracted text + refs
  - Actions:
    - Cancel job
    - Re-run fallback with settings (language candidates, thresholds, DPI policy)
    - Trigger/skip reference resolution (with "low-quality refs" prompt)
    - Mark "replace with higher-quality PDF" (and attach replacement)
    - Prune old attempts (keep last N)
    - Export debug bundle (logs + metadata + artifact pointers)

## UX Polish Backlog (From `.planning/UX-NOTES.md`)

This backlog is intentionally sequenced after the ingestion/extraction spine is real and observable.

### Attachment Workflow Improvements

1) Global attachment inbox drop zone
   - Why: dropping onto individual claims does not scale for long docs; a single drop target with auto-match + manual reassignment is faster and reduces misdrops.

2) Hide demo placeholder claims once real data flows
   - Why: demo cards are useful for development but confusing for users; the workspace should mirror the real claim list.

3) Three-pane layout (VS Code style)
   - Why: left nav/project context, center claim workspace, right citation follow/view. Streamlit’s current layout can feel cramped; consider custom CSS or another UI framework once the ops spine is stable.

4) Real processing status integration
   - Why: replace simulated timers with actual backend job progress so reviewers trust status data.

5) Attachment timeline ergonomics
   - Why: timelines scroll off-screen; prefer condensed summaries (latest status + hover/expand for history).

## Reliability/Robustness Notes (Carry Into V2)

- Correlation IDs: propagate `attempt_id`/`job_id` across app->worker->grobid->gateway; include in logs and stored metadata.
- Idempotency: dedupe on `doc_id + settings_hash`.
- State machine: `queued/running/succeeded/partial/failed/cancelled`.
- Consistency: create attempt record first, then write artifacts, then finalize attempt state. Mark `needs_reconcile` on crashes.
- Storage caps:
  - Keep last N attempts per work (default 5)
  - Default `store_intermediates=false`

## Open Questions (V2)

### A) UI Technology Choice (Streamlit vs something else)

- Do we keep Streamlit and approximate top menus (sidebar navigation / tabs), or do we switch to a more traditional web UI?
- If switching: what is the preferred stack?
  - FastAPI + a small frontend (HTMX + templates)
  - FastAPI + React/Next
  - Another lightweight UI layer
- What is the minimum UI polish required for students (low friction) vs for operators (Works Manager power tools)?

### B) Works/Project/Program Data Model

- What are the minimum fields to define Program/Project (names, tags, default profiles)?
- Are Works global and tagged into projects, or owned by a single project?

### C) Remote Services + Networking

- How will laptops reach the remote worker/services: VPN (WireGuard), SSH tunnels, or campus private network?
- Where will Postgres and the object store live (VPS vs beefy box)?
- Do we need an API gateway/reverse proxy (Caddy/Traefik/Nginx) for TLS and routing?

### D) Storage Operations

- Do we need a garbage collector for orphaned artifacts (S3 keys without DB references)?
- Do we need per-project quotas or just global monitoring?
- What is the default backup strategy for Postgres and object storage?

### E) Job/Worker Operations

- How do we surface worker health and version skew in the UI?
- What is the retry policy (which failure reasons are retryable)?
- Do we need priorities (interactive vs batch ingestion)?

### F) Model Gateway

- What are the model backends that must be supported in V2 (LLM, embeddings, ColBERT)?
- Should the gateway implement caching/rate limiting for slow student machines?

## Suggested Milestone Split (So V2 Does Not Balloon)

1) Containerize and split services (Compose, Postgres, MinIO, Grobid, fallback worker, model gateway skeleton).
2) Harden ingestion attempt model (IDs, logs, schema versions, artifact pointers).
3) Build Works Manager UI with attempt timeline and re-run/cancel/prune actions.
4) Add model gateway routing to remote services.

5) Apply UX polish backlog (attachment inbox, real status, 3-pane layout, timeline ergonomics).
