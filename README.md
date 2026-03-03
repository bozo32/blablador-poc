# OS-ERIN

OS-ERIN is a local-first system for turning citing sentences into reviewable claims and testing those claims against cited sources.

Status: alpha / work-in-progress. Expect breaking changes.

## Quickstart (Docker Compose)

Prereqs: Docker + Docker Compose.

Start the full stack (API + UI + GROBID + Postgres + MinIO):

```bash
bash scripts/dev/up.sh
```

Open:

- UI: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

Smoke test (upload + extract + fetch):

```bash
bash scripts/dev/smoke_ingest.sh fixtures/sample.pdf
```

Stop (removes volumes / persisted data):

```bash
bash scripts/dev/down.sh
```

## What Runs Where

This repo runs as multiple services under Compose (see `docker-compose.yml`):

- `app-api`: FastAPI backend (`backend/main.py`)
- `app-ui`: Streamlit UI (`frontend/ui.py`)
- `grobid`: TEI extraction service
- `postgres`: durable job/attempt metadata (“spine”)
- `minio`: S3-compatible object store for PDFs and extraction artifacts

## Data Persistence (Spine)

- Postgres tables: `works`, `attempts`, `jobs`, `artifacts`
- MinIO objects:
  - PDFs: `pdf/{work_id}/{sha256}.pdf`
  - Primary extraction artifacts:
    - `extract/{work_id}/attempts/{attempt_id}/primary/tei.xml`
    - `extract/{work_id}/attempts/{attempt_id}/primary/extraction.json`

## Troubleshooting

- If curl to `http://localhost:8000` hangs/resets, try IPv4: `http://127.0.0.1:8000`

## Specs

- Start here (single entry point): `docs/START_HERE.md`
- Repo functional description + technical spec: `docs/REPO_SPEC.md`
- Identity/scope/visibility contract: `docs/IDENTITY_SCOPE_VISIBILITY_CONTRACT.md`
- EU library citation-walking direction and gaps: `docs/EU_LIBRARY_CITATION_WALKING_PATH.md`
- System design rationale (intent + tradeoffs): `docs/SYSTEM_DESIGN_RATIONALE.md`
- UX style conventions for contributors/agents: `.planning/codebase/UX_STYLE_CONVENTIONS.md`
