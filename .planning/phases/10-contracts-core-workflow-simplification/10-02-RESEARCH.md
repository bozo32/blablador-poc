# Phase 10-02: Core Workflow Simplification (Happy Path) - Research

**Researched:** 2026-02-21
**Domain:** Background orchestration + run status + reviewer workflow surfaces (FastAPI + Streamlit) over Phase 10-01 pipeline contracts
**Confidence:** MEDIUM

## Summary

This repo already has (1) durable ingest+extract via the spine (Postgres attempts/jobs + S3 artifacts), (2) durable cited-PDF processing via attachments (Postgres + S3 artifacts, background threads, global pause), and (3) a working evidence rerun orchestrator (threaded queue + per-claim locking) that produces candidate+NLI results and persists them (Postgres + S3) and reviewer selections (Postgres) while mirroring into the span graph. Phase 10-01 added immutable, write-once pipeline stage artifacts under `pipeline/{run_id}/{stage}.json` plus minimal `/pipeline/*` endpoints.

Plan 10-02’s “thin orchestrator/state machine” should therefore *not* invent a new compute system. It should adapt existing ingestion, attachment processing, and evidence matching code paths into a claimspan-centric run model that: mints a new `run_id` on finalize, advances stages when prerequisites exist, writes immutable stage artifacts once, and maintains mutable per-target run status in new tables designed explicitly for progress, retries, cancellation, and SSE/polling.

**Primary recommendation:** Implement a `HappyPathOrchestrator` service modeled after `backend/evidence_matching/service.py` (threaded queue + global pause checks), but persist mutable state in new `pipeline_run_status/*` tables and keep *only* the final per-stage outputs in the Phase 10-01 immutable contract store.

## Standard Stack

### Core
| Library/Tool | Version (repo) | Purpose | Why Standard (in this repo) |
|---|---:|---|---|
| Python | 3.10 | Backend runtime | Matches compose + existing backend |
| FastAPI | 0.129.0 | API + streaming endpoints | Existing `backend/main.py` routing/tests |
| Pydantic | 2.12.5 | Request/response + contract models | Existing schemas + Phase 10 contracts |
| Postgres | (compose) | Spine + new run-status tables | All durable state uses Postgres |
| S3/MinIO | (compose) | Immutable artifacts + blobs | `backend/object_store/s3.py` wrapper |
| Streamlit | 1.54.0 | UI | Current frontend is Streamlit-only |

### Supporting
| Library/Tool | Version (repo) | Purpose | When to Use |
|---|---:|---|---|
| pytest | (repo lock) | Tests | Add orchestrator + status + API tests |
| streamlit-autorefresh | 1.0.1 | Polling refresh | Use for run-status polling in UI |

## Architecture Patterns

### Existing Background-Orchestration Pattern (Use This)

**Evidence reruns:** `backend/evidence_matching/service.py`
- Threaded workers + in-memory queue (`deque`) + per-claim locking
- Global pause obeyed via `backend/background_state.py`
- Durable outputs in `evidence_runs` (S3 JSON + Postgres pointer) and `evidence_selections`

**Attachments:** `backend/attachment_pipeline.py`
- One background thread per attachment; updates mutable status rows in Postgres
- Writes durable artifacts to S3 and pointers to Postgres

**Locked 10-02 constraint:** stage artifacts are immutable write-once per `(run_id, stage)` (Phase 10-01). Therefore progress must be tracked outside the contract artifacts.

### Recommended Implementation Shape (Thin, Repo-Consistent)

**1) Orchestrator service (new):**
- A single orchestrator module that can be called from API endpoints.
- Uses background threads and checks `background_state.get_state().get("paused")`.
- Writes immutable stage artifacts via `backend/pipeline_contracts/service.py:store_stage()`.
- Writes mutable status + events via new `backend/spine/pipeline_run_status.py` helpers.

**2) Status/event persistence (new tables):**
- Mutable tables for run lifecycle + per-target + per-stage progress.
- Append-only run events for SSE and for debugging.

**3) Derived “requested works” queue (no new durable store required for MVP):**
- Derive queue rows from extraction+resolution plus existing `attachments` state.
- Use attachment placement (`PATCH /attachments/{attachment_id}`) as the persistence surface for manual assignment decisions.

### Recommended Project Structure (candidate)
```
backend/
├── workflow_happy_path/
│   ├── orchestrator.py         # run state machine + background scheduling
│   ├── builders.py             # stage payload builders (extract/citespans/...)
│   └── events.py               # SSE formatting helpers
├── spine/
│   ├── pipeline_run_status.py  # mutable status + events persistence
│   └── pipeline_run_scopes.py  # reviewer/claimspan -> run history index
└── main.py                     # new /workflow/* routes
```

## Requirement Deep Dive (Repo-Specific)

### 1) Background orchestration: ingest → extract → citespans → retrieval → filter → rerank → nli → assessment

#### What already exists (reuse)
- **Ingest + auto-enqueue extraction:** `backend/main.py` `POST /ingest` enqueues extraction via `spine_extract_pool.enqueue()` when `auto_process=true` and background is not paused.
- **Extraction persistence:** `backend/spine/extraction_pool.py` + `backend/spine/artifacts.py` store `tei.xml`, `extraction.json`, `resolution.json` (after `/ingest/{doc_id}/resolve`).
- **Citation windows + deterministic span ids:** `backend/text_selectors.py` + `backend/span_graph_store.py` (`upsert_span`, `_span_id`, `upsert_citation_span_index`).
- **Cited PDFs processing:** `backend/attachment_store.py` + `backend/attachment_pipeline.py` process PDFs into `attachments/{id}/sentences.ndjson` and mark `status=matched`.
- **Candidate scoring + NLI labels:** `backend/evidence_matching/*`:
  - windows: `backend/evidence_matching/loaders.py:load_claim_windows()`
  - deterministic seeding (BM25): `backend/evidence_matching/deterministic_matcher.py:seed_windows()`
  - NLI: `backend/evidence_matching/pipeline.py` calls `backend/nli.assess()`
  - persistence: `backend/evidence_matching/store.py` (S3 + Postgres) and `backend/evidence_selection_store.py`
- **Immutable pipeline contract store:** `backend/contracts/pipeline_v1.py`, `backend/spine/pipeline_artifacts.py`, `/pipeline/*` routes.

#### What must be built for 10-02
- **A claimspan-centric orchestrator** that:
  - mints a new `run_id` when claimspan is finalized;
  - computes/loads prerequisites (extract+resolution for citing work; attachment availability for targets);
  - executes stage builders and stores final stage artifacts write-once;
  - isolates failures per target and continues;
  - retries/timeouts within the same `run_id` while keeping contract artifacts immutable.

#### Concrete mapping of stages to existing code paths
- `ingest` (precondition, not a Phase 10-01 contract stage): use existing `POST /ingest` + spine state (`GET /ingest/{doc_id}` / `GET /ingest/{doc_id}/spine`).
- `extract` (contract stage): build from spine’s `extraction.json` (+ optional `resolution.json`) using `backend/spine/ingest_view.py:build_ingested_document_from_spine(include_extraction_data=True)`.
  - Use `backend/extraction.py` payload as `ExtractData.structured_doc`.
  - Emit `ExtractData.citation_anchors` using `structured_doc["citations"]` (keeps citation order).
- `citespans` (contract stage): for each citation anchor:
  - `selector = text_selectors.build_anchor_quote(window_text)`
  - `window_fingerprint = text_selectors.fingerprint(window_text)`
  - `span_graph_store.upsert_span(kind='citation_window', selector=..., window_fingerprint=..., ingest_id=citing_doc_id)`
  - `span_graph_store.upsert_citation_span_index(ingest_id, citation_index, target_id, span_id)`
  - Store `CiteSpansData.by_target[target_id].anchors[]` with `span_id`, `citation_index`, `selectors`, `window_fingerprint`.
- `retrieval` (contract stage, MVP): reuse evidence window seeding as “retrieval”:
  - load windows from ready attachments for this claimspan via `loaders.load_claim_windows(claim_id)`
  - group resulting candidates by attachment’s `target_id` (available on `attachments` rows) to populate `CandidateStageData.by_target`.
- `filter` (contract stage, MVP): deterministic post-processing:
  - de-dupe by normalized text (repo already does similar in `deterministic_matcher.seed_windows()`)
  - cap to `caps.candidates` per target.
- `rerank` (contract stage, optional depending on settings):
  - use `backend/utils.py:rerank()` when `RERANKER_MODEL` configured (cross-encoder), or skip and copy `filter` candidates.
- `nli` (contract stage):
  - reuse `backend/evidence_matching/pipeline.py` NLI call pattern (`backend/nli.assess`) but run per-target list (cap to `caps.nli`).
- `assessment` (new contract stage in 10-02):
  - written when reviewer finalizes an assessment; see requirement (5).

### 2) Running state + per-target progress without breaking immutability

#### Why this is needed
Pipeline stage artifacts are write-once per `(run_id, stage)` (`pipeline_stage_artifacts_run_stage_uniq` + `StageArtifactAlreadyExists`). That design intentionally prevents “update artifact as progress.” Therefore the orchestrator must maintain mutable run progress elsewhere.

#### Recommended schema (additive, does not touch immutability)

**A) Run scope index (append-only run history)**
- Table: `pipeline_run_scopes`
- Purpose: query “latest run for (reviewer_uid, claimspan_id)” efficiently without JSONB scanning.
- Columns (proposed):
  - `scope_type text` (locked use: `'claimspan'`)
  - `scope_id text` (recommended: reuse existing claim id string, e.g. `cite:{doc}:{idx}:{reviewer}:{seg}`)
  - `reviewer_uid text`
  - `run_id text REFERENCES pipeline_runs(run_id) ON DELETE CASCADE`
  - `created_at timestamptz DEFAULT now()`
  - `citing_doc_id text` (redundant but simplifies UI queries)
  - Index: `(scope_type, scope_id, reviewer_uid, created_at desc)`

**B) Mutable run status (one row per run)**
- Table: `pipeline_run_status`
- Columns (proposed):
  - `run_id text PRIMARY KEY` (FK to `pipeline_runs`)
  - `reviewer_uid text`, `scope_id text`, `citing_doc_id text`
  - `state text` (`queued|running|blocked|complete|partial|error|cancelled`)
  - `started_at/finished_at timestamptz`, `updated_at timestamptz`
  - `error_json jsonb` (last error summary)
  - `metrics_json jsonb` (durations per stage, counts)

**C) Per-target status (one row per target per run)**
- Table: `pipeline_target_status`
- PK: `(run_id, target_id)`
- Columns (proposed):
  - `target_id text` (canonical target; can be `ref:{citing_doc}:{reference_id}` fallback)
  - `reference_id text NULL` (original bibl id)
  - `attachment_id text NULL` (when available)
  - `state text` (`requested|available|running|done|error|cancelled|blocked`)
  - `stage_state_json jsonb` (per-stage: `{extract:..., citespans:..., retrieval:..., ...}`)
  - `attempts_json jsonb` (retry counts per stage)
  - `last_error_json jsonb`
  - `updated_at timestamptz`

**D) Append-only events for SSE/debug**
- Table: `pipeline_run_events`
- Purpose: SSE feed + postmortem audit.
- Columns (proposed):
  - `event_id bigserial PRIMARY KEY`
  - `run_id text`, `target_id text NULL`
  - `seq int` (or use `event_id` as seq)
  - `type text` (e.g. `run_started`, `target_stage_started`, `target_stage_finished`, `target_blocked`, `target_cancelled`)
  - `payload_json jsonb`, `created_at timestamptz DEFAULT now()`
- Index: `(run_id, event_id)`

**Key immutability rule:** only the *final* outcome of a stage is written to `pipeline/{run_id}/{stage}.json`; progress, retries, and partial intermediate snapshots live in these mutable tables (and optionally in a debug-only S3 prefix).

### 3) SSE feasibility in this codebase (Streamlit realities)

#### Backend SSE: feasible
FastAPI can serve SSE using `StreamingResponse` with `media_type="text/event-stream"` and a generator that yields properly formatted lines.

Recommended endpoint shape:
- `GET /workflow/runs/{run_id}/events` (SSE)
  - Query: `after` (last seen `event_id`), `heartbeat_ms`.
  - Events sourced from `pipeline_run_events` (poll DB and yield new events).

#### Streamlit consumption: not realistic for true SSE
This Streamlit UI is rerun-based and already uses polling helpers (`streamlit_autorefresh`) and plain `requests`. There is no existing JS EventSource component for SSE.

**Repo-consistent alternative (recommended for 10-02):**
- Implement SSE server-side (to satisfy architectural intent and keep API future-proof).
- In the Streamlit UI, implement **polling-first** using `st_autorefresh`:
  - `GET /workflow/runs/{run_id}/status` every ~1-2s while a panel is open.
  - If desired, also poll `GET /workflow/runs/{run_id}/events?after=...` as a JSON endpoint (non-SSE) to show a debug log.

This matches the “SSE + polling fallback” intent without requiring a new Streamlit component build.

### 4) Where/how to store reviewer-scoped claimspans + chase state + requested-works queue

#### Claimspans (reviewer-scoped)
**Already persisted (and used in UI today):**
- `backend/judgment_store.py` (`judgments` table) stores reviewer-scoped drafts for callouts and includes `span_selectors`.
- In `frontend/components/chasing_panel.py`, segmentation drafts are saved under a callout-shaped id:
  - `callout:{doc_id}:{citation_index}:{target_id}`

**Recommended 10-02 canonical identifier:**
- Reuse existing reviewer-scoped claim ids already used for attachments/evidence:
  - `cite:{doc_id}:{citation_index}:{reviewer_uid}:{segment_id}`
This avoids introducing a second parallel “claimspan id” namespace and lets you reuse:
- `attachments.claim_id` placement
- `evidence_runs.claim_id` history
- `evidence_selections.claim_id` reviewer selections

#### Chase state (reviewer-scoped)
**Store chase intent + run linkage in new tables (not in immutable artifacts):**
- `pipeline_run_scopes` (append-only history)
- `pipeline_run_status` / `pipeline_target_status` (mutable progress)

#### Requested-works queue
**Derive it; persist only user assignment decisions:**
- Queue items: derived from `extract`/`citespans` stage data (citation order + reference ids) plus `resolution.json` (metadata) and `attachments` table (availability/processing/done).
- User decisions:
  - “this PDF belongs to this requested work” is persisted by patching attachment placement (`PATCH /attachments/{attachment_id}`) to set `doc_id`, `target_id`, and optionally `citation_index`/`claim_id`.

### 5) Mirroring persistence surfaces (what to reuse)

#### Pipeline artifacts (immutable, audit/replay)
- Store stage artifacts in Phase 10-01 store:
  - Postgres pointer: `pipeline_stage_artifacts`
  - S3 object: `pipeline/{run_id}/{stage}.json`
- Reuse `backend/pipeline_contracts/service.py:store_stage()`.

#### Existing evidence stores (keep using)
- **Evidence runs:** `evidence_runs` + S3 `evidence/{claim_id}/{run_id}.json` (`backend/evidence_matching/store.py`).
- **Reviewer evidence selections:** `evidence_selections` (`backend/evidence_selection_store.py`).
- **Reviewer judgments:** `judgments` (`backend/judgment_store.py`) for claim/callout-level notes/verdicts.
- **Span graph assertions (mirroring):** already done in `PUT /claims/{claim_id}/evidence/selection` in `backend/main.py`.

#### New assessment artifact (required by 10-02)
**Add a new pipeline contract stage:** `assessment`.

Recommended mirrored writes when a reviewer records an assessment:
1) Persist in existing stores:
   - `PUT /claims/{claim_id}/evidence/selection` for primary/secondary evidence span picks (per cited work where applicable).
   - `PUT /claims/{claim_id}/judgment` (or span-graph assertion endpoints) for final verdict/notes.
2) Additionally write `pipeline/{run_id}/assessment.json` via `store_stage(run_id, "assessment", ...)`.

Assessment contract payload (MVP) should include:
- `reviewer_uid`, `scope_id` (claim_id), and per-target verdicts (`supports|contradicts|inconsistent|silent`) + selected evidence span ids.

### 6) Tests + docker-compose verifier (extend what exists)

#### Tests to extend/add (repo-aligned)
- `tests/test_pipeline_contract_*` patterns show how to validate immutable artifacts and 409 conflicts.
- `tests/test_evidence_matching_api.py` shows how to test threaded queue behavior with blocking pipelines.
- `tests/test_attachment_pipeline.py` shows how to stub GROBID and embeddings.

Recommended new tests:
- `tests/test_workflow_happy_path_status_api.py`
  - create run scope → trigger run → poll status transitions → cancel target.
- `tests/test_workflow_happy_path_stage_writes.py`
  - orchestrator produces `extract` and `citespans` artifacts (and does not overwrite).

#### Verifier script (docker-compose-backed)
Create a script similar to `scripts/dev/verify_10_01_contracts.sh`.

Recommended: `scripts/dev/verify_10_02_happy_path.sh`
- `docker compose up -d --build`
- wait for `/docs`
- `POST /dev/wipe`
- upload a small PDF fixture (see `scripts/dev/e2e_flow.py` patterns)
- wait for extraction terminal state (`GET /ingest/{doc_id}/spine`)
- trigger claim segmentation confirmation (minimal `POST /claims/confirm` like `scripts/dev/e2e_flow.py:confirm_one_claim`)
- trigger happy-path run for one claimspan
- poll `GET /workflow/runs/{run_id}/status` until done
- fetch stage artifacts via `GET /pipeline/runs/{run_id}/stages/{stage}` and assert presence/shape
- print `OK: verify_10_02_happy_path`

## API Route Candidates (10-02)

Keep Phase 10-01 `/pipeline/*` as the immutable artifact store. Add a small “workflow” surface:

- `POST /workflow/claimspans/{claim_id}/runs`
  - Body: `{reviewer_uid, citing_doc_id, mode?: 'happy_path'}`
  - Effect: create run (`/pipeline/runs` internally), insert `pipeline_run_scopes`, enqueue background processing.
  - Returns: `{run_id}`.

- `GET /workflow/claimspans/{claim_id}/runs/latest?reviewer_uid=...`
  - Returns: `{run_id, state, updated_at}`.

- `GET /workflow/runs/{run_id}/status`
  - Returns: run state + per-target stage states + durations + actionable warnings.

- `GET /workflow/runs/{run_id}/events` (SSE)
  - Streams from `pipeline_run_events`.

- `POST /workflow/runs/{run_id}/targets/{target_id}/cancel`
  - Marks canceled in `pipeline_target_status`; worker checks and stops before writing artifacts.

## Candidate File Touch List (10-02)

Backend:
- `backend/contracts/pipeline_v1.py` (add `assessment` stage model)
- `backend/contracts/upgrade.py` (ensure validation routes new stage)
- `backend/db/migrate.py` (add new status/event/scope tables)
- `backend/spine/pipeline_runs.py` (optional: accept settings_json metadata for scope/reviewer)
- `backend/spine/pipeline_artifacts.py` (no change expected; must remain write-once)
- `backend/main.py` (add `/workflow/*` endpoints + wire orchestrator)
- `backend/workflow_happy_path/orchestrator.py` (new)
- `backend/workflow_happy_path/builders.py` (new)
- `backend/spine/pipeline_run_status.py` + `backend/spine/pipeline_run_scopes.py` (new)

Frontend (Streamlit):
- `frontend/components/chasing_panel.py` (on finalize: call `POST /workflow/claimspans/{claim_id}/runs`)
- `frontend/components/chase_queue.py` or left-rail component (poll run status; show requested works queue derived from API)
- `frontend/ui.py` (wire polling via `st_autorefresh` where appropriate)

Tests/scripts:
- `tests/test_*workflow*` (new)
- `scripts/dev/verify_10_02_happy_path.sh` (new)

## Don’t Hand-Roll

| Problem | Don’t Build | Use Instead | Why |
|---|---|---|---|
| Object store plumbing | ad-hoc boto3 calls | `backend/object_store/s3.py` | MinIO config + consistent semantics |
| Candidate id hashing | stage-specific ids | `backend/contracts/pipeline_v1.py:candidate_id_for()` | Contract-level determinism enforced by service |
| Span id hashing | new algorithm | `backend/span_graph_store.py` + `pipeline_v1.span_id_for()` | Must stay compatible with span graph |
| Background pause | local flags | `backend/background_state.py` | Already persisted + UI-consumed |
| Progress in immutable artifacts | “update stage.json” | new run-status tables | Avoids breaking write-once contract store |

## Common Pitfalls

### Pitfall 1: Stage conflicts due to retries writing the same artifact
**What goes wrong:** retry path calls `store_stage()` twice for the same `(run_id, stage)` and hits 409, leaving status stuck.
**Avoid:** only write the immutable artifact *once per stage* when terminal (complete/partial/error). Track retries in `pipeline_target_status.attempts_json`.

### Pitfall 2: Mixing claim ids / reviewer scoping
**What goes wrong:** attachments/evidence selections were historically stored under legacy (non-reviewer) claim ids.
**Avoid:** follow `EvidenceMatchingService._attachment_claim_aliases()` behavior (fallback aliasing) when looking up attachments for a reviewer-scoped claimspan.

### Pitfall 3: Streamlit “streaming” attempts fighting rerun model
**What goes wrong:** trying to keep an SSE connection open inside Streamlit blocks reruns and creates confusing UI stalls.
**Avoid:** polling-first via `streamlit_autorefresh` and a single `/workflow/*/status` endpoint.

### Pitfall 4: Requested-works queue drifting from extraction order
**What goes wrong:** deriving queue order from unordered maps (`by_target` dict) produces unstable UI ordering.
**Avoid:** preserve citation order using `ExtractData.citation_anchors` (list) and/or store a stable `order_index` per requested work.

### Pitfall 5: “Cancel discards partial results” vs immutability
**What goes wrong:** once a stage artifact is written, you cannot delete/overwrite it without violating Phase 10-01 semantics.
**Avoid:** implement cancellation as “stop before writing further artifacts; mark status canceled; downstream ignores.” Treat already-written artifacts as audit history.

## Code Examples (Repo-Verified)

### Background queue pattern (evidence reruns)
```python
# Source: backend/evidence_matching/service.py
thread = threading.Thread(
    target=self._run_job,
    args=(job,),
    daemon=True,
    name=f"evidence-rerun-{claim_id}",
)
thread.start()
```

### Immutable stage artifact keying
```python
# Source: backend/contracts/pipeline_v1.py
def stage_object_key(run_id: str, stage: str) -> str:
    return f"pipeline/{run_id}/{stage}.json"
```

### Reserve-before-S3-put to enforce write-once
```python
# Source: backend/spine/pipeline_artifacts.py
INSERT ... ON CONFLICT (run_id, stage) DO NOTHING RETURNING artifact_id
```

## Suggested Plan Breakdown (2-4 plans, each 2-3 tasks)

### Plan A: Backend Orchestrator + Status Model
- Task 1: Add Postgres tables + spine helpers for `pipeline_run_scopes`, `pipeline_run_status`, `pipeline_target_status`, `pipeline_run_events`.
- Task 2: Implement `HappyPathOrchestrator` (threaded queue, pause/cancel/retry, per-target stage machine) that writes Phase 10-01 artifacts.
- Task 3: Add `/workflow/*` endpoints for trigger + status + cancel (+ SSE endpoint wired to events table).

### Plan B: Stage Builders (Reuse Existing Compute)
- Task 1: Implement `extract` + `citespans` stage builders from spine extraction/resolution + `SpanGraphStore`.
- Task 2: Implement `retrieval/filter/rerank/nli` builders using `attachment_store` + `evidence_matching` primitives, grouped by `target_id`.

### Plan C: Assessment Persistence (Mirrored)
- Task 1: Extend pipeline contracts to include `assessment` stage (additive v1) and store `assessment` artifacts.
- Task 2: Wire assessment save path to persist to existing stores (`evidence_selections`, `judgments` and/or span graph assertions) and mirror into the assessment artifact.

### Plan D: Verifier + Regression Tests
- Task 1: Add pytest coverage for orchestrator status transitions + artifact writes.
- Task 2: Add `scripts/dev/verify_10_02_happy_path.sh` (compose-backed) modeled after `verify_10_01_contracts.sh` + `scripts/dev/e2e_flow.py` helpers.

## Open Questions

1) **Exact meaning of “claimspan” vs existing claim ids**
   - What we know: repo already uses reviewer-scoped `cite:{doc}:{idx}:{reviewer}:{seg}` claim ids for attachments/evidence.
   - Risk: 10-02 context defines claimspan as “user-defined selection(s) within a citespan”, which could be a different identity.
   - Recommendation: treat claimspan id == existing reviewer-scoped claim id for the POC; if later decoupled, add a mapping table.

2) **Assessment label mapping to existing verdict schemas**
   - What we know: existing verdict enums are `support|contradict|uncertain` (judgment) and `support|contradict|uncertain|none` (evidence selection).
   - Recommendation: map `Inconsistent -> uncertain`, `Silent -> none`, and record the exact 4-label value in the assessment artifact for audit.

## Sources (Repo)

### Primary (HIGH confidence)
- `backend/main.py` (ingest/extract/attachments/evidence/pipeline routes)
- `backend/contracts/pipeline_v1.py` (immutable stage contracts; candidate/span ids)
- `backend/spine/pipeline_artifacts.py` + `backend/spine/pipeline_runs.py` (contract persistence)
- `backend/evidence_matching/service.py` (threaded orchestration + pause)
- `backend/attachment_store.py` + `backend/attachment_pipeline.py` (cited-PDF pipeline)
- `backend/span_graph_store.py` (citespans/claimspans/assertions primitives)
- `frontend/components/chasing_panel.py` (current reviewer segmentation + chase intent patterns)
- `scripts/dev/verify_10_01_contracts.sh`, `scripts/dev/e2e_flow.py` (compose-backed verification patterns)

## Metadata

**Confidence breakdown:**
- Background orchestration reuse: HIGH (patterns verified in `evidence_matching` and `attachment_pipeline`)
- Status schema recommendation: MEDIUM (new tables required; design grounded in repo patterns)
- SSE + Streamlit integration: MEDIUM (backend SSE is straightforward; Streamlit SSE consumption is not established here)

**Valid until:** 2026-03-21
