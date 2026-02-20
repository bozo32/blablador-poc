# Phase 10-01: Contracts Arc (Stage Boundaries) - Research

**Researched:** 2026-02-20
**Domain:** Pipeline stage contracts + spine-backed artifacts (Postgres + S3/MinIO)
**Confidence:** HIGH

## Summary

This phase can build directly on existing spine patterns: JSON artifacts stored in the works bucket via `backend/object_store/s3.py`, with Postgres pointers in dedicated tables (e.g. `evidence_runs.candidates_object_key`) or via the generic ingestion `artifacts` table (`backend/spine/artifacts.py`). The existing extraction output (`extraction.json`) is already a stable JSON payload produced by `backend/extraction.py` and persisted by `backend/spine/extraction_pool.py`, and the citation-window span anchoring/IDs you need for citespans are already implemented (quote selectors + deterministic `span_id`) in `backend/text_selectors.py` and `backend/span_graph_store.py`.

For contracts, adopt a single envelope shape across all stages (`extract`, `citespans`, `retrieval`, `filter`, `rerank`, `nli`) with: `schema_version` (int), `artifact_type` including `@vN`, `run_id` (UUID string), stage name, `status` (`complete|partial|error`), `warnings[]`, and an optional structured `error` object. Store each stage as one JSON object at `pipeline/{run_id}/{stage}.json` in the `works` bucket (append-only by `run_id`), and persist a Postgres pointer row per stage so the API can return JSON bodies without clients touching S3.

**Primary recommendation:** Reuse the existing selector + `span_id` implementation from `backend/span_graph_store.py` for citespans, and follow the existing “JSON-in-S3, pointer-in-Postgres” pattern from `backend/evidence_matching/store.py` for stage artifact persistence.

## Standard Stack

### Core
| Library/Tool | Version (repo) | Purpose | Why Standard (in this repo) |
|---|---:|---|---|
| Python | 3.10 | Backend runtime | Matches `environment.yml` + Dockerfiles |
| FastAPI | 0.129.0 | HTTP API | Existing routing + TestClient patterns (`backend/main.py`) |
| Pydantic | 2.12.5 | Request/response + contract models | Already used (`backend/schemas.py`, `backend/settings.py`) |
| psycopg[binary] | 3.3.2 | Postgres spine persistence | Existing `backend/db/pg.py`, `backend/db/migrate.py` |
| boto3 | 1.42.49 | S3/MinIO object store | Existing wrapper `backend/object_store/s3.py` |

### Supporting
| Library/Tool | Version (repo) | Purpose | When to Use |
|---|---:|---|---|
| pytest | (installed via lock) | Tests | Add contract store + API wiring tests under `tests/` |
| lxml | 6.0.2 | TEI parsing | Extraction already depends on it |

## Architecture Patterns

### Existing Patterns to Build On (By Goal)

**Extraction output is shaped here (start of contract arc):**
- `backend/extraction.py`: `parse_tei()` returns extraction payload `{metadata,citations,references,extraction_version}`.
- `backend/spine/extraction_pool.py`: persists extraction payload to S3 key `extract/{work_id}/attempts/{attempt_id}/primary/extraction.json` and records Postgres artifact pointer type `extraction.json`.
- `backend/spine/ingest_view.py`: loads `extraction.json` from S3 (via pointer lookup) into API-visible “ingested document” view.

**Citation-window anchoring + deterministic span_id (citespans stage foundation):**
- `backend/text_selectors.py`: builds quote selector `{exact,prefix,suffix}` and a stable `fingerprint()`.
- `backend/span_graph_store.py`:
  - `_span_id(...)` defines deterministic `span_id` as `span:sha256(anchor_id|kind|exact|prefix|suffix|window_fingerprint)`.
  - `upsert_span(kind='citation_window', ...)` stores span rows with `selector_json` + `window_fingerprint`.
  - `find_citation_span(...)` + `span_graph_citation_span_index` map `(ingest_id,citation_index,target_id)` to a `span_id`.

**Object store + Postgres pointer pattern (stage artifacts):**
- `backend/object_store/s3.py`: `put_bytes()`, `get_bytes()`, `exists()`, and `delete_all()`.
- `backend/evidence_matching/store.py`: writes JSON bytes to S3 (`evidence/{claim_id}/{run_id}.json`) and stores the `candidates_object_key` pointer in Postgres (`evidence_runs`). This is the closest existing pattern to “API returns JSON bodies; clients don’t fetch S3 directly.”
- `backend/spine/artifacts.py`: generic artifacts pointer table for ingestion attempts (`artifacts`), used by extraction/resolution and fetched via `get_artifact_for_attempt()`.

**/dev/wipe implementation (must delete pipeline artifacts too):**
- `backend/main.py` `POST /dev/wipe` truncates many Postgres tables and calls `backend/object_store/s3.delete_all()`.
  - Note: today this deletes *all* keys in the works bucket, so pipeline artifacts under `pipeline/` are already included. If Phase 10 introduces additional buckets/prefix scoping, keep this behavior aligned.

### Recommended New Code Locations (Repo-Consistent)

Prescriptive paths consistent with existing clusters and naming conventions (`*_store.py`, feature modules in `backend/<cluster>/`):

1) Contract models + upgrade shims
- `backend/contracts/__init__.py`
- `backend/contracts/pipeline_v1.py` (Pydantic v2 `BaseModel` stage payloads)
- `backend/contracts/upgrade.py` (version upgrade shims; additive-only evolution)

2) Spine-backed persistence for pipeline stage artifacts
- `backend/spine/pipeline_runs.py` (create/read run metadata; compute `input_fingerprint`)
- `backend/spine/pipeline_artifacts.py` (record per-stage pointer rows + fetch stage JSON)
  - Must use `backend/object_store/s3.py` for bytes I/O.

3) Service/orchestration (thin endpoints)
- `backend/pipeline_contracts/service.py` (run stages, enforce caps, persist artifacts)

4) API routes
- Keep handlers thin in `backend/main.py` and delegate to `backend/pipeline_contracts/service.py`.

### Pattern: “S3 JSON + Postgres pointer, API returns JSON”

Use the same mechanics as `backend/evidence_matching/store.py`:

```python
# Source: backend/evidence_matching/store.py
candidates_key = f"evidence/{claim_id}/{run_id}.json"
candidates_bytes = json.dumps(
    annotated,
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
object_store_s3.put_bytes(candidates_key, candidates_bytes, content_type="application/json")
```

Apply the same to contract artifacts using the locked key layout:
`pipeline/{run_id}/{stage}.json`.

## Artifact Types + Contract Shapes

### Artifact Type Naming Scheme (Locked + Repo-Compatible)

**Use:** `contracts/{stage}@v{schema_version}`

Examples (schema_version = 1):
- `contracts/extract@v1`
- `contracts/citespans@v1`
- `contracts/retrieval@v1`
- `contracts/filter@v1`
- `contracts/rerank@v1`
- `contracts/nli@v1`

Rationale:
- Matches locked decision (“hierarchical `contracts/{stage}@vN`”).
- Keeps `artifact_type` self-describing even when object keys are uniform.

### Common Envelope (All Stages)

Implement as Pydantic v2 models (see existing usage in `backend/schemas.py`).

Required fields (additive-only):
- `schema_version: int` (>= 1)
- `artifact_type: str` (must include `@v{schema_version}`)
- `run_id: str` (UUID)
- `stage: Literal['extract','citespans','retrieval','filter','rerank','nli']`
- `work_id: str` (canonical; accept `doc_id` alias)
- `created_at: str` (RFC3339-ish `...Z` is used elsewhere)
- `status: Literal['complete','partial','error']`
- `warnings: list[str]` (or list of structured warnings; keep minimal initially)
- `error: {code,message,detail,retryable} | None`
- `component: {id, version, impl, config_hash} | None` (make swaps visible; locked discretion)
- `input_fingerprint: str` (append-only identity across reruns)
- `caps: {candidates:int, rerank:int, nli:int}` (defaults 200/50/12; allow override)

### Stage Payload: Minimal “Data” Blocks

These shapes are designed to be small, explicit, and compatible with existing primitives.

1) `extract` (start of arc)
- Purpose: wrap current extraction output (`extraction.json`) + resolved citation anchors.
- Data:
  - `structured_doc`: the extraction payload (currently from `backend/extraction.py`).
  - `citation_anchors[]`: items containing at least:
    - `citation_index: int` (position in `structured_doc.citations`)
    - `reference_id: str | None` (current Grobid `target_id` / bibr id)
    - `target_id: str | None` (OpenAlex work id when available; fallback policy is an open item)
    - `callout: str | None`
    - `window_text: str | None` (citation sentence/window used to anchor)
    - `sentence_id: str | None`
  - `extraction_attempt_id: str | None` (provenance pointer back to spine attempt)

2) `citespans`
- Purpose: generate deterministic `span_id` + selectors per citation anchor.
- Data grouped by `target_id` (locked):
  - `by_target: { target_id: { anchors[] } }`
  - `anchors[]` entry minimal:
    - `citation_index: int`
    - `span_id: str` (use `span_graph_store.upsert_span(kind='citation_window', ...)`)
    - `selectors: {quote:{exact,prefix,suffix}}`
    - `window_fingerprint: str`
    - `window_text: str` (optional but useful for debugging)

3) `retrieval`
- Purpose: produce up to `caps.candidates` candidates per anchor.
- Data grouped by `target_id`:
  - `by_target: { target_id: { anchors[] } }`
  - `anchors[]` entry:
    - `citation_index: int`, `span_id: str`
    - `query: {claim_text: str}` (if applicable) OR `{window_text: str}`; keep explicit
    - `candidates[]` each with:
      - `candidate_id: str` (stable per run; deterministic hash of `{run_id,target_id,span_id,paragraph_id}` or equivalent)
      - `attachment_id: str | None` (when candidate originates from a cited attachment)
      - `source`: `{work_id, attachment_id, paragraph_id, section_path, page}` (fields when available)
      - `text: str` (full paragraph text; locked)
      - `scores: {bm25?, faiss?, sbert?, colbert?, combined?, position}` (normalized floats)

4) `filter`
- Purpose: deterministic trimming/dedupe; separate from rerank (locked).
- Data:
  - Same candidate list shape as retrieval, but with:
    - `filter_reason: str | None` and/or `dropped: bool` for auditability.
    - Ensure output candidates are capped to `caps.candidates`.

5) `rerank` (optional)
- Purpose: rerank top-N from filter stage; cap default 50.
- Data:
  - Candidates (<= `caps.rerank`) with:
    - `scores.rerank` (or `cross_encoder` / `colbert`) and updated `combined`.
    - `component` metadata capturing which reranker implementation ran.

6) `nli` (optional)
- Purpose: label top candidates; cap default 12.
- Data:
  - Candidates (<= `caps.nli`) with:
    - `label: 'entails'|'contradicts'|'neutral'`
    - `scores.nli: float`
    - `highlights[]`: `{text,page,section_path,paragraph_id}` (assessment-ready snippet fields)
    - `token_saliencies: list[float] | None` (if available; mirrors existing evidence pipeline)

## Don’t Hand-Roll

| Problem | Don’t Build | Use Instead | Why |
|---|---|---|---|
| S3/MinIO client plumbing | direct boto3 usage scattered in new code | `backend/object_store/s3.py` | Centralizes MinIO path-style config + bucket checks |
| Deterministic `span_id` | custom hashing scheme per stage | `backend/span_graph_store.py` (`upsert_span`, `_span_id`) | Already implements selector-based deterministic IDs |
| Quote selector normalization | ad-hoc string slicing | `backend/text_selectors.py` | Normalization + fingerprinting already defined |
| JSON serialization determinism | default `json.dumps()` | follow `ensure_ascii=True, sort_keys=True, separators=(",",":")` | Existing artifacts rely on deterministic bytes for repeatable runs |
| Contract validation | hand-written dict checks | Pydantic v2 `BaseModel` | Repo already uses validators; produces good error messages |

## Common Pitfalls

### Pitfall 1: Breaking append-only semantics by “deduping” runs
**What goes wrong:** storing stage artifacts under a stable key (or reusing an existing run row) silently overwrites history.
**Why it happens:** existing ingestion attempts are idempotent by `(work_id, kind, settings_hash)` (`backend/spine/attempts.py:create_or_get_attempt`).
**How to avoid:** create a fresh `run_id` UUID per execution and always write to `pipeline/{run_id}/{stage}.json`; use `input_fingerprint` to group/compare runs, not to reuse keys.

### Pitfall 2: Span IDs drift if selector/window_fingerprint is not stable
**What goes wrong:** `span_id` changes across reruns, breaking joins to cited candidates (`attachment_id + span_id`) and making results hard to diff.
**Why it happens:** selector fields are whitespace-sensitive; missing/unstable `window_fingerprint` changes `_span_id`.
**How to avoid:** always compute selector via `text_selectors.build_anchor_quote(window_text)` and fingerprint via `text_selectors.fingerprint(window_text)`; store both in the citespans contract.

### Pitfall 3: Tests leak S3 keys between runs
**What goes wrong:** tests pass locally but fail intermittently when old objects are present.
**Why it happens:** `tests/conftest.py` truncates Postgres but intentionally does not delete S3 objects.
**How to avoid:** use UUID-scoped `run_id` (locked) so keys are unique; in tests assert existence via `object_store_s3.exists()` on the specific key.

### Pitfall 4: Clients accidentally depend on S3
**What goes wrong:** UI/clients start consuming S3 object keys directly, bypassing API.
**Why it happens:** storing only `object_key` in responses is tempting.
**How to avoid:** API endpoints must fetch from S3 server-side and return JSON bodies; if exposing pointers, keep them behind admin/debug-only endpoints.

## Code Examples

### Deterministic span_id generation (quote selectors + fingerprint)
```python
# Source: backend/span_graph_store.py
def _span_id(*, anchor_id: str, kind: str, selector: dict, window_fingerprint: str | None) -> str:
    raw = "|".join([anchor_id, kind, exact, prefix, suffix, fp])
    return f"span:{sha256(raw)}"
```

### Evidence-style S3 JSON artifact write (use same JSON settings)
```python
# Source: backend/evidence_matching/store.py
payload_bytes = json.dumps(
    payload,
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
object_store_s3.put_bytes(object_key, payload_bytes, content_type="application/json")
```

## Minimal Verification Steps (Tests + Scripts)

### Pytest (recommended minimum)

1) **Artifact persistence unit test** (new test file: `tests/test_pipeline_contract_store.py`)
- Create a `run_id` UUID and a small stage payload (e.g. `extract@v1`).
- Persist stage artifact; assert:
  - S3 key exists: `pipeline/{run_id}/extract.json` using `backend/object_store/s3.exists()`.
  - Postgres pointer row exists (stage -> object_key + artifact_type).

2) **API fetch test** (new test file: `tests/test_pipeline_contract_api.py`)
- Persist a stage artifact.
- Call new endpoint `GET /pipeline/runs/{run_id}/{stage}` (or equivalent).
- Assert JSON response includes:
  - `schema_version`, `artifact_type`, `run_id`, `stage`, `status`.
  - `work_id` present (and `doc_id` alias accepted if used).

3) **/dev/wipe includes pipeline artifacts**
- Write a pipeline object key.
- Call `POST /dev/wipe`.
- Assert key no longer exists in S3 (or bucket is empty if `delete_all()` remains).

### CLI verifier (optional but aligned with repo practice)

Add a small script under `scripts/dev/verify_10_01_contracts.sh` that:
- Calls `POST /dev/wipe`.
- Runs a minimal contract-producing flow.
- Fetches each stage via API and prints `artifact_type`, `schema_version`, `status`.

## State of the Art (Repo-Relevant)

| Older Pattern | Current Pattern | Where in Repo | Impact |
|---|---|---|---|
| durable blobs on local disk | Postgres pointers + S3/MinIO blobs | `backend/object_store/s3.py`, `backend/db/migrate.py` | Enables repeatable verifiers + shared state |
| ad-hoc span identifiers | selector-based deterministic ids | `backend/span_graph_store.py`, `backend/text_selectors.py` | Stable joins across re-extraction |

## Open Questions

1) **How exactly to derive `target_id` when OpenAlex is missing**
- What we know: current flows sometimes use `target_id` as Grobid bibl id; span graph falls back to `ref:{doc_id}:{target_id}`.
- What’s unclear: contract-level policy for `target_id` fallback in Phase 10-01.
- Recommendation: standardize in `extract@v1` to emit both `reference_id` and `target_id`, with `target_id` defaulting to `openalex_id` when available else `ref:{work_id}:{reference_id}`.

2) **Where to store Postgres pointers for pipeline stages**
- What we know: evidence uses a dedicated table (`evidence_runs`) + S3 pointer; ingestion uses generic `artifacts` table keyed by `attempt_id`.
- What’s unclear: whether Phase 10 should reuse `artifacts` (by creating a run record that can be referenced) or introduce a dedicated pipeline table.
- Recommendation: follow the evidence pattern (dedicated table(s) per domain) to avoid overloading ingestion attempt semantics and to keep queries stage-centric.

## Sources

### Primary (HIGH confidence; repo sources)
- `backend/extraction.py` (extraction payload shape)
- `backend/spine/extraction_pool.py` (persist extraction artifacts)
- `backend/object_store/s3.py` (S3/MinIO wrapper)
- `backend/spine/artifacts.py` (generic artifact pointers)
- `backend/evidence_matching/store.py` (S3 JSON + Postgres pointer pattern)
- `backend/text_selectors.py` (quote selectors + fingerprint)
- `backend/span_graph_store.py` (deterministic `span_id` + citation span index)
- `backend/main.py` (`POST /dev/wipe` implementation)
- `tests/conftest.py` (Postgres truncation; S3 not cleared in tests)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH (versions pinned in `requirements/app.lock.txt` and documented in `.planning/codebase/STACK.md`).
- Architecture: HIGH (patterns verified in repo modules listed in Sources).
- Pitfalls: HIGH (derived from repo behavior: idempotency in attempts + test S3 behavior).

**Research date:** 2026-02-20
**Valid until:** 2026-03-21
