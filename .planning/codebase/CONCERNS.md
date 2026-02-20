# Codebase Concerns

**Analysis Date:** 2026-02-20

## Tech Debt

**Monolithic app entrypoints (hard to change safely):**
- Issue: Very large, multi-responsibility modules (API routing + ingestion + graph + dev utilities + background orchestration) accumulate cross-cutting logic and implicit coupling.
- Files: `backend/main.py`, `frontend/ui.py`, `frontend/components/live_surfing_panel.py`
- Impact: High regression risk; difficult to reason about side effects; slow iteration; merge conflicts.
- Fix approach: Split by feature area into routers/modules (FastAPI routers, Streamlit components), keep IO boundaries explicit, and introduce thin orchestration layers.

**Stateful globals in the API process:**
- Issue: Process-wide mutable caches used as application state.
- Files: `backend/main.py` (global `retrievers`/`results`), `backend/citation_graph.py` (`OPENALEX_CACHE`, `CITED_BY_CACHE`), `backend/utils.py` (`_loaded_models`, `_ensure_index` cache)
- Impact: Memory growth over time; hard-to-reproduce bugs; unsafe with multiple workers; non-deterministic behavior after hot reload.
- Fix approach: Replace with bounded caches (size + TTL), move durable state to Postgres/S3, and make caches request-scoped or keyed by immutable config (folder hash, model, parameters).

**In-process background worker pool (non-resilient):**
- Issue: Extraction jobs run in daemon threads inside the API container/process.
- Files: `backend/spine/extraction_pool.py`, `backend/main.py`
- Impact: Jobs are lost on process restart; no horizontal scaling; difficult observability; concurrency hazards.
- Fix approach: Move job execution to a separate worker service/queue (or at least an explicit worker process) with durable job leasing and retries.

**Schema management via large idempotent DDL list:**
- Issue: Migrations implemented as a long list of DDL statements executed on startup/tests.
- Files: `backend/db/migrate.py`, `backend/main.py`, `tests/conftest.py`
- Impact: Slow startup; hard to evolve schema safely; limited rollback/visibility; drift risk.
- Fix approach: Introduce schema versioning and structured migrations (even if still idempotent), and minimize per-startup DDL execution.

**Hybrid pipeline is marked as scaffold but selectable in runtime:**
- Issue: Experimental/verbose implementation is selectable via `PIPELINE_MODE=hybrid`.
- Files: `backend/pipeline_registry.py`, `backend/hybrid.py`, `backend/settings.py`
- Impact: Production-like runs can hit incomplete logic, heavy imports, debug prints, and missing dependency failures.
- Fix approach: Gate behind an explicit feature flag, add dependency checks, and align hybrid behavior/contracts with the classic pipeline.

**Repo-local runtime artifact committed into source tree:**
- Issue: Model selection history persisted to a file within the code directory.
- Files: `backend/model_cache.py`, `backend/model_cache.json`
- Impact: Dirty worktrees and merge conflicts; unclear environment separation.
- Fix approach: Move runtime caches under `data/` or user config dir, and ensure they are gitignored.

## Known Bugs

**UI error handling references undefined `response` and catches `BaseException`:**
- Symptoms: On request failure, UI can throw `UnboundLocalError` or hide real failures; also swallows `KeyboardInterrupt`/`SystemExit`.
- Files: `frontend/ui.py`
- Trigger: Any exception before `response` is assigned in `requests.post(...)` blocks (e.g. backend down) or user interrupts.
- Workaround: None (restart UI); failures can appear as empty/incorrect results.

**HTTP requests without timeouts (hang risk):**
- Symptoms: UI or backend call can hang indefinitely, blocking request threads and/or Streamlit render.
- Files: `frontend/ui.py` (calls to backend without `timeout=`), `backend/utils.py` (`requests.post(f"{api_url}/search", ...)` without `timeout=`)
- Trigger: Network stall, backend deadlock, ColBERT service not responding.
- Workaround: Restart processes.

**ColBERT rerank assumes exact text match and can raise `StopIteration`:**
- Symptoms: Rerank crashes if ColBERT returns text that does not exactly match a window (normalization, duplicates, truncation).
- Files: `backend/utils.py` (`colbert_api_rerank`)
- Trigger: Duplicate window texts or service-side normalization.
- Workaround: Disable ColBERT rerank path.

**Potential contract mismatch in `filter_and_snap`:**
- Symptoms: Indexing errors or type errors if called with FAISS IDs instead of integer indices.
- Files: `backend/utils.py`
- Trigger: Passing string ids (as returned by `Retriever.search`) into `filter_and_snap`.
- Workaround: Avoid calling `filter_and_snap` unless inputs are verified integer indices.

**Launcher does not start the ColBERT server despite building command:**
- Symptoms: `application.py` mentions starting ColBERT but never executes `colbert_cmd`.
- Files: `application.py`
- Trigger: Running `python application.py` expecting ColBERT availability.
- Workaround: Start ColBERT server separately.

## Security Considerations

**No authentication/authorization on API endpoints:**
- Risk: Anyone who can reach the API can ingest documents, wipe data, and access project artifacts.
- Files: `backend/main.py`
- Current mitigation: None detected (no auth middleware / per-route checks).
- Recommendations: Add auth (at minimum bearer token), restrict privileged endpoints, and isolate dev-only routes.

**Dangerous dev wipe endpoint is publicly exposed:**
- Risk: Remote data loss (Postgres TRUNCATE + S3 delete-all) if API is reachable.
- Files: `backend/main.py` (`POST /dev/wipe`), `backend/object_store/s3.py` (`delete_all`)
- Current mitigation: Requires JSON body `{"confirm":"WIPE"}` only.
- Recommendations: Require admin auth, remove from production builds, or gate behind environment flag.

**Overly permissive CORS configuration:**
- Risk: Cross-site requests from arbitrary origins; also `allow_credentials=True` with `allow_origins=["*"]` is incompatible with browsers.
- Files: `backend/main.py`
- Current mitigation: None detected.
- Recommendations: Restrict `allow_origins` to trusted UI origins and set `allow_credentials` accordingly.

**Client-controlled filesystem paths used by server-side indexing:**
- Risk: If the API is exposed beyond localhost, a caller can point `folder` to arbitrary server directories and potentially exfiltrate content via downstream processing.
- Files: `backend/main.py` (`/segment`, `/prebuild`), `backend/retriever.py`, `backend/parser.py`
- Current mitigation: None detected.
- Recommendations: Never accept raw server paths from clients; require uploads into a controlled workspace and validate paths against that workspace.

**HTML injection surface in Streamlit UI:**
- Risk: `unsafe_allow_html=True` renders HTML built from backend/external data; if any untrusted content flows into these blocks, it can enable script injection in the browser.
- Files: `frontend/ui.py`, `frontend/components/evidence_card.py`, `frontend/components/chasing_panel.py`, `frontend/components/rationale_sidebar.py`
- Current mitigation: None detected.
- Recommendations: Avoid `unsafe_allow_html=True` for untrusted strings; sanitize/escape user and network-derived fields.

**XML parsing of untrusted documents (XXE / entity expansion):**
- Risk: TEI/XML parsing without hardened parser settings can expose entity expansion or external fetch risks.
- Files: `backend/parser.py` (lxml `etree.parse`), `backend/bl_client.py` (`xml.etree.ElementTree.parse`)
- Current mitigation: Not detected.
- Recommendations: Configure parsers with safe options (no network, no entity resolution) and treat uploads as untrusted.

**Insecure defaults for tokens/credentials in dev configuration:**
- Risk: Accidentally deploying with default secrets (`dev-internal-token`, `minio12345`) or wide-open local endpoints.
- Files: `backend/settings.py`, `docker-compose.yml`, `.env.example`
- Current mitigation: `.env` is gitignored via `.gitignore`.
- Recommendations: Require explicit secrets in production, remove default tokens, and use separate dev/prod config.

## Performance Bottlenecks

**Ingest reads entire PDF into memory:**
- Problem: `UploadFile.read()` buffers the full PDF; large uploads can spike memory.
- Files: `backend/main.py`
- Cause: Full in-memory read for hashing/storage.
- Improvement path: Stream upload to object store while hashing; enforce file size limits.

**CPU-bound embedding/NLI work runs inline on request thread/event loop:**
- Problem: Heavy ML inference blocks API responsiveness.
- Files: `backend/main.py` (`/segment`), `backend/utils.py`, `backend/nli.py`
- Cause: Local model inference and embedding done directly in request handlers.
- Improvement path: Offload to worker queue or dedicated service; add caching keyed by (model, inputs) where appropriate.

**Unbounded model caches can grow without limit:**
- Problem: Caching multiple HF models/pipelines increases RAM/VRAM usage.
- Files: `backend/nli.py` (`@lru_cache(maxsize=None)`), `backend/utils.py` (`_loaded_models`)
- Cause: Cache keys include model names; no eviction.
- Improvement path: Set bounded cache sizes + explicit eviction; track loaded model footprints.

**Repeated heavy imports/initialization in hot paths:**
- Problem: Some rerankers initialize expensive objects per call.
- Files: `backend/utils.py` (`sbert_rerank`, `bm25_rerank`)
- Cause: Model/tokenizer/NLP objects created on demand.
- Improvement path: Cache model instances and tokenizers; avoid per-request initialization.

**Naive in-memory caching for citation graph expansion:**
- Problem: Caches grow with unique identifiers and are never evicted.
- Files: `backend/citation_graph.py`
- Cause: Global dict caches without TTL/size cap.
- Improvement path: Add bounded TTL cache and/or store cache in external store.

## Fragile Areas

**Segment/prebuild cache invalidation and concurrency:**
- Files: `backend/main.py`
- Why fragile: Global `retrievers` is mutated (`clear()`/`update()`) without locking; concurrent users/requests can see partially rebuilt state or stale indexes.
- Safe modification: Introduce per-request retriever creation or lock + versioned cache keys (folder hash + embed model + parameters).
- Test coverage: Not detected for concurrent requests.

**Graph and span graph stores are large and tightly coupled to schema:**
- Files: `backend/graph_store.py`, `backend/span_graph_store.py`, `backend/db/migrate.py`
- Why fragile: Many SQL contracts and implicit assumptions; schema evolution requires coordinated changes.
- Safe modification: Add focused integration tests per store method and keep schema changes localized with migrations.
- Test coverage: Partial (tests exist, but coverage of edge cases and migrations not guaranteed).

**Hybrid pipeline behavior depends on optional, environment-specific dependencies:**
- Files: `backend/hybrid.py`, `backend/utils.py`, `colbert_server/colbert.py`
- Why fragile: `faiss`, `torch`, `sentence-transformers`, and external ColBERT service vary across environments.
- Safe modification: Make dependency availability explicit and add graceful fallback paths.
- Test coverage: Not detected for hybrid path end-to-end.

## Scaling Limits

**Single-process state and background threads prevent horizontal scaling:**
- Current capacity: One API instance with in-process caches and threads.
- Limit: Multiple replicas do not share caches; extraction thread pool does not coordinate; memory grows per replica.
- Scaling path: Externalize background work (queue + worker), move caches out of process, add stateless API layer.

## Dependencies at Risk

**Unpinned "latest" container images in Compose:**
- Risk: Breaking changes or supply-chain issues.
- Impact: Non-reproducible dev/prod environments.
- Migration plan: Pin images to specific versions/digests.
- Files: `docker-compose.yml`

**ML stack version churn and device semantics:**
- Risk: Transformers/Torch API changes (eg. device handling, caching env vars) can break runtime.
- Impact: Runtime failures and hard-to-debug performance regressions.
- Migration plan: Keep lockfiles current, add smoke tests for model loading/inference.
- Files: `requirements/app.lock.txt`, `backend/nli.py`, `backend/main.py`

## Missing Critical Features

**Production-grade auth and multi-tenant isolation:**
- Problem: Project/user scoping exists in settings but enforcement and authentication are not implemented for API requests.
- Blocks: Secure multi-user deployments.
- Files: `backend/settings.py`, `backend/main.py`

## Test Coverage Gaps

**Tests depend on live Postgres and do not isolate object store:**
- What's not tested: Clean-room runs without external services; object store cleanup and lifecycle.
- Files: `tests/conftest.py`, `backend/db/pg.py`, `backend/object_store/s3.py`
- Risk: CI instability; local tests pass with leftover S3 objects; environment-specific failures.
- Priority: High

**No automated coverage for the Streamlit UI and browser-level flows:**
- What's not tested: UI rendering/interaction, XSS surfaces, and full-stack flows.
- Files: `frontend/ui.py`, `frontend/components/*`
- Risk: Regressions ship unnoticed; security issues not caught.
- Priority: Medium

---

*Concerns audit: 2026-02-20*
