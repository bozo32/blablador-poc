# Codebase Concerns

**Analysis Date:** 2026-01-23

## Tech Debt

**Hybrid ColBERT internal mode is a stub:**
- Issue: `mode == "internal"` uses `...` and `reranked_internal` placeholder, so selecting internal mode will crash.
- Files: `backend/hybrid.py`
- Impact: Hybrid pipeline cannot run with internal ColBERT, blocking expected reranking path.
- Fix approach: Implement internal ColBERT rerank path or remove the mode and validate `COLBERT_MODE` values.

**Parser caption support is parked in comments:**
- Issue: Figure/table caption handling is commented out and marked FIXME, with no implementation path.
- Files: `backend/parser.py`
- Impact: Caption evidence is never indexed, so claims referencing figures/tables miss relevant context.
- Fix approach: Implement caption extraction with stable `p_id` and integrate into window stream.

**Frontend UI is a monolith:**
- Issue: Single 700+ line Streamlit module mixes API calls, state, rendering, and business logic.
- Files: `frontend/ui.py`
- Impact: High change risk and makes isolated testing/refactoring difficult.
- Fix approach: Split into modules (api, state, rendering, helpers) and add unit tests.

**Global retriever cache has no lifecycle management:**
- Issue: `retrievers` dict grows without eviction or persistence strategy.
- Files: `backend/main.py`
- Impact: Memory usage grows with each segment/row; long-lived server can degrade or OOM.
- Fix approach: Add LRU eviction, explicit teardown, or on-disk cache with size limits.

## Known Bugs

**Unhandled exception path on request failure:**
- Symptoms: If `requests.post` raises before `response` is assigned, the exception handler references an undefined variable.
- Files: `frontend/ui.py`
- Trigger: Network errors or connection refusal during segment POST.
- Workaround: Not available; current handler can raise `UnboundLocalError`.

**Incorrect NLI helper signature usage:**
- Symptoms: `predict_nli` calls `get_nli_pipeline()` without required `model_name`.
- Files: `backend/nli.py`
- Trigger: Any call to `predict_nli`.
- Workaround: Call `get_nli_pipeline(model_name)` directly.

**Broken tests in repository:**
- Symptoms: `tests/test_segment_endpoint.py` references `first_ev` without definition, and `tests/test_parser.py` asserts `meta['type'] == 'sentence'` although parser produces `sentence_window`.
- Files: `tests/test_segment_endpoint.py`, `tests/test_parser.py`, `backend/parser.py`
- Trigger: Running pytest.
- Workaround: None; tests must be updated to current behavior.

## Security Considerations

**Backend CORS is fully open:**
- Risk: Any origin can issue browser requests to the API, enabling accidental exposure in shared networks.
- Files: `backend/main.py`
- Current mitigation: None beyond FastAPI defaults.
- Recommendations: Restrict `allow_origins` to known frontend URLs or make it configurable.

**API key is entered and displayed in plain text:**
- Risk: Users can leak keys via screen sharing or logs; keys live in session state.
- Files: `frontend/ui.py`
- Current mitigation: None.
- Recommendations: Use `type="password"` for the Streamlit input and avoid logging keys.

## Performance Bottlenecks

**Hybrid pipeline builds full in-memory FAISS per request:**
- Problem: `build_all` embeds all windows and performs FAISS search from scratch.
- Files: `backend/hybrid.py`
- Cause: No reuse of cached embeddings/indices across requests.
- Improvement path: Cache FAISS indices per document or prebuild them once via a background job.

**Model availability check is N+1 network calls:**
- Problem: UI pings the completions endpoint for every model during sidebar render.
- Files: `frontend/ui.py`
- Cause: `get_responsive_models` loops sequentially over model IDs.
- Improvement path: Add caching in session state and batch validation where possible.

**Large session state payloads scale with dataset size:**
- Problem: Entire CSV results, segments, and evidence are stored in Streamlit session state.
- Files: `frontend/ui.py`
- Cause: In-memory per-row state accumulation.
- Improvement path: Store results on disk or page through data instead of keeping all rows in memory.

## Fragile Areas

**Paragraph IDs are derived from incrementing counters:**
- Files: `backend/parser.py`
- Why fragile: `p_id` uses a running counter fallback, so ID stability depends on parsing order; captions/tables would disrupt alignment.
- Safe modification: Introduce stable paragraph IDs and keep mapping logic centralized.
- Test coverage: Missing stable-ID tests (no coverage in `tests/test_parser.py`).

**Shared global retriever state is not concurrency-safe:**
- Files: `backend/main.py`
- Why fragile: Concurrent requests mutate shared dicts without locks, risking inconsistent cache state.
- Safe modification: Guard updates with locks or move cache to an external store.
- Test coverage: No concurrency tests (no coverage in `tests/`).

## Scaling Limits

**In-memory FAISS indexes are unbounded:**
- Current capacity: Limited by process RAM; every segment can add entries to `retrievers`.
- Files: `backend/main.py`, `backend/retriever.py`
- Limit: Large datasets or many users can exhaust memory and slow queries.
- Scaling path: External index service or persistent cache with eviction and size caps.

**Streamlit state scales linearly with rows:**
- Current capacity: Entire dataset and results live in RAM per user session.
- Files: `frontend/ui.py`
- Limit: Large CSVs can slow UI and exhaust memory.
- Scaling path: Pagination plus storing results in files or a database.

## Dependencies at Risk

**Optional dependencies are assumed present:**
- Risk: `fastcoref` and `colbert` imports are executed in runtime paths without guard rails when modes are enabled.
- Impact: Missing packages crash the hybrid path or tests.
- Migration plan: Add optional dependency checks and clearer error messages.
- Files: `backend/hybrid.py`, `tests/test_coref.py`, `tests/test_colbert.py`

## Missing Critical Features

**Not detected:**
- No explicit missing feature markers found in `backend/*.py`, `frontend/*.py`.

## Test Coverage Gaps

**Hybrid pipeline is largely untested:**
- What's not tested: Core hybrid build flow, reranking, and coref integration.
- Files: `backend/hybrid.py`
- Risk: Regressions in hybrid mode go undetected.
- Priority: High.

**Frontend has no automated tests:**
- What's not tested: UI flow, API payloads, session state updates.
- Files: `frontend/ui.py`
- Risk: UI regressions and integration failures surface late.
- Priority: Medium.

**Scripts under tests/ are not real tests:**
- What's not tested: ColBERT and coref scripts are not pytest cases and provide no assertions.
- Files: `tests/test_colbert.py`, `tests/test_coref.py`, `tests/test_blablador.py`
- Risk: CI passes without covering key integrations.
- Priority: Medium.

---

*Concerns audit: 2026-01-23*
