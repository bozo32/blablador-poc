# Coding Conventions

**Analysis Date:** 2026-02-20

## Naming Patterns

**Files:**
- Python modules use `snake_case.py` in `backend/` and `frontend/` (e.g. `backend/attachment_store.py`, `frontend/project_api.py`).
- Test files use `tests/test_*.py` (e.g. `tests/test_attachment_pipeline.py`).

**Functions:**
- Use `snake_case` for functions and methods (e.g. `backend/object_store/s3.py:67` `put_bytes`, `backend/evidence_matching/service.py:79` `ensure_current_run`).

**Variables:**
- Module constants use `UPPER_SNAKE_CASE` (e.g. `frontend/project_api.py:9` `DEFAULT_TIMEOUT`, `backend/utils.py:17` `HF_INFERENCE_API_BASE`).
- Local variables use `snake_case`.

**Types:**
- Classes use `PascalCase` (e.g. `backend/settings.py:14` `AppSettings`, `backend/evidence_matching/service.py:28` `EvidenceMatchingService`).
- Enums use `PascalCase` with `UPPER_SNAKE_CASE` members (e.g. `backend/evidence_matching/types.py:10` `EvidenceLabel.ENTAILS`).

## Code Style

**Formatting:**
- Tool: Black
- Config: `.pre-commit-config.yaml` (rev `23.9.1`, `language_version: python3.11`)

**Linting:**
- Tool: Flake8
- Config: `.flake8` (`max-line-length = 88`)
- Pre-commit: `.pre-commit-config.yaml` (rev `6.1.0` + `flake8-docstrings`)
- Key ignores: `.flake8` `extend-ignore = E203, W503, D100, D101, D102, D103, E402`
  - `E402` being ignored enables deliberate "set env vars, then import" patterns in `backend/main.py`.

## Import Organization

**Order:**
1. Standard library
2. Third-party
3. Local application (`backend.*`, `frontend.*`)

**Pattern examples:**
- `backend/evidence_matching/service.py` keeps stdlib imports first, then internal imports, then local package imports.
- `backend/main.py` intentionally imports `backend/utils.py` early to set env/threading before heavy ML deps, then continues with normal imports (`.flake8` ignores `E402`).

**Path Aliases:**
- Not applicable (Python package imports use `backend.*` / `frontend.*` directly).

## Error Handling

**Patterns:**
- FastAPI endpoints raise `fastapi.HTTPException` with explicit `status_code` and `detail` (e.g. `backend/main.py:253`, `backend/main.py:315`).
- Validation errors are surfaced via `pydantic.ValidationError` in API handlers and tests (e.g. `backend/main.py:57`, `tests/test_judgment_store.py:26`).
- Internal helpers often raise `RuntimeError` for "caller must handle / fallback" semantics (e.g. `backend/utils.py:20` `hf_inference_post`).
- Broad exception catches are allowed when explicitly documented and/or marked with `# noqa: BLE001` (e.g. `backend/utils.py:41`).

## Logging

**Framework:** `logging` (stdlib)

**Patterns:**
- Module-level logger: `logger = logging.getLogger(__name__)` (e.g. `backend/attachment_pipeline.py:34`, `backend/evidence_matching/service.py:25`).
- Central `basicConfig` in API entrypoint with dependency log-level suppression (e.g. `backend/main.py:138`).
- Prefer structured-ish messages with parameterized args (`logger.warning("... %s", value)`) in long-running/retry loops (e.g. `backend/main.py:186`).

## Comments

**When to Comment:**
- Use short comments for non-obvious runtime constraints and ordering (e.g. `backend/main.py:17` threading env vars before importing numpy/torch).
- Use phase/intent notes where behavior is driven by repo roadmap constraints (e.g. `tests/conftest.py:10`).

**JSDoc/TSDoc:**
- Not applicable.

**Docstrings:**
- Docstrings are used for modules/classes and key public methods (e.g. `backend/evidence_matching/service.py:1`, `backend/evidence_matching/service.py:28`).
- Flake8-docstrings is enabled but missing docstrings are largely not enforced (`.flake8` ignores `D100..D103`); imperative mood rules still apply for docstrings that exist.

## Function Design

**Size:**
- Keep orchestration in service classes and route handlers; keep data shaping/normalization in small helpers (e.g. `_normalize_label` nested helper in `backend/evidence_matching/service.py:137`).

**Parameters:**
- Use keyword-only parameters for complex call signatures (e.g. `backend/evidence_matching/service.py:82` `ensure_current_run(..., *, claim_text=..., force=..., execute=...)`).

**Return Values:**
- Backend service methods frequently return dict payloads intended for API consumption (e.g. `backend/evidence_matching/service.py:87`).

## Module Design

**Exports:**
- Some modules define explicit public exports via `__all__` (e.g. `backend/evidence_matching/types.py:249`).

**Barrel Files:**
- Not applicable.

## UX Conventions

- For workspace UX changes, use `.planning/codebase/UX_STYLE_CONVENTIONS.md` as the source of truth.
- If a UX decision increases coupling to Streamlit rerun/widget semantics, record it in:
  - `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-PRUNE-CANDIDATES.md`

---

*Convention analysis: 2026-02-20*
