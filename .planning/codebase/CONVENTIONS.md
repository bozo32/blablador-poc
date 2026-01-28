# Coding Conventions

**Analysis Date:** 2026-01-23

## Naming Patterns

**Files:**
- Use snake_case for Python modules and scripts (e.g., `backend/retriever.py`, `backend/model_cache.py`, `frontend/ui.py`, `tests/test_retriever.py`).

**Functions:**
- Use snake_case for functions and methods (e.g., `backend/parser.py` uses `tei_to_chunks`, `backend/utils.py` uses `set_sane_threads`, `backend/retriever.py` uses `query_many`).

**Variables:**
- Use snake_case for locals/params (e.g., `backend/main.py` uses `pipeline_mode`, `backend/utils.py` uses `texts`), and ALL_CAPS for module constants (e.g., `backend/utils.py` uses `MODEL_CACHE_DIR`, `backend/main.py` uses `CSV_PATH`).

**Types:**
- Use PascalCase for classes (e.g., `backend/schemas.py` uses `RequestSettings`, `SegmentRequest`) and TitleCase aliases (e.g., `backend/model_cache.py` uses `ModelCategory`).

## Code Style

**Formatting:**
- Use 4-space indentation and standard PEP 8 spacing as in `backend/main.py` and `backend/parser.py`.
- Keep line-length reasonable with multi-line arguments aligned by parentheses (see `backend/utils.py` and `frontend/ui.py`).
- No formatter config detected in repo root (`pyproject.toml`, `setup.cfg`, `.prettierrc`).

**Linting:**
- Not detected (no `pyproject.toml`, `setup.cfg`, `.flake8`, or `ruff.toml` present in repo root).

## Import Organization

**Order:**
1. Standard library imports (e.g., `backend/parser.py` uses `logging`, `re`, `Path`, `typing`).
2. Third-party imports (e.g., `backend/parser.py` uses `lxml`, `backend/retriever.py` uses `faiss`, `numpy`).
3. Local application imports (e.g., `backend/parser.py` uses `from .utils import read_csv`).

**Path Aliases:**
- No import alias system detected; use absolute package imports like `from backend import utils` in `backend/main.py` and relative package imports like `from .utils import read_csv` in `backend/parser.py`.

## Error Handling

**Patterns:**
- API handlers raise `HTTPException` with status/detail on failures in `backend/main.py`.
- Utility functions raise concrete exceptions for failure states (e.g., `RuntimeError` in `backend/utils.py`, `ValueError` in `backend/retriever.py`).
- Use `try/except` with logging before returning fallbacks (e.g., `backend/nli.py` logs and returns empty evidence, `backend/utils.py` logs JSON parsing failures).

## Logging

**Framework:** `logging`

**Patterns:**
- Configure logging via `logging.basicConfig` and module-level loggers (e.g., `backend/main.py`, `backend/nli.py`, `backend/parser.py`).
- Use `logger.debug/info/warning/error` for trace-level pipeline steps (`backend/main.py`, `backend/nli.py`).
- CLI-style scripts use `print` for user feedback (e.g., `backend/hybrid.py`, `application.py`).

## Comments

**When to Comment:**
- Use module-level docstrings for context and constraints (e.g., `backend/parser.py`).
- Use inline step markers and section separators for long pipelines (e.g., `backend/hybrid.py`).

**JSDoc/TSDoc:**
- Not applicable; Python code uses docstrings in `backend/utils.py` and `backend/parser.py`.

## Function Design

**Size:**
- Prefer small helpers for single responsibilities (e.g., `_strip_citations`, `_clean` in `backend/parser.py`, `filter_and_snap` in `backend/utils.py`).

**Parameters:**
- Use type hints with `list[str]`, `Optional`, and `Literal` where possible (e.g., `backend/utils.py`, `backend/schemas.py`).

**Return Values:**
- Return dictionaries/lists for pipeline data structures and include metadata keys consistently (e.g., `backend/retriever.py`, `backend/parser.py`).

## Module Design

**Exports:**
- Define explicit public API in `backend/utils.py` via `__all__`.

**Barrel Files:**
- Not used; modules are imported directly by path (e.g., `backend/main.py` imports `backend.pipeline_registry`).

---

*Convention analysis: 2026-01-23*
