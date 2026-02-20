# Testing Patterns

**Analysis Date:** 2026-02-20

## Test Framework

**Runner:**
- pytest (not pinned in `requirements/app.in`; must be installed in the dev/test environment)
- Config: `pytest.ini` (scopes discovery to `tests/`)

**Assertion Library:**
- Built-in `assert` + `pytest.raises` (e.g. `tests/test_evidence_selection_store.py:16`, `tests/test_evidence_store.py:108`).

**Run Commands:**
```bash
pytest -q              # Run all tests
pytest -q tests/test_attachment_store.py   # Run a single file
pytest -q -k rerun     # Run by keyword expression
```

## Test File Organization

**Location:**
- Centralized under `tests/` (enforced by `pytest.ini`).

**Naming:**
- Files: `tests/test_*.py`
- Test functions: `test_*`

**Structure:**
```
tests/
  conftest.py
  test_attachment_*.py
  test_evidence_*.py
  test_span_graph_*.py
```

## Test Structure

**Suite Organization:**
```python
from pathlib import Path

from backend import attachment_pipeline, attachment_store


def _make_source(tmp_path: Path) -> Path:
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-sample")
    return source


def test_process_attachment_creates_artifacts(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    record = attachment_store.create_attachment(
        claim_id="claim-1",
        doc_id="doc-9",
        local_path=source,
        claim_text="Pipeline claim text",
    )
    monkeypatch.setattr(attachment_pipeline.grobid_client, "extract_tei", lambda _: "<TEI/>")
    ...
```

**Patterns:**
- Setup via `tmp_path` for filesystem artifacts and `monkeypatch` for dependency injection (e.g. `tests/test_attachment_pipeline.py:19`).
- FastAPI endpoints exercised using `fastapi.testclient.TestClient` against `backend/main.py` app with globals patched (e.g. `tests/test_span_graph_rebuild.py:9`).

## Mocking

**Framework:**
- pytest `monkeypatch` + `types.SimpleNamespace`

**Patterns:**
```python
def test_process_attachment_creates_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(attachment_pipeline.grobid_client, "extract_tei", lambda _: "<TEI/>")
    monkeypatch.setattr(attachment_pipeline.utils, "embed", lambda texts, **_: [[1.0] for _ in texts])
    ...
```

**What to Mock:**
- Expensive / external work: GROBID calls, embedding model calls, NLI scoring (e.g. `tests/test_attachment_pipeline.py:28`, `tests/test_evidence_matching_pipeline.py:190`).
- Global singletons in `backend/main.py` by patching module attributes before constructing `TestClient` (e.g. `tests/test_span_graph_rebuild.py:11`).

**What NOT to Mock:**
- Avoid stubbing `sys.modules` during import-time in tests; it leaks cross-test state (explicit note in `tests/test_evidence_matching_api.py:11`).

## Fixtures and Factories

**Test Data:**
- Use small helper factories that write minimal PDF-like bytes to `tmp_path` (e.g. `tests/test_attachment_store.py:7`, `tests/test_attachment_pipeline.py:13`).
- Use in-test constants for TEI stubs and sentence rows (e.g. `tests/test_attachment_pipeline.py:6`).

**Location:**
- Shared fixtures live in `tests/conftest.py`.

## Coverage

**Requirements:** None enforced/detected.

**View Coverage:**
```bash
Not detected (no coverage tool/config committed)
```

## Test Types

**Unit Tests:**
- Pure-ish logic and store behavior with stubbed APIs/UI shims (e.g. `tests/test_claim_queue.py`, `tests/test_evidence_store.py`).

**Integration Tests:**
- Tests rely on real Postgres DDL + truncation between tests via `tests/conftest.py` (`backend/db/migrate.py`, `backend/db/pg.py`).
- Tests rely on a reachable S3-compatible object store for attachment PDFs (e.g. `tests/test_attachment_store.py:26` uses `backend/object_store/s3.py`).

**Integration Test Requirements:**
- Postgres configured via `POSTGRES_DSN` (default in `backend/settings.py:74`).
- S3/MinIO configured via `S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET_WORKS` (defaults in `backend/settings.py:81`).

**E2E Tests:**
- Not detected (smoke workflows exist outside pytest via `scripts/dev/*` and `smoke.py`).

## Common Patterns

**Async Testing:**
- Not a primary pattern; concurrency is tested using threads/events and deterministic timeouts (e.g. `tests/test_evidence_matching_api.py:203`).

**Error Testing:**
```python
import pytest


with pytest.raises(SomeError):
    ...
```

---

*Testing analysis: 2026-02-20*
