# Testing Patterns

**Analysis Date:** 2026-01-23

## Test Framework

**Runner:**
- pytest (imported in `tests/test_retriever.py`)
- Config: Not detected (no `pytest.ini`, `pyproject.toml`, or `setup.cfg`)

**Assertion Library:**
- pytest plain `assert` statements in `tests/test_retriever.py` and `tests/test_parser.py`.

**Run Commands:**
```bash
pytest                     # Run all tests
pytest -k <pattern>        # Targeted subset (no watch config detected)
Not detected               # Coverage command
```

## Test File Organization

**Location:**
- Tests live under `tests/` with mostly flat structure (`tests/test_retriever.py`, `tests/test_parser.py`).

**Naming:**
- Use `test_*.py` filenames for pytest discovery (`tests/test_retriever.py`, `tests/test_parser.py`).

**Structure:**
```
tests/
├── test_retriever.py
├── test_parser.py
├── test_blablador.py
├── test_coref.py
├── test_colbert.py
└── dummy data/
```

## Test Structure

**Suite Organization:**
```python
import pytest

@pytest.fixture
def dummy_chunks():
    return [{"text": "...", "meta": {"id": "0", "type": "sentence"}}]

def test_max_sentences_cap(tmp_path, monkeypatch, dummy_chunks):
    ...
```

**Patterns:**
- Use pytest fixtures for shared data (`tests/test_retriever.py`).
- Use `tmp_path` for filesystem isolation (`tests/test_retriever.py`).
- Use direct asserts with descriptive messages (`tests/test_parser.py`).

## Mocking

**Framework:** pytest monkeypatch (`tests/test_retriever.py`).

**Patterns:**
```python
def fake_embed(texts):
    return np.arange(len(texts), dtype="float32").reshape(-1, 1)

monkeypatch.setattr(utils, "embed", fake_embed)
```

**What to Mock:**
- Mock embedding/model calls to keep tests deterministic (`tests/test_retriever.py`).

**What NOT to Mock:**
- Leave file parsing and metadata structure intact for unit checks (`tests/test_parser.py`).

## Fixtures and Factories

**Test Data:**
```python
@pytest.fixture
def dummy_chunks():
    return [{"text": f"sentence {i}", "meta": {"id": str(i), "type": "sentence"}} for i in range(20)]
```

**Location:**
- Fixtures are defined inline in test modules (`tests/test_retriever.py`).
- Static fixtures/data files live under `tests/dummy data/` (e.g., `tests/dummy data/source.csv`).

## Coverage

**Requirements:** None enforced (no coverage config detected in repo root).

**View Coverage:**
```bash
Not detected
```

## Test Types

**Unit Tests:**
- Parser/retriever unit tests use pytest (`tests/test_parser.py`, `tests/test_retriever.py`).

**Integration Tests:**
- External service checks are implemented as runnable scripts in `tests/test_blablador.py` and `tests/test_colbert.py`.
- Model sanity checks are script-style in `tests/test_coref.py`.

**E2E Tests:**
- Not used (no browser or API-level E2E framework detected).

## Common Patterns

**Async Testing:**
- Not detected in `tests/test_retriever.py` or `tests/test_parser.py`.

**Error Testing:**
- No explicit error-case assertions found; tests focus on successful paths (`tests/test_retriever.py`, `tests/test_parser.py`).

---

*Testing analysis: 2026-01-23*
