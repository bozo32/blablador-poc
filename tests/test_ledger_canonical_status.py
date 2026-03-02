from __future__ import annotations

import backend.main as backend_main


def test_apply_canonical_ledger_status_from_spine() -> None:
    rows = [
        {
            "num": 1,
            "ingest_id": "doc-1",
            "status": "orange",
        }
    ]
    by_ingest = {
        "doc-1": {
            "id": "doc-1",
            "extraction": {"status": "complete", "error": None},
            "body_extraction": {"status": "running", "error": None},
            "resolution": {"status": "error", "error": "resolver failed"},
        }
    }

    backend_main._apply_canonical_ledger_status(rows, by_ingest)

    row = rows[0]
    assert row["canonical_extraction_status"] == "complete"
    assert row["canonical_body_extraction_status"] == "running"
    assert row["canonical_resolution_status"] == "error"
    assert row["canonical_resolution_error"] == "resolver failed"
    # Legacy fields remain populated for compatibility.
    assert row["extraction_status"] == "complete"
    assert row["resolution_status"] == "error"


def test_get_document_ledger_uses_canonical_status_builder(monkeypatch) -> None:
    monkeypatch.setattr(
        backend_main,
        "_require_project_id_for_upload",
        lambda *_args, **_kwargs: "proj-ledger",
    )
    monkeypatch.setattr(
        backend_main,
        "_build_ledger_response",
        lambda **kwargs: {"rows": [{"num": 1}], "options": [], "_meta": kwargs},
    )

    payload = backend_main.get_document_ledger(x_project_id="proj-ledger")
    assert payload["_meta"] == {"project_id": "proj-ledger", "reconcile": True}
