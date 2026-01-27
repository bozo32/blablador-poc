from pathlib import Path

import pytest

from backend import attachment_store
from backend.settings import settings


@pytest.fixture(autouse=True)
def attachment_dir(tmp_path):
    settings.ATTACHMENT_DIR = tmp_path / "attachments"
    yield


def _make_source(tmp_path: Path, name: str = "sample.pdf") -> Path:
    source = tmp_path / name
    source.write_bytes(b"%PDF-1.4\nSample attachment")
    return source


def test_create_attachment_persists_metadata(tmp_path):
    source = _make_source(tmp_path)
    record = attachment_store.create_attachment(
        claim_id="claim-1",
        doc_id="doc-42",
        local_path=source,
        filename="evidence.pdf",
        size_bytes=42,
        reference_hint={"doi": "10.1000/demo"},
    )

    assert record["status"] == attachment_store.STATUS_PENDING
    assert Path(record["file_path"]).exists()

    public = attachment_store.public_status(record["id"])
    assert public is not None
    assert public["filename"] == "evidence.pdf"
    assert public["history"][0]["event"] == "queued"
    assert public["reference_hint"]["doi"] == "10.1000/demo"


def test_update_trimmed_history(tmp_path):
    source = _make_source(tmp_path)
    record = attachment_store.create_attachment(
        claim_id="claim-2",
        doc_id=None,
        local_path=source,
        filename="trial.pdf",
    )

    for idx in range(10):
        attachment_store.update_attachment(
            record["id"],
            timeline_event="heartbeat",
            timeline_detail=f"{idx}",
        )

    public = attachment_store.public_status(record["id"])
    assert public is not None
    assert len(public["history"]) == attachment_store.MAX_TIMELINE_EVENTS
    assert public["history"][0]["detail"] == "9"


def test_list_resumable_filters_status(tmp_path):
    source = _make_source(tmp_path)
    record = attachment_store.create_attachment(
        claim_id="claim-3",
        doc_id=None,
        local_path=source,
    )

    attachment_store.mark_parsing(record["id"], attempt=1)
    resumable = attachment_store.list_resumable()
    assert any(item["id"] == record["id"] for item in resumable)

    attachment_store.mark_ready(record["id"], artifacts={})
    resumable_after = attachment_store.list_resumable()
    assert all(item["id"] != record["id"] for item in resumable_after)
