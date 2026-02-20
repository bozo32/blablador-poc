from pathlib import Path

from backend import attachment_store


def _make_source(tmp_path: Path, name: str = "source.pdf") -> Path:
    source = tmp_path / name
    source.write_bytes(b"%PDF-1.4\nSample attachment")
    return source


def test_unassigned_attachment_can_be_archived_and_filtered(tmp_path):
    source = _make_source(tmp_path)
    record = attachment_store.create_attachment(
        claim_id=None,
        doc_id=None,
        local_path=source,
        filename="unassigned.pdf",
    )
    assert record.get("claim_id") is None
    assert record.get("archived") is False

    listed = attachment_store.list_attachments()
    assert any(item["id"] == record["id"] for item in listed)

    archived = attachment_store.set_archived(record["id"], archived=True)
    assert archived.get("archived") is True
    assert archived.get("archived_at")

    listed_default = attachment_store.list_attachments()
    assert all(item["id"] != record["id"] for item in listed_default)

    listed_all = attachment_store.list_attachments(archived=None)
    assert any(item["id"] == record["id"] for item in listed_all)


def test_unassigned_attachment_can_be_placed_and_persists_metadata(tmp_path):
    source = _make_source(tmp_path, name="placed.pdf")
    record = attachment_store.create_attachment(
        claim_id=None,
        doc_id=None,
        local_path=source,
        filename="placed.pdf",
    )

    updated = attachment_store.set_placement(
        record["id"],
        claim_id="claim-123",
        doc_id="doc-abc",
        citation_index=7,
        target_id="ref-42",
    )
    assert updated.get("claim_id") == "claim-123"
    assert updated.get("doc_id") == "doc-abc"
    assert updated.get("citation_index") == 7
    assert updated.get("target_id") == "ref-42"

    reloaded = attachment_store.get_attachment(record["id"])
    assert reloaded is not None
    assert reloaded.get("claim_id") == "claim-123"
    assert reloaded.get("doc_id") == "doc-abc"
    assert reloaded.get("citation_index") == 7
    assert reloaded.get("target_id") == "ref-42"

    public = attachment_store.public_status(record["id"])
    assert public is not None
    assert "file_path" not in public
    events = [evt["event"] for evt in public.get("history", [])]
    assert "placement" in events
