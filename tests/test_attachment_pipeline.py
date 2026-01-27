import json
from pathlib import Path

import pytest

from backend import attachment_pipeline, attachment_store
from backend.settings import settings


TEI_STUB = (
    '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
    "<text><body><p><s xml:id='s1'>Sentence one.</s>"
    "<s xml:id='s2'>Sentence two.</s></p></body></text></TEI>"
)


@pytest.fixture(autouse=True)
def attachment_dir(tmp_path):
    settings.ATTACHMENT_DIR = tmp_path / "attachments"
    yield


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
    )

    monkeypatch.setattr(
        attachment_pipeline.grobid_client,
        "extract_tei",
        lambda _: TEI_STUB,
    )

    def fake_embed(texts, model_name=None, mode=None):
        return [[float(idx)] for idx, _ in enumerate(texts, start=1)]

    monkeypatch.setattr(attachment_pipeline.utils, "embed", fake_embed)

    attachment_pipeline.process_attachment(record["id"])

    updated = attachment_store.get_attachment(record["id"])
    assert updated is not None
    assert updated["status"] == attachment_store.STATUS_MATCHED
    artifacts = updated["artifacts"]
    assert artifacts
    sentences_path = Path(artifacts["sentences"])
    rows = [json.loads(line) for line in sentences_path.read_text().splitlines()]
    assert rows[0]["embedding"] == [1.0]


def test_process_attachment_marks_error_after_retries(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    record = attachment_store.create_attachment(
        claim_id="claim-err",
        doc_id=None,
        local_path=source,
    )

    def boom(_):
        raise RuntimeError("explode")

    monkeypatch.setattr(attachment_pipeline.grobid_client, "extract_tei", boom)

    attachment_pipeline.process_attachment(record["id"], max_attempts=2)

    updated = attachment_store.get_attachment(record["id"])
    assert updated is not None
    assert updated["status"] == attachment_store.STATUS_ERROR
    assert updated["error"] == "explode"
