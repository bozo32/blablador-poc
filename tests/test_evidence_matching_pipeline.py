from __future__ import annotations

from pathlib import Path

import pytest

from backend import attachment_store
from backend.evidence_matching import loaders
from backend.settings import settings


@pytest.fixture(autouse=True)
def attachment_workspace(tmp_path):
    settings.ATTACHMENT_DIR = tmp_path / "attachments"
    attachment_store.clear_sentence_cache()
    yield
    attachment_store.clear_sentence_cache()


def _make_attachment(tmp_path: Path, claim_id: str, sentences: list[dict]) -> str:
    source = tmp_path / f"{claim_id}.pdf"
    source.write_bytes(b"%PDF-1.4 test")
    record = attachment_store.create_attachment(
        claim_id=claim_id,
        doc_id="doc-x",
        local_path=source,
    )
    artifacts = attachment_store.save_artifacts(
        record["id"],
        tei_xml="<TEI/>",
        tei_json={"metadata": {}},
        sentences=sentences,
    )
    attachment_store.mark_matched(record["id"], artifacts)
    return record["id"]


def test_loader_skips_blank_sentences(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_WINDOW_SIZE", 2)
    monkeypatch.setattr(settings, "EVIDENCE_WINDOW_STRIDE", 2)
    attachment_id = _make_attachment(
        tmp_path,
        "claim-blanks",
        [
            {"sentence_id": "s1", "text": "Alpha", "page": 1, "position": 0},
            {"sentence_id": "s2", "text": "   ", "page": 1, "position": 1},
            {"sentence_id": "s3", "text": "Beta", "page": 2, "position": 2},
        ],
    )

    windows = loaders.load_attachment_windows(
        claim_id="claim-blanks", attachment_id=attachment_id
    )

    assert len(windows) == 1
    assert windows[0].tokens == ["Alpha", "Beta"]
    assert all(span.text for span in windows[0].spans)


def test_loader_normalizes_missing_page(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_WINDOW_SIZE", 1)
    attachment_id = _make_attachment(
        tmp_path,
        "claim-pages",
        [
            {"sentence_id": "a1", "text": "Gamma", "page": None, "position": 0},
        ],
    )

    windows = loaders.load_attachment_windows(
        claim_id="claim-pages", attachment_id=attachment_id
    )

    assert windows
    assert windows[0].spans[0].page == "Unknown"


def test_loader_preserves_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_WINDOW_SIZE", 1)
    vector = [0.1, 0.2, 0.3]
    attachment_id = _make_attachment(
        tmp_path,
        "claim-embed",
        [
            {
                "sentence_id": "e1",
                "text": "Embedding kept",
                "page": 3,
                "position": 0,
                "embedding": vector,
            }
        ],
    )

    windows = loaders.load_attachment_windows(
        claim_id="claim-embed", attachment_id=attachment_id
    )

    assert windows
    assert windows[0].spans[0].embedding == vector
