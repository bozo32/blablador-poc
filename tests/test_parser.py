from pathlib import Path

from backend.parser import tei_to_chunks


def _write_stub_tei(tmp_path: Path) -> Path:
    tei_path = tmp_path / "stub.tei.xml"
    tei_path.write_text(
        (
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            "<text><body>"
            "<div xml:id='secA' type='intro'><head>Intro</head>"
            "<p xml:id='p1'>"
            "<s xml:id='s1'>Sentence one.</s>"
            "<s xml:id='s2'>Sentence two.</s>"
            "</p>"
            "</div>"
            "</body></text></TEI>"
        ),
        encoding="utf-8",
    )
    return tei_path


def test_tei_to_chunks_count(tmp_path: Path):
    """tei_to_chunks returns a non-empty list."""
    chunks = tei_to_chunks(_write_stub_tei(tmp_path))
    assert isinstance(chunks, list), "tei_to_chunks should return a list"
    assert len(chunks) > 0, "Expected at least one chunk from the TEI file"


def test_chunk_structure(tmp_path: Path):
    """tei_to_chunks chunks include expected meta keys."""
    chunks = tei_to_chunks(_write_stub_tei(tmp_path))
    chunk = chunks[0]
    # Top-level keys
    assert "text" in chunk, "Chunk must have a 'text' field"
    assert "meta" in chunk, "Chunk must have a 'meta' field"

    meta = chunk["meta"]
    # Required metadata keys
    expected_meta_keys = {
        "type",
        "id",
        "p_id",
        "section_id",
        "section_type",
        "section_head",
    }
    assert expected_meta_keys.issubset(
        meta.keys()
    ), f"Missing keys in meta: {expected_meta_keys - set(meta.keys())}"

    # Type should be 'sentence_window' (parser emits 1-3 sentence sliding windows)
    assert meta["type"] == "sentence_window"
