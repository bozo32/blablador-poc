import pytest
from pathlib import Path
from backend.parser import tei_to_chunks

# Adjust this path if your TEI source directory differs
TEI_FILE = Path(__file__).parents[1] / "data" / "Alae-Carew.pdf.tei.xml"

def test_tei_to_chunks_count():
    """
    Ensure that tei_to_chunks returns a non-empty list of sentence chunks.
    """
    chunks = tei_to_chunks(TEI_FILE)
    assert isinstance(chunks, list), "tei_to_chunks should return a list"
    assert len(chunks) > 0, "Expected at least one chunk from the TEI file"

def test_chunk_structure():
    """
    Validate that each chunk has the expected structure and metadata fields.
    """
    chunks = tei_to_chunks(TEI_FILE)
    chunk = chunks[0]
    # Top-level keys
    assert "text" in chunk, "Chunk must have a 'text' field"
    assert "meta" in chunk, "Chunk must have a 'meta' field"
    
    meta = chunk["meta"]
    # Required metadata keys
    expected_meta_keys = {"type", "id", "p_id", "section_id", "section_type", "section_head"}
    assert expected_meta_keys.issubset(meta.keys()), f"Missing keys in meta: {expected_meta_keys - set(meta.keys())}"
    
    # Type should be 'sentence'
    assert meta["type"] == "sentence", f"Expected meta['type']=='sentence', got {meta['type']}"