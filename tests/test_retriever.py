import os
import numpy as np
import pytest
from pathlib import Path

import backend.utils as utils
from backend.retriever import Retriever

@pytest.fixture
def dummy_chunks():
    # 20 dummy sentence‐chunks
    return [
        {"text": f"sentence {i}", "meta": {"id": str(i), "type": "sentence"}}
        for i in range(20)
    ]

def test_max_sentences_cap(tmp_path, monkeypatch, dummy_chunks):
    idx_base = tmp_path / "idx"
    # monkey‐patch the real embed to just return random vectors
    def fake_embed(texts):
        # one‐dimensional embeddings so index.dim = 1
        arr = np.arange(len(texts), dtype="float32").reshape(-1, 1)
        # normalize for cosine
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return (arr / norms).astype("float32")
    monkeypatch.setattr(utils, "embed", fake_embed)

    # cap at 10 sentences
    r = Retriever(idx_base, max_sentences=10, min_score=0.0)
    r.build(dummy_chunks)

    # we should only keep the first 10 chunks
    assert len(r.chunks) == 10
    # and FAISS index should have 10 entries
    assert r.index.ntotal == 10

def test_faiss_min_score(tmp_path, monkeypatch, dummy_chunks):
    idx_base = tmp_path / "idx2"
    # fake embed that returns unit vectors where chunk 0 matches query perfectly
    def fake_embed(texts):
        # if text contains "0" return [1.0], else [0.0]
        arr = np.array([[1.0] if "sentence 0" in t else [0.0] for t in texts], dtype="float32")
        return arr
    monkeypatch.setattr(utils, "embed", fake_embed)

    # no sentence cap, but filter out anything with score < 0.5
    r = Retriever(idx_base, max_sentences=None, min_score=0.5)
    r.build(dummy_chunks)

    # query "sentence 0" → should only return the chunk with id "0"
    hits = r.query("sentence 0", k=5)
    assert len(hits) == 1
    assert hits[0]["meta"]["id"] == "0"

    # query "sentence 1" → embedding [0.0] so similarity = 0 → below threshold → no hits
    hits2 = r.query("sentence 1", k=5)
    assert hits2 == []