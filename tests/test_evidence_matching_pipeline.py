from __future__ import annotations

from pathlib import Path

import pytest

from backend import attachment_store
from backend.evidence_matching import deterministic_matcher, loaders, serializers
from backend.evidence_matching.pipeline import EvidencePipeline
from backend.evidence_matching.types import Provenance
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


def test_matcher_respects_score_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_WINDOW_SIZE", 1)
    monkeypatch.setattr(settings, "EVIDENCE_SEED_LIMIT", 10)
    _make_attachment(
        tmp_path,
        "claim-threshold",
        [
            {"sentence_id": "t1", "text": "alpha evidence", "page": 1, "position": 0},
            {"sentence_id": "t2", "text": "beta evidence", "page": 1, "position": 1},
        ],
    )
    _make_attachment(
        tmp_path,
        "claim-threshold",
        [
            {
                "sentence_id": "t3",
                "text": "completely unrelated",
                "page": 2,
                "position": 0,
            },
        ],
    )
    monkeypatch.setattr(settings, "EVIDENCE_BM25_MIN_SCORE", 0.0)
    windows = loaders.load_claim_windows("claim-threshold")
    baseline = deterministic_matcher.seed_windows("alpha evidence", windows)
    assert len(baseline) >= 2
    low_score = float(baseline[-1].scores.bm25 or 0.0)
    monkeypatch.setattr(settings, "EVIDENCE_BM25_MIN_SCORE", low_score + 0.1)
    filtered = deterministic_matcher.seed_windows("alpha evidence", windows)
    assert len(filtered) < len(baseline)


def test_matcher_tags_provenance_and_badges(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_WINDOW_SIZE", 1)
    monkeypatch.setattr(settings, "EVIDENCE_SEED_LIMIT", 10)
    cited_id = _make_attachment(
        tmp_path,
        "claim-provenance",
        [{"sentence_id": "p1", "text": "alpha mechanism", "page": 1, "position": 0}],
    )
    _make_attachment(
        tmp_path,
        "claim-provenance",
        [{"sentence_id": "p2", "text": "beta mechanism", "page": 1, "position": 0}],
    )
    monkeypatch.setattr(settings, "EVIDENCE_BM25_MIN_SCORE", 0.0)
    windows = loaders.load_claim_windows("claim-provenance")
    seeds = deterministic_matcher.seed_windows(
        "alpha beta", windows, cited_attachment_ids=[cited_id]
    )
    cited_candidates = [seed for seed in seeds if seed.provenance is Provenance.CITED]
    assert cited_candidates, "Expected at least one cited candidate"
    heuristic_candidates = [
        seed for seed in seeds if seed.provenance is Provenance.HEURISTIC
    ]
    assert heuristic_candidates, "Expected at least one heuristic candidate"
    assert "ambiguous-attachment" in heuristic_candidates[0].badges


def test_matcher_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_WINDOW_SIZE", 1)
    monkeypatch.setattr(settings, "EVIDENCE_SEED_LIMIT", 10)
    monkeypatch.setattr(settings, "EVIDENCE_BM25_MIN_SCORE", 0.0)
    _make_attachment(
        tmp_path,
        "claim-repro",
        [
            {"sentence_id": "r1", "text": "alpha beta", "page": 1, "position": 0},
            {"sentence_id": "r2", "text": "beta gamma", "page": 1, "position": 1},
        ],
    )
    windows = loaders.load_claim_windows("claim-repro")
    first = deterministic_matcher.seed_windows("alpha beta", windows)
    second = deterministic_matcher.seed_windows("alpha beta", windows)
    assert [cand.id for cand in first] == [cand.id for cand in second]
    assert [cand.scores.bm25 for cand in first] == [cand.scores.bm25 for cand in second]


def test_pipeline_orders_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_WINDOW_SIZE", 1)
    monkeypatch.setattr(settings, "EVIDENCE_BM25_MIN_SCORE", 0.0)
    monkeypatch.setattr(settings, "EVIDENCE_SEED_LIMIT", 10)
    monkeypatch.setattr(settings, "EVIDENCE_MAX_CANDIDATES", 2)
    seeds = _build_seeds_for_pipeline(tmp_path)

    class FakeNLI:
        @staticmethod
        def assess(_claim, passages, metadatas, **_kwargs):
            return [
                {"id": metadatas[0]["id"], "label": "entailment", "score": 0.91},
                {"id": metadatas[1]["id"], "label": "contradiction", "score": 0.87},
            ]

    pipeline = EvidencePipeline(settings=settings, nli_module=FakeNLI)
    results = pipeline.run(
        claim_id="claim-pipeline",
        claim_text="alpha mechanism",
        seeds=seeds,
    )
    assert len(results) == 2
    assert results[0].scores.position == 1
    summary = pipeline.summarize_labels(results)
    assert summary["entails"] == 1
    assert summary["contradicts"] == 1


def test_serializers_include_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCE_WINDOW_SIZE", 1)
    monkeypatch.setattr(settings, "EVIDENCE_BM25_MIN_SCORE", 0.0)
    monkeypatch.setattr(settings, "EVIDENCE_SEED_LIMIT", 10)
    monkeypatch.setattr(settings, "EVIDENCE_MAX_CANDIDATES", 2)
    seeds = _build_seeds_for_pipeline(tmp_path)

    class NeutralNLI:
        @staticmethod
        def assess(_claim, passages, metadatas, **_kwargs):
            return []

    pipeline = EvidencePipeline(settings=settings, nli_module=NeutralNLI)
    results = pipeline.run(
        claim_id="claim-pipeline",
        claim_text="alpha mechanism",
        seeds=seeds,
    )
    serialized = serializers.serialize_candidates(results, max_text=20)
    assert len(serialized) == 2
    first = serialized[0]
    assert len(first["text"]) <= 20
    assert "page" in first["metadata"]
    assert "section" in first["metadata"]
    assert "bbox_count" in first["metadata"]
    assert first["metadata"]["bbox_count"] >= 0
    assert first["spans"]
    assert "highlights" in first


def _build_seeds_for_pipeline(tmp_path: Path) -> list:
    _make_attachment(
        tmp_path,
        "claim-pipeline",
        [
            {
                "sentence_id": "pp1",
                "text": "alpha evidence chunk",
                "page": 1,
                "position": 0,
            },
            {
                "sentence_id": "pp2",
                "text": "supporting rationale",
                "page": 2,
                "position": 1,
            },
        ],
    )
    _make_attachment(
        tmp_path,
        "claim-pipeline",
        [
            {
                "sentence_id": "pp3",
                "text": "contradicting fragment",
                "page": 3,
                "position": 0,
            },
        ],
    )
    windows = loaders.load_claim_windows("claim-pipeline")
    return deterministic_matcher.seed_windows("alpha evidence", windows)
