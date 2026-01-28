"""Deterministic BM25-based seeding for evidence windows."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Iterable, Sequence

from rank_bm25 import BM25Okapi

try:  # pragma: no cover - optional heavy dependency
    import spacy
except Exception:  # noqa: BLE001 - fall back to regex tokeniser
    spacy = None

from backend.settings import AppSettings, settings as app_settings

from .loaders import AttachmentWindow
from .types import EvidenceCandidate, EvidenceLabel, Provenance, RankScores

__all__ = ["seed_windows"]


def seed_windows(
    claim_text: str,
    windows: Sequence[AttachmentWindow],
    *,
    cited_attachment_ids: Iterable[str] | None = None,
    settings_override: AppSettings | None = None,
) -> list[EvidenceCandidate]:
    """Run deterministic BM25 scoring to seed candidate windows."""
    if not windows:
        return []

    cfg = settings_override or app_settings
    cited_ids = {str(value) for value in cited_attachment_ids or []}
    window_tokens = [_tokenize(window.text) for window in windows]
    query_tokens = _tokenize(claim_text)
    bm25 = BM25Okapi(window_tokens)
    scores = bm25.get_scores(query_tokens or [""])  # keep deterministic order
    min_score = float(getattr(cfg, "EVIDENCE_BM25_MIN_SCORE", 0.0))
    seed_limit = int(getattr(cfg, "EVIDENCE_SEED_LIMIT", 50))

    indexed_scores = [
        (idx, float(score)) for idx, score in enumerate(scores) if score >= min_score
    ]
    indexed_scores.sort(key=lambda item: (-item[1], windows[item[0]].window_id))
    if seed_limit > 0:
        indexed_scores = indexed_scores[:seed_limit]

    counts = Counter(window.attachment_id for window in windows if window.attachment_id)
    multiple_attachments = len([key for key in counts if key]) > 1

    candidates: list[EvidenceCandidate] = []
    for position, (idx, score) in enumerate(indexed_scores, start=1):
        window = windows[idx]
        provenance = (
            Provenance.CITED
            if window.attachment_id in cited_ids
            else Provenance.HEURISTIC
        )
        badges = list(window.metadata.get("badges", []))
        if provenance is Provenance.HEURISTIC and multiple_attachments:
            badges.append("ambiguous-attachment")
        metadata = dict(window.metadata)
        metadata["bm25_score"] = score
        metadata["seed_rank"] = position
        metadata["seed_provenance"] = provenance.value
        payload = {
            "window_id": window.window_id,
            "claim_id": window.claim_id,
            "attachment_id": window.attachment_id,
            "text": window.text,
            "spans": window.spans,
            "metadata": metadata,
            "badges": badges,
        }
        scores_obj = RankScores(bm25=score, combined=score, position=position)
        candidates.append(
            EvidenceCandidate.from_window(
                payload,
                label=EvidenceLabel.NEUTRAL,
                provenance=provenance,
                scores=scores_obj,
            )
        )
    return candidates


@lru_cache(maxsize=1)
def _get_tokenizer():  # pragma: no cover - simple cache wrapper
    if spacy is None:
        return None
    return spacy.blank("en")


def _tokenize(text: str) -> list[str]:
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return re.findall(r"\w+", text.lower())
    doc = tokenizer.make_doc(text)
    return [token.text.lower() for token in doc if token.text.strip()]
