"""Retrieve → rerank → NLI orchestration for evidence candidates."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from backend import nli


class _FallbackHybridPipeline:  # pragma: no cover - simple shim for tests
    @staticmethod
    def build_all(*_args, **_kwargs):
        return {}


DefaultHybridPipeline: Any = _FallbackHybridPipeline

try:  # pragma: no cover - override fallback when FAISS is installed
    from backend.hybrid import HybridPipeline as _RealHybridPipeline

    DefaultHybridPipeline = _RealHybridPipeline  # type: ignore[assignment]
except ModuleNotFoundError:  # noqa: F401 - keep fallback class
    pass


from backend.settings import AppSettings, settings as app_settings

from .types import EvidenceCandidate, EvidenceLabel


class EvidencePipeline:
    """Coarse pipeline that enriches deterministic seeds with ranking + NLI."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        hybrid_cls: Any | None = None,
        nli_module: Any = nli,
    ) -> None:
        """Create a pipeline with optional dependency overrides."""
        self.settings = settings or app_settings
        self.hybrid_cls = hybrid_cls or DefaultHybridPipeline
        self.nli = nli_module

    def run(
        self,
        *,
        claim_id: str,
        claim_text: str,
        seeds: Sequence[EvidenceCandidate],
        attachment_context: dict | None = None,
    ) -> list[EvidenceCandidate]:
        """Execute retrieve → rerank → NLI over the provided seed candidates."""
        if not seeds:
            return []

        candidates = [seed for seed in seeds]
        self._maybe_prime_hybrid(attachment_context)
        self._score_candidates(claim_text, candidates)
        self._apply_nli(claim_text, candidates)
        cap = getattr(self.settings, "EVIDENCE_MAX_CANDIDATES", len(candidates))
        trimmed = candidates[:cap]
        for idx, cand in enumerate(trimmed, start=1):
            cand.scores.position = idx
            if cand.token_saliencies is None and cand.metadata.get("token_scores"):
                cand.token_saliencies = cand.metadata["token_scores"]
        return trimmed

    def summarize_labels(
        self, candidates: Iterable[EvidenceCandidate]
    ) -> dict[str, int]:
        """Return a label distribution for downstream QA."""
        counter = Counter(c.label for c in candidates)
        return {label.value: counter.get(label, 0) for label in EvidenceLabel}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_prime_hybrid(self, attachment_context: dict | None) -> None:
        if not attachment_context:
            return
        # Placeholder for future HybridPipeline hydration. The return value is
        # ignored for now, but the call ensures attachment_context is validated.
        try:  # pragma: no cover - trivial guard
            self.hybrid_cls.build_all(**attachment_context)
        except Exception:  # noqa: BLE001 - hybrid prep is best-effort right now
            return

    def _score_candidates(
        self, claim_text: str, candidates: list[EvidenceCandidate]
    ) -> None:
        for cand in candidates:
            metadata = dict(cand.metadata or {})
            bm25 = cand.scores.bm25 or 0.0
            faiss_score = metadata.get("faiss_score") or cand.scores.faiss or 0.0
            sbert_score = metadata.get("_sbert_score") or cand.scores.sbert or 0.0
            colbert_score = metadata.get("colbert_score") or cand.scores.colbert or 0.0
            combined = (
                (bm25 * 0.5)
                + (faiss_score * 0.3)
                + (sbert_score * 0.15)
                + (colbert_score * 0.05)
            )
            cand.scores.faiss = faiss_score or None
            cand.scores.sbert = sbert_score or None
            cand.scores.colbert = colbert_score or None
            cand.scores.combined = combined
            metadata.setdefault("query", claim_text)
            metadata.setdefault("seed_score", bm25)
            cand.metadata = metadata

        candidates.sort(key=lambda c: c.scores.combined or 0.0, reverse=True)

    def _apply_nli(self, claim_text: str, candidates: list[EvidenceCandidate]) -> None:
        metadatas = []
        passages = []
        for cand in candidates:
            meta = dict(cand.metadata or {})
            meta.setdefault("id", cand.id)
            meta.setdefault("chunk_id", cand.id)
            meta.setdefault("type", "seed")
            metadatas.append(meta)
            passages.append(cand.text)

        try:
            nli_results = self.nli.assess(
                claim_text,
                passages,
                metadatas,
                nli_model=getattr(self.settings, "NLI_MODEL", None),
            )
        except Exception:  # pragma: no cover - fall back to neutral labels
            nli_results = []

        label_map = {}
        for result in nli_results:
            cid = result.get("id") or result.get("chunk_id")
            if cid:
                label_map.setdefault(cid, result)

        for cand in candidates:
            result = label_map.get(cand.id)
            if result:
                label_value = result.get("label", "").lower()
                if "contrad" in label_value:
                    cand.label = EvidenceLabel.CONTRADICTS
                elif "entail" in label_value:
                    cand.label = EvidenceLabel.ENTAILS
                else:
                    cand.label = EvidenceLabel.NEUTRAL
                cand.scores.nli = result.get("score")
            else:
                cand.label = EvidenceLabel.NEUTRAL
