"""Evidence matching pipeline primitives and helpers."""

from .types import CandidateSpan, EvidenceCandidate, RankScores
from . import loaders, deterministic_matcher, pipeline, serializers, store, service
from .service import EvidenceMatchingService, evidence_service
from .store import EvidenceRunStore

__all__ = [
    "CandidateSpan",
    "EvidenceCandidate",
    "RankScores",
    "loaders",
    "deterministic_matcher",
    "pipeline",
    "serializers",
    "store",
    "service",
    "EvidenceRunStore",
    "EvidenceMatchingService",
    "evidence_service",
]
