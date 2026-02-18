"""Evidence matching primitives.

Keep this module import-light: importing `pipeline` pulls in heavyweight ML
dependencies (torch/transformers). Callers should import those modules
explicitly when needed.
"""

from .types import CandidateSpan, EvidenceCandidate, RankScores
from . import loaders, deterministic_matcher, serializers

__all__ = [
    "CandidateSpan",
    "EvidenceCandidate",
    "RankScores",
    "loaders",
    "deterministic_matcher",
    "serializers",
]
