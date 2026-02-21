"""Versioned contract models for pipeline stage artifacts.

Phase 10 introduces explicit stage boundaries using JSON artifacts persisted in
object storage with spine pointer rows.
"""

from .pipeline_v1 import (
    STAGES,
    Caps,
    PipelineArtifactError,
    PipelineArtifactComponent,
    PipelineArtifactEnvelope,
    ExtractArtifact,
    CiteSpansArtifact,
    RetrievalArtifact,
    FilterArtifact,
    RerankArtifact,
    NliArtifact,
    artifact_type_for,
    stage_object_key,
    deterministic_json_bytes,
    span_id_for,
    candidate_id_for,
)
from .upgrade import upgrade_contract_payload, validate_contract_payload

__all__ = [
    "STAGES",
    "Caps",
    "PipelineArtifactError",
    "PipelineArtifactComponent",
    "PipelineArtifactEnvelope",
    "ExtractArtifact",
    "CiteSpansArtifact",
    "RetrievalArtifact",
    "FilterArtifact",
    "RerankArtifact",
    "NliArtifact",
    "artifact_type_for",
    "stage_object_key",
    "deterministic_json_bytes",
    "span_id_for",
    "candidate_id_for",
    "upgrade_contract_payload",
    "validate_contract_payload",
]
