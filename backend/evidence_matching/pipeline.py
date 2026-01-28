"""Retrieve → rerank → NLI pipeline orchestration.

The full pipeline is implemented in a later task. This stub ensures importers
can safely reference :class:`EvidencePipeline` before the implementation lands.
"""

from __future__ import annotations


class EvidencePipeline:  # pragma: no cover - placeholder
    """Placeholder pipeline definition.

    The concrete implementation will be added alongside the pipeline task.
    """

    def __init__(self, *_args, **_kwargs):
        """Raise to signal that the concrete pipeline is added later."""
        raise NotImplementedError(
            "EvidencePipeline will be implemented in the pipeline task"
        )
