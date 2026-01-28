"""Serializer utilities for evidence matching outputs.

Real implementations arrive with the pipeline task. The placeholder keeps
module imports stable for earlier tasks that need to reference this file.
"""

from __future__ import annotations


def serialize_candidates(*_args, **_kwargs):  # pragma: no cover - placeholder
    raise NotImplementedError(
        "serialize_candidates will be implemented in the pipeline task"
    )


__all__ = ["serialize_candidates"]
