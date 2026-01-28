"""Deterministic citation seeding helpers.

This module is populated in a later plan task. It is intentionally kept
lightweight here so that package imports succeed before the matcher logic is
implemented.
"""

from __future__ import annotations

# Placeholder exports to keep static analyzers happy. Real implementations will
# be provided in the deterministic matcher task.
__all__ = ["seed_windows"]


def seed_windows(*_args, **_kwargs):  # pragma: no cover - placeholder
    """Temporary no-op that will be replaced with real seeding logic."""
    raise NotImplementedError(
        "seed_windows will be implemented in the deterministic matcher task"
    )
