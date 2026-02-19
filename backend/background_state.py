"""Persistent global pause/resume state for background work.

Phase 09.3 removes durable local `data/**` state, so this module persists the
pause toggle in Postgres.
"""

from __future__ import annotations

import threading
from typing import Optional

from backend.db.pg import connect
from backend.settings import settings as app_settings


_LOCK = threading.RLock()


def _project_id() -> str:
    return str(getattr(app_settings, "DEFAULT_PROJECT_ID", "default") or "default")


def _default_state() -> dict:
    return {
        "paused": False,
        "updated_at": None,
        "reason": None,
    }


def get_state() -> dict:
    """Return the persisted background state.

    Always returns a dict containing at least:
      - paused: bool
      - updated_at: str | None
      - reason: str | None
    """
    pid = _project_id()
    with _LOCK:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT paused, updated_at, reason
                    FROM background_state
                    WHERE project_id=%s
                    """,
                    (pid,),
                )
                row = cur.fetchone()
                if not row:
                    return _default_state()

        paused, updated_at, reason = row
        return {
            "paused": bool(paused),
            "updated_at": updated_at.isoformat().replace("+00:00", "Z")
            if updated_at is not None
            else None,
            "reason": str(reason) if reason is not None else None,
        }


def set_paused(paused: bool, reason: Optional[str] = None) -> dict:
    """Persist the pause toggle and return the new state."""
    pid = _project_id()
    next_paused = bool(paused)
    next_reason = (str(reason).strip() if reason is not None else None) or None

    with _LOCK:
        current = get_state()
        if (
            bool(current.get("paused")) == next_paused
            and (current.get("reason") or None) == next_reason
        ):
            return current

        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO background_state(project_id, paused, updated_at, reason)
                    VALUES (%s, %s, now(), %s)
                    ON CONFLICT(project_id) DO UPDATE
                      SET paused=excluded.paused,
                          updated_at=excluded.updated_at,
                          reason=excluded.reason
                    """,
                    (pid, next_paused, next_reason),
                )

        return get_state()
