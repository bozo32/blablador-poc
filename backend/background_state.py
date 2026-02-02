"""Persistent global pause/resume state for background work.

This module intentionally uses stdlib only and stores state under ./data/.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional


_LOCK = threading.Lock()


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _state_path() -> Path:
    return Path("data") / "background_state.json"


def _default_state() -> dict:
    return {
        "paused": False,
        "updated_at": _utcnow(),
        "reason": None,
    }


def _load_state(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _default_state()
    except Exception:
        # Corrupt or partial file should not crash callers.
        return _default_state()

    paused = bool(payload.get("paused", False))
    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at.strip():
        updated_at = _utcnow()
    reason = payload.get("reason")
    if reason is not None and not isinstance(reason, str):
        reason = str(reason)
    return {
        "paused": paused,
        "updated_at": updated_at,
        "reason": reason,
    }


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_name = handle.name
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp_name, path)


def get_state() -> dict:
    """Return the persisted background state.

    Always returns a dict containing at least:
      - paused: bool
      - updated_at: str
    """
    with _LOCK:
        return _load_state(_state_path())


def set_paused(paused: bool, reason: Optional[str] = None) -> dict:
    """Persist the pause toggle and return the new state."""
    with _LOCK:
        current = _load_state(_state_path())
        next_state = {
            "paused": bool(paused),
            "updated_at": _utcnow(),
            "reason": (str(reason).strip() if reason is not None else None) or None,
        }
        # Avoid rewriting if nothing changes other than whitespace in reason.
        if (
            bool(current.get("paused")) == next_state["paused"]
            and (current.get("reason") or None) == next_state["reason"]
        ):
            return {
                "paused": next_state["paused"],
                "updated_at": current.get("updated_at") or next_state["updated_at"],
                "reason": next_state["reason"],
            }
        _atomic_write(_state_path(), next_state)
        return dict(next_state)
