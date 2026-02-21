"""Upgrade + validation shims for pipeline stage contracts."""

from __future__ import annotations

import re

from . import pipeline_v1


_ARTIFACT_VER_RE = re.compile(r"@v(?P<v>\d+)$")


def _infer_schema_version(payload: dict) -> int | None:
    raw = payload.get("schema_version")
    if raw is not None:
        try:
            return int(raw)
        except Exception:
            return None
    at = str(payload.get("artifact_type") or "").strip()
    m = _ARTIFACT_VER_RE.search(at)
    if not m:
        return None
    try:
        return int(m.group("v"))
    except Exception:
        return None


def latest_schema_version() -> int:
    return 1


def upgrade_contract_payload(payload: dict) -> dict:
    """Upgrade older payload dicts to the latest supported schema.

    Additive-only evolution: newer versions can be upgraded forward in code.
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    out = dict(payload)

    v = _infer_schema_version(out)
    if v is None:
        # v1 is the initial published schema; treat missing version as v1.
        v = 1
        out.setdefault("schema_version", v)

    v = int(v)
    if v <= 0:
        raise ValueError("schema_version must be >= 1")

    latest = latest_schema_version()
    if v > latest:
        raise ValueError(
            f"Unsupported schema_version={v}; latest supported is v{latest}"
        )

    # v1 -> latest (v1) no-op.
    if v == 1:
        return out

    # Future: add vN->vN+1 transforms here.
    return out


def validate_contract_payload(payload: dict) -> dict:
    """Upgrade then validate a payload, returning a normalized dict."""
    upgraded = upgrade_contract_payload(payload)
    v = int(upgraded.get("schema_version") or 0)
    if v == 1:
        return pipeline_v1.validate_v1(upgraded)
    raise ValueError(f"No validator registered for schema_version={v}")
