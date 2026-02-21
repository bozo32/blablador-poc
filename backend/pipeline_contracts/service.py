"""Service layer for Phase 10 pipeline stage contracts."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

from backend.contracts.pipeline_v1 import (
    STAGES,
    Caps,
    artifact_type_for,
    candidate_id_for,
    deterministic_json_bytes,
)
from backend.spine import pipeline_artifacts, pipeline_runs


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_caps() -> dict:
    caps = Caps()
    return {
        "candidates": int(caps.candidates),
        "rerank": int(caps.rerank),
        "nli": int(caps.nli),
    }


def _merge_caps(base: dict, override: dict | None) -> dict:
    out = dict(base or {})
    for k, v in dict(override or {}).items():
        if v is None:
            continue
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    out.setdefault("candidates", 200)
    out.setdefault("rerank", 50)
    out.setdefault("nli", 12)
    return out


def _fingerprint(*, work_id: str, caps: dict, settings_json: dict) -> str:
    payload = {
        "work_id": str(work_id),
        "caps": dict(caps or {}),
        "settings": dict(settings_json or {}),
    }
    return hashlib.sha256(deterministic_json_bytes(payload)).hexdigest()


def create_run(
    work_id: str, caps_override: dict | None = None, note: str | None = None
) -> dict:
    wid = str(work_id or "").strip()
    if not wid:
        raise ValueError("work_id is required")
    settings_json: dict = {}
    caps = _merge_caps(_default_caps(), caps_override)
    fp = _fingerprint(work_id=wid, caps=caps, settings_json=settings_json)
    run = pipeline_runs.create_run(
        wid,
        caps=caps,
        settings_json=settings_json,
        input_fingerprint=fp,
        note=note,
    )
    return {
        "run_id": run.get("run_id"),
        "work_id": run.get("work_id"),
        "created_at": run.get("created_at"),
        "caps": run.get("caps_json") or {},
        "input_fingerprint": run.get("input_fingerprint"),
    }


def _ensure_candidate_ids(*, run_id: str, stage: str, data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("data must be an object")
    by_target = data.get("by_target")
    if not isinstance(by_target, dict):
        return data

    for target_id, group in by_target.items():
        if not isinstance(group, dict):
            continue
        candidates = group.get("candidates")
        if not isinstance(candidates, list):
            continue
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            selector = cand.get("selector")
            source_doc_id = cand.get("source_doc_id")
            window_fp = cand.get("window_fingerprint")
            if not isinstance(selector, dict) or not selector:
                raise ValueError("candidate.selector is required")
            if not str(source_doc_id or "").strip():
                raise ValueError("candidate.source_doc_id is required")

            computed = candidate_id_for(
                run_id=str(run_id),
                stage=str(stage),
                target_id=str(target_id),
                source_doc_id=str(source_doc_id),
                selector=dict(selector),
                window_fingerprint=str(window_fp).strip()
                if window_fp is not None
                else None,
            )
            existing = str(cand.get("candidate_id") or "").strip()
            if not existing:
                cand["candidate_id"] = computed
            elif existing != computed:
                raise ValueError("candidate_id must be computed via candidate_id_for")
    return data


def store_stage(
    run_id: str,
    stage: str,
    *,
    status: str,
    data: dict,
    warnings: Optional[list[str]] = None,
    error: Optional[dict] = None,
    component: Optional[dict] = None,
    caps_override: Optional[dict] = None,
) -> dict:
    rid = str(run_id or "").strip()
    st = str(stage or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    if st not in STAGES:
        raise ValueError(f"Unknown stage: {st}")

    run = pipeline_runs.get_run(rid)
    if run is None:
        raise KeyError("run not found")

    work_id = str(run.get("work_id") or "").strip()
    if not work_id:
        raise ValueError("Run missing work_id")

    caps = _merge_caps(dict(run.get("caps_json") or {}), caps_override)
    payload_data: dict = dict(data or {})
    if st in {"retrieval", "filter", "rerank", "nli"}:
        payload_data = _ensure_candidate_ids(run_id=rid, stage=st, data=payload_data)

    payload = {
        "schema_version": 1,
        "artifact_type": artifact_type_for(st, 1),
        "run_id": rid,
        "stage": st,
        "work_id": work_id,
        "created_at": _utcnow_z(),
        "status": str(status or "").strip(),
        "warnings": [str(w) for w in (warnings or []) if str(w).strip()],
        "error": error,
        "component": component,
        "caps": caps,
        "input_fingerprint": str(run.get("input_fingerprint") or "").strip(),
        "data": payload_data,
    }
    return pipeline_artifacts.put_stage_payload(payload_dict=payload)


def fetch_stage(run_id: str, stage: str) -> dict | None:
    rid = str(run_id or "").strip()
    st = str(stage or "").strip()
    if not rid or not st:
        return None
    return pipeline_artifacts.get_stage_payload(rid, st)


__all__ = ["create_run", "store_stage", "fetch_stage"]
