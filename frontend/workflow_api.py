from __future__ import annotations

from typing import Any, Optional

import requests
from datetime import datetime, timezone


DEFAULT_TIMEOUT_S = 30


class WorkflowApiError(RuntimeError):
    pass


def _root(api_url: str) -> str:
    return str(api_url or "").rstrip("/")


def _json(resp: requests.Response) -> dict:
    try:
        return resp.json() if resp.text else {}
    except Exception as exc:
        raise WorkflowApiError(
            f"Non-JSON response {resp.status_code}: {resp.text[:200]}"
        ) from exc


def start_claimspan_run(
    api_url: str, *, claim_id: str, reviewer_uid: str, citing_doc_id: str
) -> dict:
    url = f"{_root(api_url)}/workflow/claimspans/{str(claim_id)}/runs"
    payload = {
        "reviewer_uid": str(reviewer_uid or "default"),
        "citing_doc_id": str(citing_doc_id or "").strip(),
    }
    try:
        resp = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
        return _json(resp)
    except requests.RequestException as exc:
        raise WorkflowApiError(str(exc)) from exc


def finalize_assessment(
    api_url: str,
    *,
    claim_id: str,
    reviewer_uid: str,
    citing_doc_id: str,
    judgment_snapshot: dict,
    rollup_label: Optional[str] = None,
    by_target: Optional[dict] = None,
    assessed_at: Optional[str] = None,
) -> dict:
    url = f"{_root(api_url)}/workflow/claimspans/{str(claim_id)}/assessment/finalize"
    ts = assessed_at
    if ts is None or not str(ts).strip():
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    payload: dict[str, Any] = {
        "reviewer_uid": str(reviewer_uid or "default"),
        "citing_doc_id": str(citing_doc_id or "").strip(),
        "assessed_at": str(ts),
        "rollup_label": rollup_label,
        "by_target": by_target,
        "judgment_snapshot": judgment_snapshot or {},
    }
    try:
        resp = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
        return _json(resp)
    except requests.RequestException as exc:
        raise WorkflowApiError(str(exc)) from exc


def resume_run(api_url: str, *, run_id: str) -> dict:
    url = f"{_root(api_url)}/workflow/runs/{str(run_id)}/resume"
    try:
        resp = requests.post(url, timeout=DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
        return _json(resp)
    except requests.RequestException as exc:
        raise WorkflowApiError(str(exc)) from exc


def get_run_status(api_url: str, *, run_id: str) -> dict:
    url = f"{_root(api_url)}/workflow/runs/{str(run_id)}/status"
    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
        return _json(resp)
    except requests.RequestException as exc:
        raise WorkflowApiError(str(exc)) from exc


def get_latest_run(api_url: str, *, claim_id: str, reviewer_uid: str) -> dict | None:
    url = f"{_root(api_url)}/workflow/claimspans/{str(claim_id)}/runs/latest"
    try:
        resp = requests.get(
            url,
            params={"reviewer_uid": str(reviewer_uid or "default")},
            timeout=DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = _json(resp)
        rid = (data or {}).get("run_id")
        return data if rid else None
    except requests.RequestException as exc:
        raise WorkflowApiError(str(exc)) from exc


def cancel_target(api_url: str, *, run_id: str, target_id: str) -> dict:
    url = (
        f"{_root(api_url)}/workflow/runs/{str(run_id)}/targets/{str(target_id)}/cancel"
    )
    try:
        resp = requests.post(url, timeout=DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
        return _json(resp)
    except requests.RequestException as exc:
        raise WorkflowApiError(str(exc)) from exc


__all__ = [
    "WorkflowApiError",
    "start_claimspan_run",
    "finalize_assessment",
    "resume_run",
    "get_run_status",
    "get_latest_run",
    "cancel_target",
]
