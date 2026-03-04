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


def _project_headers(project_id: Optional[str]) -> Optional[dict[str, str]]:
    pid = str(project_id or "").strip()
    if not pid:
        return None
    return {"X-Project-Id": pid}


def start_claimspan_run(
    api_url: str,
    *,
    claim_id: str,
    reviewer_uid: str,
    citing_doc_id: str,
    project_id: Optional[str] = None,
) -> dict:
    url = f"{_root(api_url)}/workflow/claimspans/{str(claim_id)}/runs"
    payload = {
        "reviewer_uid": str(reviewer_uid or "default"),
        "citing_doc_id": str(citing_doc_id or "").strip(),
    }
    headers: dict[str, str] = {}
    pid = str(project_id or "").strip()
    if pid:
        headers["X-Project-Id"] = pid
    try:
        resp = requests.post(
            url,
            json=payload,
            headers=headers or None,
            timeout=DEFAULT_TIMEOUT_S,
        )
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
    project_id: Optional[str] = None,
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
        resp = requests.post(
            url,
            json=payload,
            headers=_project_headers(project_id),
            timeout=DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
        return _json(resp)
    except requests.RequestException as exc:
        raise WorkflowApiError(str(exc)) from exc


def resume_run(api_url: str, *, run_id: str, project_id: Optional[str] = None) -> dict:
    url = f"{_root(api_url)}/workflow/runs/{str(run_id)}/resume"
    try:
        resp = requests.post(
            url,
            headers=_project_headers(project_id),
            timeout=DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
        return _json(resp)
    except requests.RequestException as exc:
        raise WorkflowApiError(str(exc)) from exc


def get_run_status(api_url: str, *, run_id: str, project_id: Optional[str] = None) -> dict:
    url = f"{_root(api_url)}/workflow/runs/{str(run_id)}/status"
    try:
        resp = requests.get(
            url,
            headers=_project_headers(project_id),
            timeout=DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
        return _json(resp)
    except requests.RequestException as exc:
        raise WorkflowApiError(str(exc)) from exc


def get_latest_run(
    api_url: str,
    *,
    claim_id: str,
    reviewer_uid: str,
    project_id: Optional[str] = None,
) -> dict | None:
    url = f"{_root(api_url)}/workflow/claimspans/{str(claim_id)}/runs/latest"
    try:
        resp = requests.get(
            url,
            params={"reviewer_uid": str(reviewer_uid or "default")},
            headers=_project_headers(project_id),
            timeout=DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = _json(resp)
        rid = (data or {}).get("run_id")
        return data if rid else None
    except requests.RequestException as exc:
        raise WorkflowApiError(str(exc)) from exc


def cancel_target(
    api_url: str,
    *,
    run_id: str,
    target_id: str,
    project_id: Optional[str] = None,
) -> dict:
    url = (
        f"{_root(api_url)}/workflow/runs/{str(run_id)}/targets/{str(target_id)}/cancel"
    )
    try:
        resp = requests.post(
            url,
            headers=_project_headers(project_id),
            timeout=DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
        return _json(resp)
    except requests.RequestException as exc:
        raise WorkflowApiError(str(exc)) from exc


def open_events_stream(
    api_url: str,
    *,
    run_id: str,
    after_event_id: int = 0,
    heartbeat_ms: int = 10000,
    project_id: Optional[str] = None,
) -> requests.Response:
    url = f"{_root(api_url)}/workflow/runs/{str(run_id)}/events"
    params = {
        "after_event_id": int(after_event_id),
        "heartbeat_ms": int(heartbeat_ms),
    }
    try:
        resp = requests.get(
            url,
            params=params,
            headers=_project_headers(project_id),
            timeout=DEFAULT_TIMEOUT_S,
            stream=True,
        )
        resp.raise_for_status()
        return resp
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
    "open_events_stream",
]
