#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class ProjectDoc:
    project_id: str
    doc_id: str


def _get(api_url: str, path: str, *, headers: dict[str, str] | None = None, params: dict | None = None) -> requests.Response:
    return requests.get(
        f"{api_url.rstrip('/')}{path}",
        headers=headers,
        params=params,
        timeout=60,
    )


def _post(api_url: str, path: str, *, headers: dict[str, str] | None = None, json_body: dict | None = None) -> requests.Response:
    return requests.post(
        f"{api_url.rstrip('/')}{path}",
        headers=headers,
        json=json_body,
        timeout=60,
    )


def _require_ok(resp: requests.Response, *, label: str) -> dict:
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"{label} failed ({resp.status_code}): {resp.text[:400]}")
    try:
        return resp.json() if resp.text else {}
    except Exception as exc:
        raise RuntimeError(f"{label} returned non-JSON ({resp.status_code})") from exc


def _validate_doc_ready(api_url: str, pair: ProjectDoc) -> None:
    resp = _get(
        api_url,
        f"/ingest/{pair.doc_id}",
        headers={"X-Project-Id": pair.project_id},
    )
    data = _require_ok(resp, label=f"get ingest {pair.project_id}:{pair.doc_id}")
    status = str(((data.get("extraction") or {}).get("status") or "")).strip()
    if status != "complete":
        raise RuntimeError(
            "document extraction must be complete before workflow smoke: "
            f"project={pair.project_id} doc={pair.doc_id} status={status or 'unknown'}"
        )


def _ensure_membership(api_url: str, *, user_id: str, project_id: str) -> None:
    resp = _post(
        api_url,
        "/projects",
        headers={"X-User-Id": user_id},
        json_body={"project_id": project_id, "name": project_id},
    )
    if resp.status_code not in (200, 409):
        raise RuntimeError(
            f"ensure membership failed for {project_id} ({resp.status_code}): {resp.text[:400]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test workflow scope behavior across two projects using the same claim_id."
        )
    )
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--user", required=True, help="Reviewer/user id")
    parser.add_argument("--project-a", required=True, help="First project id")
    parser.add_argument("--doc-a", required=True, help="Citing doc id for first project")
    parser.add_argument("--project-b", required=True, help="Second project id")
    parser.add_argument("--doc-b", required=True, help="Citing doc id for second project")
    parser.add_argument(
        "--claim-id",
        default="cite:shared-doc:0:scope-smoke:seg-1",
        help="Shared claim id used in both projects",
    )
    args = parser.parse_args()

    api_url = str(args.api or "").strip()
    user_id = str(args.user or "").strip()
    claim_id = str(args.claim_id or "").strip()
    a = ProjectDoc(project_id=str(args.project_a).strip(), doc_id=str(args.doc_a).strip())
    b = ProjectDoc(project_id=str(args.project_b).strip(), doc_id=str(args.doc_b).strip())
    if not api_url or not user_id or not claim_id:
        raise RuntimeError("api, user, and claim-id are required")

    _ensure_membership(api_url, user_id=user_id, project_id=a.project_id)
    _ensure_membership(api_url, user_id=user_id, project_id=b.project_id)
    _validate_doc_ready(api_url, a)
    _validate_doc_ready(api_url, b)

    runs: dict[str, str] = {}
    for pair in (a, b):
        start = _post(
            api_url,
            f"/workflow/claimspans/{claim_id}/runs",
            headers={"X-Project-Id": pair.project_id},
            json_body={"reviewer_uid": user_id, "citing_doc_id": pair.doc_id},
        )
        data = _require_ok(start, label=f"start run {pair.project_id}")
        rid = str(data.get("run_id") or "").strip()
        if not rid:
            raise RuntimeError(f"start run returned empty run_id for {pair.project_id}")
        runs[pair.project_id] = rid

    latest: dict[str, str] = {}
    for pair in (a, b):
        resp = _get(
            api_url,
            f"/workflow/claimspans/{claim_id}/runs/latest",
            headers={"X-Project-Id": pair.project_id},
            params={"reviewer_uid": user_id},
        )
        data = _require_ok(resp, label=f"latest run {pair.project_id}")
        latest[pair.project_id] = str(data.get("run_id") or "").strip()

    run_a = runs[a.project_id]
    status_no_header = _get(api_url, f"/workflow/runs/{run_a}/status")
    status_with_right_header = _get(
        api_url,
        f"/workflow/runs/{run_a}/status",
        headers={"X-Project-Id": a.project_id},
    )
    status_with_wrong_header = _get(
        api_url,
        f"/workflow/runs/{run_a}/status",
        headers={"X-Project-Id": b.project_id},
    )
    resume_with_wrong_header = _post(
        api_url,
        f"/workflow/runs/{run_a}/resume",
        headers={"X-Project-Id": b.project_id},
    )
    resume_with_right_header = _post(
        api_url,
        f"/workflow/runs/{run_a}/resume",
        headers={"X-Project-Id": a.project_id},
    )

    checks = {
        "latest_is_project_scoped": (
            latest.get(a.project_id) == runs.get(a.project_id)
            and latest.get(b.project_id) == runs.get(b.project_id)
            and runs.get(a.project_id) != runs.get(b.project_id)
        ),
        "status_read_without_header_allowed": status_no_header.status_code == 200,
        "status_read_wrong_project_blocked": status_with_wrong_header.status_code == 409,
        "resume_wrong_project_blocked": resume_with_wrong_header.status_code == 409,
        "resume_right_project_allowed": resume_with_right_header.status_code == 200,
    }
    summary = {
        "runs": runs,
        "latest": latest,
        "status_codes": {
            "status_no_header": status_no_header.status_code,
            "status_right_header": status_with_right_header.status_code,
            "status_wrong_header": status_with_wrong_header.status_code,
            "resume_wrong_header": resume_with_wrong_header.status_code,
            "resume_right_header": resume_with_right_header.status_code,
        },
        "checks": checks,
        "wrong_header_read_body": status_with_wrong_header.text[:240],
        "wrong_header_write_body": resume_with_wrong_header.text[:240],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    failed = [k for k, v in checks.items() if not v]
    if failed:
        print("FAILED_CHECKS: " + ", ".join(failed), file=sys.stderr)
        return 1
    print("ALL_CHECKS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
