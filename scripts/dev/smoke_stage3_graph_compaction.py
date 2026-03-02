from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def _post(api_url: str, path: str, payload: dict) -> dict:
    resp = requests.post(
        f"{api_url.rstrip('/')}{path}",
        json=payload,
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"POST {path} failed: {resp.status_code} {resp.text[:200]}")
    return resp.json() if resp.text else {}


def _upload_fixture(api_url: str, fixture: Path) -> str:
    with fixture.open("rb") as fh:
        resp = requests.post(
            f"{api_url.rstrip('/')}/ingest",
            params={"auto_process": "false"},
            files={"file": (fixture.name, fh, "application/pdf")},
            headers={"X-Project-Id": "default", "X-User-Id": "default"},
            timeout=120,
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"/ingest failed: {resp.status_code} {resp.text[:200]}")
    payload = resp.json() if resp.text else {}
    doc = payload.get("document") if isinstance(payload, dict) else {}
    doc_id = str((doc or {}).get("id") or "").strip()
    if not doc_id:
        raise RuntimeError("/ingest succeeded but document.id is missing")
    return doc_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3 graph compaction smoke test")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--fixture", required=True)
    args = parser.parse_args()

    api_url = str(args.api_url).rstrip("/")
    fixture = Path(args.fixture).resolve()
    if not fixture.exists():
        raise SystemExit(f"Fixture not found: {fixture}")

    uploaded_doc_id = _upload_fixture(api_url, fixture)

    dry_run = _post(api_url, "/maintenance/graph/compact/dry-run", {"project_id": "default"})
    apply_run = _post(api_url, "/maintenance/graph/compact/apply", {"project_id": "default"})
    rollback = _post(
        api_url,
        "/maintenance/graph/compact/rollback",
        {"project_id": "default", "run_id": apply_run.get("run_id")},
    )

    if str(dry_run.get("mode")) != "dry-run":
        raise RuntimeError("dry-run mode mismatch")
    if str(apply_run.get("mode")) != "apply":
        raise RuntimeError("apply mode mismatch")
    if str(rollback.get("mode")) != "rollback":
        raise RuntimeError("rollback mode mismatch")

    print(
        json.dumps(
            {
                "ok": True,
                "uploaded_doc_id": uploaded_doc_id,
                "dry_run": dry_run,
                "apply": apply_run,
                "rollback": rollback,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
