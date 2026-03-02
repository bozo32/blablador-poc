#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

import requests


def _headers(project_id: str, user_id: str) -> dict[str, str]:
    return {
        "X-Project-Id": project_id,
        "X-User-Id": user_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke Stage 1+2 projection contract")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--project-id", default="stage12-smoke")
    parser.add_argument("--user-id", default="stage12-smoke-user")
    args = parser.parse_args()

    api = str(args.api_url).rstrip("/")
    fixture = pathlib.Path(args.fixture)
    if not fixture.exists():
        raise SystemExit(f"fixture not found: {fixture}")

    headers = _headers(project_id=args.project_id, user_id=args.user_id)

    with fixture.open("rb") as handle:
        ingest_resp = requests.post(
            f"{api}/ingest",
            headers=headers,
            files={"file": (fixture.name, handle, "application/pdf")},
            timeout=90,
        )
    ingest_resp.raise_for_status()

    ledger_resp = requests.get(
        f"{api}/ledger",
        headers={"X-Project-Id": args.project_id},
        timeout=30,
    )
    ledger_resp.raise_for_status()
    payload = ledger_resp.json() or {}
    rows = payload.get("rows") or []

    canonical_keys = {
        "canonical_extraction_status",
        "canonical_body_extraction_status",
        "canonical_resolution_status",
    }
    missing = [
        idx
        for idx, row in enumerate(rows)
        if isinstance(row, dict) and not canonical_keys.issubset(set(row.keys()))
    ]
    if missing:
        raise SystemExit(f"rows missing canonical fields at indices: {missing}")

    print("PASS smoke_stage12_projection")
    return 0


if __name__ == "__main__":
    sys.exit(main())
