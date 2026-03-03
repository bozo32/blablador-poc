#!/usr/bin/env bash
set -euo pipefail

curl_json() {
  curl -fsS --retry 20 --retry-delay 1 --retry-connrefused --max-time 30 "$@"
}

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
  fi
fi
DC=("${DOCKER[@]}" compose)

"${DC[@]}" up -d postgres minio minio-init grobid fallback-worker

# Ensure the running API container reflects the current working tree without
# requiring a full image rebuild (use bind mounts to overlay code).
"${DC[@]}" stop app-api >/dev/null 2>&1 || true
"${DC[@]}" rm -f app-api >/dev/null 2>&1 || true

# Clean up any prior `docker compose run -d app-api` containers that may still
# be holding the 8000 port.
old_ids=$(${DOCKER[*]} ps -aq --filter "name=blablador-poc-app-api-run" 2>/dev/null || true)
if [ -n "${old_ids}" ]; then
  ${DOCKER[*]} rm -f ${old_ids} >/dev/null 2>&1 || true
fi

"${DC[@]}" run -d --service-ports \
  -v "${PWD}/backend:/app/backend" \
  -v "${PWD}/frontend:/app/frontend" \
  -v "${PWD}/application.py:/app/application.py" \
  app-api >/dev/null

API_URL="http://127.0.0.1:8000"

for i in {1..120}; do
  code=$(curl -s --max-time 2 -o /dev/null -w "%{http_code}" "${API_URL}/docs" || true)
  if [ "$code" = "200" ]; then
    break
  fi
  sleep 1
done

if [ "${code:-}" != "200" ]; then
  echo "ERROR: API did not become ready at ${API_URL} (last code=${code:-})" >&2
  exit 1
fi

curl_json -X POST "${API_URL}/dev/wipe" \
  -H "Content-Type: application/json" \
  -d '{"confirm":"WIPE"}' >/dev/null

export API_URL

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import requests

from scripts.dev import e2e_flow


API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
e2e_flow.API_URL = API_URL
PROJECT_ID = os.environ.get("P1", "default").strip() or "default"
USER_ID = os.environ.get("USER_ID", "default").strip() or "default"


def _scope_headers(*, reviewer_uid: str = "default") -> dict[str, str]:
    reviewer = str(reviewer_uid or "").strip() or USER_ID
    return {
        "X-Project-Id": PROJECT_ID,
        "X-User-Id": USER_ID,
        "X-Reviewer-Uid": reviewer,
    }


def req(method: str, path: str, **kwargs) -> dict:
    url = f"{API_URL}{path}"
    headers = kwargs.pop("headers", None)
    merged_headers = dict(_scope_headers())
    if isinstance(headers, dict):
        merged_headers.update(headers)
    resp = requests.request(
        method,
        url,
        headers=merged_headers,
        timeout=kwargs.pop("timeout", 60),
        **kwargs,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"{method} {path} failed: {resp.status_code} {resp.text[:200]}")
    return resp.json() if resp.text else {}


def ensure_project_membership() -> None:
    resp = requests.post(
        f"{API_URL}/projects",
        headers={"X-User-Id": USER_ID},
        json={"project_id": PROJECT_ID},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"failed ensuring project membership: {resp.status_code} {resp.text[:200]}"
        )


repo = Path(".").resolve()
fixtures = [
    repo / "fixtures" / "citations-2.pdf",
    repo / "fixtures" / "sample.pdf",
    repo / "fixtures" / "text-1.pdf",
]
fixtures = [p for p in fixtures if p.exists()]
if not fixtures:
    raise RuntimeError(
        "missing fixtures: fixtures/citations-2.pdf (preferred) or fixtures/sample.pdf"
    )

ensure_project_membership()


def find_first_citation(body: dict) -> tuple[str, str, int, str]:
    for para in body.get("paragraphs") or []:
        for sent in (para or {}).get("sentences") or []:
            sentence_id = str(sent.get("sentence_id") or "").strip() or None
            segments = sent.get("segments") or []
            bits = []
            found = None
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                t = str(seg.get("type") or "").strip()
                if t == "text":
                    bits.append(str(seg.get("text") or "").strip())
                elif t == "citation":
                    bits.append(str(seg.get("callout") or "").strip())
                    try:
                        citation_index = int(seg.get("citation_index"))
                    except Exception:
                        continue
                    target_id = str(seg.get("target_id") or "").strip() or None
                    if sentence_id and target_id is not None:
                        found = (sentence_id, citation_index, target_id)
            sentence_text = " ".join(bit for bit in bits if bit).strip() or None
            if found and sentence_text:
                sid, idx, tid = found
                return sid, sentence_text, int(idx), str(tid)
    raise RuntimeError(
        "fixture did not yield any citation segments with citation_index + target_id"
    )


doc_id = None
seed = None
for pdf in fixtures:
    did = e2e_flow.upload_pdf(pdf)
    e2e_flow.trigger_extract(did)
    state, _spine = e2e_flow.poll_attempt(did, timeout_s=240)
    if state not in {"succeeded", "partial"}:
        continue
    body = e2e_flow.get_body(did)
    if len(body.get("paragraphs") or []) == 0:
        e2e_flow.force_fallback(did)
        _ = e2e_flow.poll_until_artifact(
            did, artifact_type="fallback.body.txt", timeout_s=240
        )
        body = e2e_flow.get_body(did)
    try:
        sentence_id, sentence_text, citation_index, target_id = find_first_citation(body)
    except Exception:
        continue
    doc_id = did
    seed = {
        "sentence_id": sentence_id,
        "sentence_text": sentence_text,
        "citation_index": int(citation_index),
        "target_id": str(target_id),
    }
    break

if not doc_id or not seed:
    raise RuntimeError(
        "no fixture produced an extracted body with at least one citation segment"
    )

payload = {
    "document_id": doc_id,
    "sentence_id": seed["sentence_id"],
    "sentence_text": seed["sentence_text"],
    "citation_index": int(seed["citation_index"]),
    "target_id": seed["target_id"],
    "segmentation_model": "e2e",
    "reviewer_uid": "default",
    "confirmed_claims": [
        {
            "claim_index": 1,
            "parsed_text": seed["sentence_text"] or "seed",
            "original_text": seed["sentence_text"] or "seed",
            "confidence": 0.5,
        }
    ],
}
_ = req("POST", "/claims/confirm", json=payload)

span = req(
    "GET",
    "/spans/lookup-citation-window",
    params={
        "ingest_id": doc_id,
        "citation_index": int(seed["citation_index"]),
        "target_id": seed["target_id"],
    },
)
span_id = str(span.get("span_id") or "").strip()
if not span_id:
    raise RuntimeError("lookup-citation-window did not return span_id")

graph = req(
    "GET",
    "/nav/graph",
    params={
        "reviewer_uid": "default",
        "focus_type": "citespan",
        "focus_id": span_id,
        "show_claimspans": "true",
    },
)
elements = graph.get("elements") or []
if not isinstance(elements, list) or not elements:
    raise RuntimeError("/nav/graph returned empty elements")

for el in elements:
    if not isinstance(el, dict) or not isinstance(el.get("data"), dict):
        raise RuntimeError("/nav/graph elements must be dicts with data")
    data = el["data"]
    if not str(data.get("id") or "").strip():
        raise RuntimeError("element missing data.id")
    if not str(data.get("selectable_type") or "").strip():
        raise RuntimeError("element missing data.selectable_type")
    if not str(data.get("state") or "").strip():
        raise RuntimeError("element missing data.state")


work_ids = []
for el in elements:
    data = el.get("data") or {}
    if data.get("selectable_type") != "work":
        continue
    wid = str(data.get("work_id") or "").strip()
    if wid and wid != doc_id:
        work_ids.append(wid)

claimspan_nodes = [
    el
    for el in elements
    if isinstance(el, dict)
    and isinstance(el.get("data"), dict)
    and (el.get("data") or {}).get("selectable_type") == "claimspan"
]
if not claimspan_nodes:
    raise RuntimeError("expected claimspan nodes when show_claimspans=true")

if not work_ids:
    raise RuntimeError("expected at least one cited work node")

work_id = work_ids[0]
contexts = req(
    "GET",
    f"/nav/works/{work_id}/contexts",
    params={"reviewer_uid": "default"},
)
rows = contexts.get("contexts") or []
if not isinstance(rows, list) or not rows:
    raise RuntimeError("expected >=1 /nav/works contexts")
row0 = rows[0] or {}
citing_doc_id = str(row0.get("citing_doc_id") or "").strip()
reference_id = str(row0.get("reference_id") or "").strip()
if not citing_doc_id or not reference_id:
    raise RuntimeError("contexts row missing citing_doc_id/reference_id")

dossier = req(
    "GET",
    f"/references/{citing_doc_id}/{reference_id}/retrieval",
)
canonical = str(dossier.get("canonical_citation") or "").strip()
if not canonical:
    raise RuntimeError("retrieval dossier missing canonical_citation")
has_any = bool(
    str(dossier.get("manual_instructions") or "").strip()
    or str(dossier.get("primary_url") or "").strip()
    or (
        isinstance(dossier.get("sources"), list)
        and dossier.get("sources")
        and str((dossier.get("sources")[0] or {}).get("url") or "").strip()
    )
)
if not has_any:
    raise RuntimeError("retrieval dossier missing instructions/urls")

print("OK: verify_10_03_graph_nav")
PY
