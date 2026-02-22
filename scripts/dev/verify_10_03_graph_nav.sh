#!/usr/bin/env bash
set -euo pipefail

curl_json() {
  curl -fsS --retry 20 --retry-delay 1 --retry-connrefused --max-time 30 "$@"
}

docker compose up -d postgres minio minio-init grobid fallback-worker

# Ensure the running API container reflects the current working tree without
# requiring a full image rebuild (use bind mounts to overlay code).
docker compose stop app-api >/dev/null 2>&1 || true
docker compose rm -f app-api >/dev/null 2>&1 || true

# Clean up any prior `docker compose run -d app-api` containers that may still
# be holding the 8000 port.
old_ids=$(docker ps -aq --filter "name=blablador-poc-app-api-run" || true)
if [ -n "${old_ids}" ]; then
  docker rm -f ${old_ids} >/dev/null 2>&1 || true
fi

docker compose run -d --service-ports \
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

python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import requests

from scripts.dev import e2e_flow


API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
e2e_flow.API_URL = API_URL


def req(method: str, path: str, **kwargs) -> dict:
    url = f"{API_URL}{path}"
    resp = requests.request(method, url, timeout=kwargs.pop("timeout", 60), **kwargs)
    if resp.status_code >= 400:
        raise RuntimeError(f"{method} {path} failed: {resp.status_code} {resp.text[:200]}")
    return resp.json() if resp.text else {}


repo = Path(".").resolve()
fixtures = [repo / "fixtures" / "text-1.pdf", repo / "fixtures" / "sample.pdf"]
fixtures = [p for p in fixtures if p.exists()]
if not fixtures:
    raise RuntimeError("missing fixtures: fixtures/text-1.pdf and fixtures/sample.pdf")

doc_id = None
for pdf in fixtures:
    did = e2e_flow.upload_pdf(pdf)
    e2e_flow.trigger_extract(did)
    state, _spine = e2e_flow.poll_attempt(did, timeout_s=240)
    if state in {"succeeded", "partial"}:
        doc_id = did
        break

if not doc_id:
    raise RuntimeError("no fixture produced a successful extraction attempt")

ref_id = "ref-1"

body = e2e_flow.get_body(doc_id)
if len(body.get("paragraphs") or []) == 0:
    e2e_flow.force_fallback(doc_id)
    _ = e2e_flow.poll_until_artifact(doc_id, artifact_type="fallback.body.txt", timeout_s=240)
    body = e2e_flow.get_body(doc_id)

sentence_id = None
sentence_text = None
for para in body.get("paragraphs") or []:
    for sent in (para or {}).get("sentences") or []:
        sentence_id = str(sent.get("sentence_id") or "").strip() or None
        segments = sent.get("segments") or []
        bits = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            t = str(seg.get("type") or "").strip()
            if t == "text":
                bits.append(str(seg.get("text") or "").strip())
            elif t == "citation":
                bits.append(str(seg.get("callout") or "").strip())
        sentence_text = " ".join(bit for bit in bits if bit).strip() or None
        if sentence_id and sentence_text:
            break
    if sentence_id and sentence_text:
        break
if not sentence_id or not sentence_text:
    raise RuntimeError("no sentence available for claim confirmation")

c = {
    "sentence_id": sentence_id,
    "sentence_text": sentence_text,
    "target_id": ref_id,
    "citation_index": 0,
}

payload = {
    "document_id": doc_id,
    "sentence_id": c["sentence_id"],
    "sentence_text": c["sentence_text"],
    "citation_index": int(c["citation_index"]),
    "target_id": c["target_id"],
    "segmentation_model": "e2e",
    "reviewer_uid": "default",
    "confirmed_claims": [
        {
            "claim_index": 1,
            "parsed_text": c["sentence_text"] or "seed",
            "original_text": c["sentence_text"] or "seed",
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
        "citation_index": int(c["citation_index"]),
        "target_id": c["target_id"],
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

if work_ids:
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
