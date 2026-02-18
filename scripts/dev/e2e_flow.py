from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


API_URL = os.environ.get("API_URL", "http://app-api:8000").rstrip("/")


def _req(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{API_URL}{path}"
    resp = requests.request(method, url, timeout=kwargs.pop("timeout", 60), **kwargs)
    return resp


def _json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()  # type: ignore[no-any-return]
    except Exception:
        raise RuntimeError(f"Non-JSON response {resp.status_code}: {resp.text[:500]}")


def wipe() -> None:
    resp = _req(
        "POST",
        "/dev/wipe",
        json={"confirm": "WIPE"},
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"wipe failed: {resp.status_code} {resp.text[:200]}")


def wait_for_api(*, timeout_s: int = 90) -> None:
    t0 = time.time()
    last_err: Optional[str] = None
    while True:
        try:
            resp = _req("GET", "/docs", timeout=5)
            if resp.status_code == 200:
                return
            last_err = f"http {resp.status_code}"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"API not ready at {API_URL} (last={last_err})")
        time.sleep(1)


def upload_pdf(pdf_path: Path) -> str:
    with pdf_path.open("rb") as f:
        resp = _req(
            "POST",
            "/ingest?auto_process=false",
            files={"file": (pdf_path.name, f, "application/pdf")},
            timeout=120,
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"upload failed: {resp.status_code} {resp.text[:200]}")
    data = _json(resp)
    doc_id = ((data.get("document") or {}).get("id") or "").strip()
    if not doc_id:
        raise RuntimeError("upload response missing document.id")
    return doc_id


def trigger_extract(doc_id: str) -> None:
    resp = _req("POST", f"/ingest/{doc_id}/extract", timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"extract trigger failed: {resp.status_code} {resp.text[:200]}"
        )


def force_fallback(doc_id: str) -> None:
    resp = _req("POST", f"/ingest/{doc_id}/fallback-extract", timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"fallback trigger failed: {resp.status_code} {resp.text[:200]}"
        )


def poll_attempt(doc_id: str, *, timeout_s: int = 240) -> Tuple[str, Dict[str, Any]]:
    t0 = time.time()
    last: Dict[str, Any] = {}
    while True:
        resp = _req("GET", f"/ingest/{doc_id}/spine", timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"spine poll failed: {resp.status_code} {resp.text[:200]}"
            )
        last = _json(resp)
        attempt = last.get("attempt") or {}
        state = str(attempt.get("state") or "")
        if state in {"succeeded", "partial", "failed", "cancelled"}:
            return state, last
        if time.time() - t0 > timeout_s:
            raise RuntimeError(
                f"timeout waiting for attempt terminal state (last={state})"
            )
        time.sleep(2)


def poll_until_artifact(
    doc_id: str, *, artifact_type: str, timeout_s: int = 240
) -> Dict[str, Any]:
    t0 = time.time()
    last: Dict[str, Any] = {}
    want = str(artifact_type or "").strip()
    if not want:
        raise ValueError("artifact_type is required")
    while True:
        resp = _req("GET", f"/ingest/{doc_id}/spine", timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"spine poll failed: {resp.status_code} {resp.text[:200]}"
            )
        last = _json(resp)
        arts = last.get("artifacts") or []
        for a in arts:
            if str((a or {}).get("artifact_type") or "") == want:
                return last
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"timeout waiting for artifact {want}")
        time.sleep(2)


def get_body(doc_id: str) -> Dict[str, Any]:
    resp = _req("GET", f"/ingest/{doc_id}/body", timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"body failed: {resp.status_code} {resp.text[:200]}")
    return _json(resp)


def confirm_one_claim(*, doc_id: str, body: Dict[str, Any]) -> None:
    paras = body.get("paragraphs") or []
    if not paras:
        raise RuntimeError("no paragraphs returned")
    sent = None
    for p in paras:
        sents = (p or {}).get("sentences") or []
        if sents:
            sent = sents[0]
            break
    if not sent:
        raise RuntimeError("no sentences returned")
    sentence_id = str(sent.get("sentence_id") or "").strip()
    segs = sent.get("segments") or []
    sentence_text = " ".join([str(x.get("text") or "") for x in segs]).strip()
    if not sentence_id or not sentence_text:
        raise RuntimeError("missing sentence_id or sentence_text")

    payload = {
        "document_id": doc_id,
        "sentence_id": sentence_id,
        "sentence_text": sentence_text,
        "citation_index": 0,
        "target_id": None,
        "segmentation_model": "e2e",
        "reviewer_uid": "default",
        "cited_work_id": None,
        "citation_anchor": None,
        "confirmed_claims": [
            {
                "claim_index": 1,
                "parsed_text": sentence_text,
                "original_text": sentence_text,
                "confidence": 0.5,
            }
        ],
    }
    resp = _req("POST", "/claims/confirm", json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"claims/confirm failed: {resp.status_code} {resp.text[:200]}"
        )
    data = _json(resp)
    if int(data.get("inserted") or 0) < 1:
        raise RuntimeError(f"expected inserted>=1, got: {data}")


def check_graph_has_claims() -> None:
    resp = _req("GET", "/graph/claim-nodes", timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"graph claim-nodes failed: {resp.status_code} {resp.text[:200]}"
        )
    data = _json(resp)
    nodes = data.get("nodes") or []
    if not nodes:
        raise RuntimeError("expected at least one claim node")


def main() -> int:
    repo = Path(".").resolve()
    sample = repo / "fixtures" / "text-1.pdf"
    scanned = repo / "fixtures" / "scanned-1.pdf"
    if not sample.exists():
        print(f"missing fixture: {sample}", file=sys.stderr)
        return 2
    if not scanned.exists():
        print(f"missing fixture: {scanned}", file=sys.stderr)
        return 2

    print(f"API_URL={API_URL}")
    wait_for_api(timeout_s=120)
    wipe()

    # Ingest + extract normal-ish doc.
    doc1 = upload_pdf(sample)
    trigger_extract(doc1)
    state1, spine1 = poll_attempt(doc1, timeout_s=240)
    print(f"doc1={doc1} state={state1}")
    body1 = get_body(doc1)
    pcount1 = len(body1.get("paragraphs") or [])
    print(f"doc1 paragraphs={pcount1}")
    if pcount1 == 0:
        # Some PDFs may extract to an empty body via TEI; force fallback to
        # guarantee a usable body substrate.
        force_fallback(doc1)
        spine1b = poll_until_artifact(
            doc1, artifact_type="fallback.body.txt", timeout_s=240
        )
        state1b = str(((spine1b.get("attempt") or {}).get("state")) or "")
        body1 = get_body(doc1)
        pcount1 = len(body1.get("paragraphs") or [])
        print(f"doc1 fallback state={state1b} paragraphs={pcount1}")
    confirm_one_claim(doc_id=doc1, body=body1)
    check_graph_has_claims()

    # Ingest scanned fixture and force fallback OCR.
    wipe()
    doc2 = upload_pdf(scanned)
    force_fallback(doc2)
    spine2 = poll_until_artifact(doc2, artifact_type="fallback.body.txt", timeout_s=420)
    state2 = str(((spine2.get("attempt") or {}).get("state")) or "")
    q2 = (spine2.get("attempt") or {}).get("quality_json") or {}
    print(f"doc2={doc2} state={state2} quality={q2}")
    body2 = get_body(doc2)
    print(f"doc2 paragraphs={len(body2.get('paragraphs') or [])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
