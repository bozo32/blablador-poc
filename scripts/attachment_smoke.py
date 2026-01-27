#!/usr/bin/env python3
"""Upload a PDF attachment and monitor backend status transitions."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload a PDF via the attachment API and watch status updates",
    )
    parser.add_argument("--doc", required=True, help="Document/ingestion identifier")
    parser.add_argument("--claim", required=True, help="Claim identifier")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the PDF to upload (shared workspace path)",
    )
    parser.add_argument(
        "--backend",
        default="http://localhost:8000",
        help="Backend base URL for HTTP mode",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Maximum seconds to wait for READY/ERROR",
    )
    parser.add_argument(
        "--local-app",
        action="store_true",
        help=(
            "Run against in-process attachment pipeline with stubbed parsing/embeddings"
        ),
    )
    return parser


class _Transport:
    def __init__(self, post: Callable[..., dict], get: Callable[..., dict]):
        self._post = post
        self._get = get

    def post(self, path: str, **kwargs) -> dict:
        return self._post(path, **kwargs)

    def get(self, path: str, **kwargs) -> dict:
        return self._get(path, **kwargs)


def _http_transport(base_url: str) -> _Transport:
    base = base_url.rstrip("/")

    def _post(path: str, **kwargs) -> dict:
        resp = requests.post(f"{base}{path}", timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _get(path: str, **kwargs) -> dict:
        resp = requests.get(f"{base}{path}", timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()

    return _Transport(_post, _get)


def _local_transport() -> _Transport:
    from backend import attachment_pipeline, attachment_store
    from backend.settings import settings

    tmp_dir = Path(tempfile.mkdtemp(prefix="attachment-smoke-"))
    settings.ATTACHMENT_DIR = tmp_dir

    def fake_extract(path: str | Path) -> str:
        data = Path(path).read_text(encoding="utf-8", errors="ignore")
        return (
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            f"<text><body><p><s xml:id='s1'>{data}</s></p></body></text></TEI>"
        )

    def fake_embed(texts, **_) -> list[list[float]]:
        return [[float(idx + 1)] for idx, _ in enumerate(texts)]

    attachment_pipeline.grobid_client.extract_tei = (  # type: ignore[attr-defined]
        fake_extract
    )
    attachment_pipeline.utils.embed = fake_embed  # type: ignore[assignment]

    def _post(path: str, **kwargs) -> dict:
        payload = kwargs.get("json") or {}
        if path.endswith("/attachments") and path.startswith("/claims/"):
            claim_id = path.split("/")[2]
            local_path = payload.get("local_path")
            if not local_path:
                raise RuntimeError("local_path is required for attachment upload")
            record = attachment_store.create_attachment(
                claim_id=claim_id,
                doc_id=payload.get("doc_id"),
                local_path=local_path,
                filename=payload.get("filename"),
                size_bytes=payload.get("size_bytes"),
                reference_hint=payload.get("reference_hint"),
            )
            attachment_pipeline.process_attachment(record["id"])
            return {"attachment": attachment_store.public_status(record["id"])}
        raise RuntimeError(f"Unsupported local path: {path}")

    def _get(path: str, **_) -> dict:
        if path.startswith("/attachments/"):
            attachment_id = path.split("/")[2]
            return {"attachment": attachment_store.public_status(attachment_id)}
        raise RuntimeError(f"Unsupported local path: {path}")

    return _Transport(_post, _get)


def _print_event(message: str, payload: Optional[Dict[str, Any]] = None) -> None:
    print(message)
    if payload:
        print(json.dumps(payload, indent=2))


def _poll_status(
    transport: _Transport, attachment_id: str, interval: float, timeout: float
) -> dict:
    start = time.time()
    last_status = None
    while time.time() - start <= timeout:
        payload = transport.get(f"/attachments/{attachment_id}")
        attachment = payload.get("attachment") or {}
        status = attachment.get("status")
        if status != last_status:
            _print_event(f"Status → {status}")
            last_status = status
        if status in {"ready", "error"}:
            return attachment
        time.sleep(interval)
    raise TimeoutError(f"Attachment did not finish within {timeout} seconds")


def main() -> int:
    args = _build_parser().parse_args()
    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        return 1

    transport = _local_transport() if args.local_app else _http_transport(args.backend)
    upload_payload = {
        "doc_id": args.doc,
        "filename": file_path.name,
        "local_path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "reference_hint": {},
    }

    _print_event("Uploading attachment", upload_payload)
    response = transport.post(
        f"/claims/{args.claim}/attachments",
        json=upload_payload,
    )
    attachment = response.get("attachment") or {}
    attachment_id = attachment.get("id")
    if not attachment_id:
        print("Backend did not return attachment id", file=sys.stderr)
        return 2
    _print_event("Attachment accepted", attachment)

    try:
        final_state = _poll_status(
            transport,
            attachment_id=attachment_id,
            interval=args.poll_interval,
            timeout=args.timeout,
        )
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    timeline = final_state.get("history") or []
    _print_event("Final timeline", {"events": timeline})
    print(f"Attachment ended in status: {final_state.get('status')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
