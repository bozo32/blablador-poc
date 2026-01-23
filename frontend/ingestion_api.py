import requests

DEFAULT_TIMEOUT = 30


def _parse_response(response: requests.Response) -> dict | None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(response.text) from exc
    if not response.text:
        return None
    return response.json()


def upload_pdf(api_url: str, file) -> dict:
    url = f"{api_url.rstrip('/')}/ingest"
    content_type = getattr(file, "type", None) or "application/pdf"
    files = {"file": (file.name, file.getbuffer(), content_type)}
    try:
        response = requests.post(url, files=files, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ingestion API at {url}") from exc
    payload = _parse_response(response) or {}
    return payload.get("document") or {}


def list_documents(api_url: str) -> list[dict]:
    url = f"{api_url.rstrip('/')}/ingest"
    try:
        response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ingestion API at {url}") from exc
    payload = _parse_response(response) or {}
    return payload.get("documents") or []


def get_document(api_url: str, doc_id: str) -> dict:
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}"
    try:
        response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ingestion API at {url}") from exc
    payload = _parse_response(response) or {}
    return payload


def trigger_extraction(api_url: str, doc_id: str) -> dict:
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}/extract"
    try:
        response = requests.post(url, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ingestion API at {url}") from exc
    return _parse_response(response) or {}


def trigger_resolution(api_url: str, doc_id: str) -> dict:
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}/resolve"
    try:
        response = requests.post(url, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ingestion API at {url}") from exc
    return _parse_response(response) or {}
