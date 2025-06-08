import os
import xml.etree.ElementTree as ET
from pathlib import Path

import requests


def tei_to_chunks(xml_path: Path) -> list[dict]:
    """Parse a TEI XML file and split into chunks (e.g. paragraphs)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    chunks = []
    for p in root.findall(".//tei:p", ns):
        text = "".join(p.itertext()).strip()
        if text:
            chunks.append({"text": text, "meta": {"source": str(xml_path)}})
    return chunks


def tei_and_csv_to_documents(folder: Path, csv_path: str) -> list[dict]:
    """Combine TEI XML chunks and CSV citing sentences into one unified document list."""
    from .utils import read_csv

    docs: list[dict] = []
    # 1) TEI XML chunks
    for xml_file in Path(folder).glob("*.xml"):
        docs.extend(tei_to_chunks(xml_file))
    # 2) CSV citing sentences
    df = read_csv(csv_path)
    for idx, row in df.iterrows():
        meta = {"type": "citing", "csv_row": int(idx)}
        if "citing_title" in row:
            meta["citing_title"] = row["citing_title"]
        if "citing_id" in row:
            meta["citing_id"] = row["citing_id"]
        docs.append({"text": row["citing_sentence"], "meta": meta})
    return docs


class BlabladorClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def completion(self, prompt, model=None, temperature=0.0, max_tokens=512):
        """
        Call the Blablador `/v1/completions` endpoint.
        """
        # if no model explicitly given, defer to environment or legacy
        model = model or os.getenv("LLM_MODEL") or "gpt-3.5-turbo"

        base = self.base_url.rstrip("/")
        if not base.startswith(("http://", "https://")):
            base = "https://" + base
        url = f"{base}/v1/completions"

        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        from requests import HTTPError

        try:
            response.raise_for_status()
        except HTTPError:
            # Log the full error payload for debugging
            print(f"[Blablador API error {response.status_code}]: {response.text}")
            raise
        return response.json()["choices"][0]["text"]


def embed_documents(docs, model, api_key, base_url):
    """
    Embed a list of document dicts by batching inputs to avoid service errors.
    """

    # Prepare the full list of texts
    texts = [doc["text"] for doc in docs]
    url = f"{base_url.rstrip('/')}/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"}

    embeddings = []
    batch_size = 50  # adjust as needed

    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        payload = {"model": model, "input": batch}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            print(f"[Embedding batch error] start={start}: {e}")
            # append zero-vectors or skip -- here we raise
            raise
        data = resp.json().get("data", [])
        # Extract embeddings in order
        embeddings.extend([item["embedding"] for item in data])

    return embeddings


def embeddings(texts: list[str], model: str, api_key: str, base_url: str):
    docs = [{"text": t} for t in texts]
    return embed_documents(docs, model=model, api_key=api_key, base_url=base_url)


def completion(
    prompt,
    model="gpt-3.5-turbo",
    temperature=0.0,
    max_tokens=512,
    api_key=None,
    base_url=None,
):
    """
    Back-compat wrapper so modules can import and call
        from backend.bl_client import completion
    """
    client = BlabladorClient(
        api_key=api_key,
        base_url=base_url,
    )
    return client.completion(
        prompt, model=model, temperature=temperature, max_tokens=max_tokens
    )
