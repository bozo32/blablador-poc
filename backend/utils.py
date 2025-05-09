import os
import re
from pathlib import Path
import pandas as pd
import json
from requests import HTTPError
from backend.bl_client import BlabladorClient


def clean_text(text: str) -> str:
    # remove TEI tags remnants and citation markers
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*\*\*#.*?\*\*\*", "", text)
    text = re.sub(r"\[\d+.*?\]", "", text)
    return text.strip()

def read_csv(path: Path | str) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame, ensuring correct path resolution.
    """
    path = Path(path)
    df = pd.read_csv(path)
    return df

# Embedding utility using OpenAI

def embed(texts: list[str]) -> list[list[float]]:
    from backend.bl_client import embeddings
    return embeddings(
        texts,
        model="alias-embeddings",
        api_key=os.getenv("BLABLADOR_API_KEY"),
        base_url=os.getenv("BLABLADOR_BASE")
    )

# backend/utils.py

def call_llm_justification(payload: dict, model_name: str = None) -> dict:
    prompt = (
        "Rank and justify which of the passages best supports the claim. Return a JSON object with keys \"best_id\" and \"rationale\".\n\n"
        + json.dumps(payload)
    )
    client = BlabladorClient(
        api_key=os.getenv("BLABLADOR_API_KEY"),
        base_url=os.getenv("BLABLADOR_BASE")
    )
    content = client.completion(
        prompt,
        model_name or "alias-large",
        temperature=0.0,
        max_tokens=256
    )

    try:
        result = json.loads(content)
        if not isinstance(result, dict):
            result = {"rationale": content, "best_id": None}
    except:
        result = {"rationale": content, "best_id": None}
    return result

__all__ = ['clean_text', 'read_csv', 'embed', 'call_llm_justification']
