# backend/utils.py

import os
import re
from pathlib import Path
import pandas as pd
import json
from requests import HTTPError
from backend.bl_client import BlabladorClient
from typing import List, Tuple, Literal  # new import


def clean_text(text: str) -> str:
    # remove TEI tags remnants and citation markers
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*\*\*#.*?\*\*\*", "", text)
    text = re.sub(r"\[\d+.*?\]", "", text)
    return text.strip()

def read_csv(path: Path | str) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame, ensuring correct path resolution.
    Handles edge cases like globally quoted files and nested quoting.
    """
    from io import StringIO
    import csv

    path = Path(path)
    # First, try pandas’ parser in a slightly more forgiving mode:
    try:
        df = pd.read_csv(
            path,
            dtype=str,
            quotechar='"',
            skipinitialspace=True,
            on_bad_lines='warn',
            engine='python'
        )
        # if it came back as a single column, likely the file is globally quoted
        if len(df.columns) > 1:
            return df
    except Exception:
        # fall back to the csv module below
        pass

    # Fallback: use the stdlib csv.reader to correctly handle nested quoting
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return pd.DataFrame()
    header, *data = rows
    return pd.DataFrame(data, columns=header)

# Embedding utility using OpenAI

def embed(texts: list[str]) -> list[list[float]]:
    from backend.bl_client import embeddings
    return embeddings(
        texts,
        model="alias-embeddings",
        api_key=os.getenv("BLABLADOR_API_KEY"),
        base_url=os.getenv("BLABLADOR_BASE")
    )

def pick_best_passage(
    claim:       str,
    passages:    List[str],
    mode:        Literal["support", "contradict"],
    model_name:  str,
    api_key:     str,
    base_url:    str,
) -> Tuple[int, str]:
    """
    Ask the LLM to choose the single best passage that
    either supports or contradicts `claim` from `passages`.
    Returns (best_index, rationale).
    """
    instruction = {
        "support":   "Which passage BEST SUPPORTS the claim? Respond ONLY with JSON {\"best_id\":<int>,\"rationale\":<string>}.",
        "contradict":"Which passage BEST CONTRADICTS the claim? Respond ONLY with JSON {\"best_id\":<int>,\"rationale\":<string>}.",
    }[mode]

    payload = {"claim": claim, "passages": passages}
    prompt  = f"{instruction}\n\n```json\n{json.dumps(payload)}\n```"

    client = BlabladorClient(api_key=api_key, base_url=base_url)
    resp_text = client.completion(
        prompt,
        model=model_name,
        temperature=0.0,
        max_tokens=128,
    )

    # Try to pull out the JSON object from the response
    try:
        # simple bracket‐matching extract
        start = resp_text.find("{")
        end   = resp_text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response")
        json_str = resp_text[start : end+1]
        out = json.loads(json_str)
        return out["best_id"], out["rationale"]
    except Exception as e:
        # Log the raw output for debugging
        import logging
        logging.error(f"Failed to parse JSON from LLM response: {e}\nRaw response:\n{resp_text}")
        # Return nothing rather than crash
        return None, None

__all__ = ['clean_text', 'read_csv', 'embed', 'call_llm_justification', 'pick_best_passage']
