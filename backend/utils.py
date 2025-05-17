# backend/utils.py

import os
import re
from pathlib import Path
import pandas as pd
import json
import logging
from requests import HTTPError
from backend.bl_client import BlabladorClient
from typing import List, Tuple, Literal, Optional  # new import


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
    claim: str,
    passages: List[str],
    mode: Literal["support", "contradict"],
    model_name: str,
    api_key: str,
    base_url: str,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Pick the single best passage by index and produce a one-line rationale.
    Returns (best_index, rationale) or (None, None) on failure.
    """
    # 1) Build a minimal, unambiguous prompt
    verb = "SUPPORTS" if mode == "support" else "CONTRADICTS"
    instruction = (
        f"Claim: {claim}\n"
        f"Passages:\n" + "\n".join(f"{i}: {p}" for i, p in enumerate(passages)) + "\n\n"
        f"Which single passage index {verb} the claim?  \n"
        "Respond with exactly this JSON (no extra text):\n"
        '{"best_id": <index>, "rationale": "<very brief explanation>"}\n'
        "Now your answer:\n"
    )

    client = BlabladorClient(api_key=api_key, base_url=base_url)
    resp_text = client.completion(
        instruction,
        model=model_name,
        temperature=0.0,
        max_tokens=150,
    )

    # 2) Log raw output for debugging—even if it’s empty
    logging.debug(f"pick_best_passage raw response: {resp_text!r}")

    # 3) Try to extract the first {...} block
    if not resp_text or not resp_text.strip():
        logging.error("Empty response from LLM in pick_best_passage")
        return None, None

    # Regex to find {...} (non-greedy)
    match = re.search(r"\{.*?\}", resp_text, flags=re.DOTALL)
    if match:
        candidate = match.group()
        try:
            out = json.loads(candidate)
            bid = out.get("best_id")
            rat = out.get("rationale")
            if isinstance(bid, int) and isinstance(rat, str):
                return bid, rat
            else:
                logging.error(f"Parsed JSON missing expected types: {candidate}")
        except Exception as e:
            logging.error(f"Failed to json.loads() in pick_best_passage: {e}\nCandidate: {candidate}")

    else:
        logging.error(f"No JSON object found in pick_best_passage output:\n{resp_text}")

    # 4) Fallback: pick the highest-score passage automatically
    logging.warning("Falling back to highest-FAISS-score passage (index 0) without rationale")
    return 0, "no rationale"

__all__ = ['clean_text', 'read_csv', 'embed', 'pick_best_passage']
