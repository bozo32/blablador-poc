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
from sentence_transformers import SentenceTransformer


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

# ----------------------------------------
# Local‐HF sentence‐transformers cache & loading
# ----------------------------------------

# Use a dedicated cache folder for local HF models
MODEL_CACHE_DIR = Path.home() / ".cache/hf_sentence_models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# List locally cached models (those with a config.json file)
def list_local_models() -> list[str]:
    return sorted([
        d.name
        for d in MODEL_CACHE_DIR.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    ])

# Cache loaded models in memory for performance
_loaded_models: dict[str, SentenceTransformer] = {}

def get_model(model_name: str) -> SentenceTransformer:
    """
    Load a SentenceTransformer from local cache if present; otherwise, download to MODEL_CACHE_DIR.
    """
    if model_name in _loaded_models:
        return _loaded_models[model_name]
    try:
        local_path = MODEL_CACHE_DIR / model_name
        if local_path.exists():
            model = SentenceTransformer(str(local_path))
        else:
            model = SentenceTransformer(model_name, cache_folder=str(MODEL_CACHE_DIR))
        _loaded_models[model_name] = model
        return model
    except Exception as e:
        raise RuntimeError(f"Could not load embedding model '{model_name}': {e}")

def embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    """
    Embed texts using a local Hugging Face model (as chosen by the user).
    """
    model = get_model(model_name)
    return model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True
    ).tolist()

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

__all__ = ['clean_text', 'read_csv', 'embed', 'list_local_models', 'get_model', 'pick_best_passage']
