# backend/nli.py

import os, json
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s ▶ %(message)s")
import functools
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.bl_client import BlabladorClient   # new central wrapper
from requests import HTTPError
from functools import lru_cache

THRESHOLD = 0.0

@lru_cache(maxsize=None)
def get_nli_pipeline(model_name: str):
    """
    Dynamically load any Hugging Face sequence-classification model as an NLI pipeline.
    Cached to avoid reloading.
    """
    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # Return a text-classification pipeline configured for NLI
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True
    )

def predict_nli(premise: str, hypothesis: str):
    nli_pipeline = get_nli_pipeline()
    result = nli_pipeline(f"{premise} [SEP] {hypothesis}", top_k=None)
    return result

# -----------------------------------------------------------------------------
# Local NLI model registry: defer loading until needed
# -----------------------------------------------------------------------------

# Map model names to HuggingFace repo paths
_LOCAL_NLI_PATHS = {
    "deberta-base": "cross-encoder/nli-deberta-v3-base",
    "deberta-large": "microsoft/deberta-v3-large"
}

_LOCAL_NLI_MODELS: dict[str, any] = {}

def get_local_nli_pipeline(name: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
    if name not in _LOCAL_NLI_PATHS:
        raise KeyError(f"No local NLI model configured for '{name}'")
    if name not in _LOCAL_NLI_MODELS[name]:
        path = _LOCAL_NLI_PATHS[name]
        tok = AutoTokenizer.from_pretrained(path, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        _LOCAL_NLI_MODELS[name] = hf_pipeline(
            "text-classification",
            model=model,
            tokenizer=tok,
            return_all_scores=True,
            device=-1
        )
    return _LOCAL_NLI_MODELS[name]

# A system prompt that fixes the model’s role and constraints:
SYSTEM_INSTR = """
You are an expert evidence‐checking assistant. Your goal is to determine if each claim is supported by the provided passages, allowing for synonyms, paraphrases, and implied meanings. Use only the passages given; do not draw on external knowledge.

For each claim:
- Return a JSON object with a single key "evidence" which is an array of evidence objects.
- Each evidence object must have "quote", "location", "label", "chunk_id", and "type" keys.
- The "label" for each evidence object must be one of: "entailment" or "contradiction".
If no passage supports or contradicts the claim, return an empty "evidence" array.

Always produce valid JSON with the described structure.
"""

def assess(
    claim: str,
    passages: list[str],
    metadatas: list[dict],
    nli_model: str = None
) -> list[dict]:
    """
    Always runs a HF sequence-classification model for NLI.
    Accepts any HF checkpoint string via `nli_model`.
    Returns a list of { quote, chunk_id, type, label, score } dicts,
    keeping only entailment & contradiction above THRESHOLD.
    """
    model_to_use = nli_model or "cross-encoder/nli-deberta-v3-base"
    logging.debug(f"[NLI] using model {model_to_use}")
    pipe = get_nli_pipeline(model_to_use)
    evidence = []
    for text, meta in zip(passages, metadatas):
        # the cross-encoder expects "[premise] [SEP] [hypothesis]"
        # here: passage is the premise, claim is the hypothesis
        in_text = f"{text} [SEP] {claim}"
        try:
            preds = pipe(in_text)
            # HF sometimes returns nested lists—flatten to List[dict]
            if isinstance(preds, dict):
                preds = [preds]
            elif isinstance(preds, list) and preds and isinstance(preds[0], list):
                preds = preds[0]
            logging.debug(f"[NLI→preds] {preds}")
        except Exception as e:
            logging.error(f"[NLI] error during pipeline(...) for a passage: {e}")
            continue

        # extract scores for entailment & contradiction
        ent = next((p for p in preds if p["label"].lower() == "entailment"), None)
        con = next((p for p in preds if p["label"].lower() == "contradiction"), None)

        if ent and ent["score"] >= THRESHOLD:
            evidence.append({
                "text":       text,
                "chunk_id":   meta.get("chunk_id"),
                "type":       meta.get("type"),
                "label":      "entailment",
                "score":      ent["score"],
            })

        if con and con["score"] >= THRESHOLD:
            evidence.append({
                "text":       text,
                "chunk_id":   meta.get("chunk_id"),
                "type":       meta.get("type"),
                "label":      "contradiction",
                "score":      con["score"],
            })

    return evidence