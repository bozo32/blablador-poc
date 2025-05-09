# backend/nli.py

import os, json
import logging
import functools
from transformers import pipeline
from backend.bl_client import BlabladorClient   # new central wrapper
from requests import HTTPError

THRESHOLD = 0.0

@functools.lru_cache()
def get_nli_pipeline():
    return pipeline("text-classification", model="microsoft/deberta-v3-large-mnli")

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
    if name not in _LOCAL_NLI_MODELS:
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
- Return  a JSON object with a single key "evidence" which is an array of evidence objects.
- Each evidence object must have "quote", "location", "assessment", "chunk_id", and "type" keys.
- The "assessment" for each evidence object must be one of: "yes", "circumstantial", "partial", or "no".
- If no passage supports or implies the claim, return an empty "evidence" array.

Always produce valid JSON with the described structure.
"""

def assess(
    claim: str,
    passages: list[str],
    metadatas: list[dict],
    nli_model: str = None,
    llm_model: str = None,
    api_key: str = None,
    base_url: str = None
) -> list[dict]:
    # If a local NLI model is requested, run locally
    if nli_model in _LOCAL_NLI_PATHS:
        pipeline = get_local_nli_pipeline(nli_model)
        evidence = []
        for text, meta in zip(passages, metadatas):
            # run NLI on claim vs text
            in_text = f"{claim} [SEP] {text}"
            preds = pipeline(in_text, truncation=True)
            # find highest entailment or contradiction
            best = max(preds, key=lambda p: p["score"])
            if best["score"] < THRESHOLD:
                continue
            label = best["label"].lower()
            assessment = "yes" if "entail" in label else ("no" if "contradict" in label else "circumstantial")
            evidence.append({
                "quote": text,
                "location": meta.get("chunk_id"),
                "assessment": assessment,
                "chunk_id": meta.get("chunk_id"),
                "type": meta.get("type"),
                "score": best["score"]
            })
        return evidence
    else:
        # ------------------------------------------------------------------
        # Remote LLM call via Blablador /v1/completions (through wrapper)
        # ------------------------------------------------------------------
        actual_model = llm_model or "alias-large"
        user_payload = json.dumps({
            "claim": claim,
            "passages": [
                {"text": p,
                 "chunk_id": m.get("chunk_id"),
                 "type":     m.get("type")}
                for p, m in zip(passages, metadatas)
            ]
        }, ensure_ascii=False)

        prompt = (
            SYSTEM_INSTR.strip()
            + "\n\n<INPUT>\n"
            + user_payload
            + "\n</INPUT>\n\nReturn the JSON object as specified above."
        )

        # use the BlabladorClient instead of bare completion()
        client = BlabladorClient(
            api_key=api_key or os.getenv("BLABLADOR_API_KEY"),
            base_url=base_url or os.getenv("BLABLADOR_BASE")
        )
        raw = client.completion(
            prompt=prompt,
            model=actual_model,
            temperature=0.0,
            max_tokens=512
        )

        try:
            result = json.loads(raw)
            return result.get("evidence", [])
        except Exception:
            # model returned non‑JSON – treat as no evidence
            return []