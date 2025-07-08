import os
os.environ["TRANSFORMERS_CACHE"] = str(os.path.expanduser("~/.cache/huggingface"))

# backend/nli.py

import logging
from functools import lru_cache
from backend.settings import settings as app_settings


from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s ▶ %(message)s"
)


THRESHOLD = 0.3


@lru_cache(maxsize=None)
def get_nli_pipeline(model_name: str):
    """Dynamically load any Hugging Face sequence-classification model as an
    NLI pipeline.

    Cached to avoid reloading.
    """
    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    # Return a text-classification pipeline configured for NLI
    return pipeline(
        "text-classification", model=model, tokenizer=tokenizer, return_all_scores=True
    )


def predict_nli(premise: str, hypothesis: str):
    nli_pipeline = get_nli_pipeline()
    result = nli_pipeline(f"{premise} [SEP] {hypothesis}", top_k=None)
    return result


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
    claim: str, passages: list[str], metadatas: list[dict], nli_model: str = None
) -> list[dict]:
    """Always runs a HF sequence-classification model for NLI.

    Accepts any HF checkpoint string via `nli_model`.
    Returns a list of { quote, chunk_id, type, label, score } dicts,
    keeping only entailment & contradiction above THRESHOLD.
    """
    model_to_use = nli_model or "cross-encoder/nli-deberta-v3-base"
    logging.debug(f"[NLI] using model {model_to_use}")
    pipe = get_nli_pipeline(model_to_use)

    input_texts = [f"{text} [SEP] {claim}" for text in passages]
    try:
        results = pipe(
            input_texts, batch_size=app_settings.NLI_BATCH_SIZE
        )  # optionally set batch_size param
    except Exception as e:
        logging.error(f"[NLI] error during batched pipeline call: {e}")
        return []

    evidence = []
    for preds, text, meta in zip(results, passages, metadatas):
        # HF sometimes returns nested lists—flatten to List[dict]
        if isinstance(preds, dict):
            preds = [preds]
        elif isinstance(preds, list) and preds and isinstance(preds[0], list):
            preds = preds[0]
        logging.debug(f"[NLI→preds] {preds}")

        # Extract all label: score pairs for this passage
        class_scores = {p["label"].lower(): p["score"] for p in preds}

        # extract scores for entailment & contradiction
        ent = next((p for p in preds if p["label"].lower() == "entailment"), None)
        con = next((p for p in preds if p["label"].lower() == "contradiction"), None)

        if ent and ent["score"] >= THRESHOLD:
            ev_dict = {
                "text": text,
                "label": "entailment",
                "score": ent["score"],
                **meta,
            }
            # Always set chunk_id and type explicitly for consistency:
            ev_dict["chunk_id"] = meta.get("id") or meta.get("chunk_id")
            ev_dict["type"] = meta.get("type")
            ev_dict["all_scores"] = class_scores
            evidence.append(ev_dict)

        if con and con["score"] >= THRESHOLD:
            ev_dict = {
                "text": text,
                "label": "contradiction",
                "score": con["score"],
                **meta,
            }
            ev_dict["chunk_id"] = meta.get("id") or meta.get("chunk_id")
            ev_dict["type"] = meta.get("type")
            ev_dict["all_scores"] = class_scores
            evidence.append(ev_dict)

    return evidence
