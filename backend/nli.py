import os

os.environ["TRANSFORMERS_CACHE"] = str(os.path.expanduser("~/.cache/huggingface"))

import logging
from functools import lru_cache

import torch

from backend.settings import settings as app_settings
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

_log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper().strip()
_log_level = getattr(logging, _log_level_name, logging.INFO)
logging.basicConfig(level=_log_level, format="%(asctime)s %(levelname)s ▶ %(message)s")
for _logger_name in ("urllib3", "huggingface_hub", "transformers"):
    logging.getLogger(_logger_name).setLevel(max(logging.WARNING, _log_level))


def _threshold() -> float:
    try:
        return float(getattr(app_settings, "NLI_THRESHOLD", 0.5) or 0.5)
    except Exception:
        return 0.5


def _default_device() -> str:
    # Prefer MPS on macOS when available, fall back to CPU.
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _effective_batch_size() -> int:
    bs = int(getattr(app_settings, "NLI_BATCH_SIZE", 1) or 1)
    # MPS has limited memory headroom; keep batches small.
    try:
        if torch.backends.mps.is_available():
            return max(1, min(bs, 4))
    except Exception:
        pass
    return max(1, bs)


@lru_cache(maxsize=None)
def get_nli_pipeline(model_name: str, *, device: str | None = None):
    """Load a Hugging Face model as an NLI pipeline.

    Cached to avoid reloading.
    """
    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )
    chosen = (device or _default_device()).strip().lower()
    torch_device = torch.device("cpu")
    if chosen == "mps":
        torch_device = torch.device("mps")
    # Return a text-classification pipeline configured for NLI.
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=torch_device,
    )


def predict_nli(premise: str, hypothesis: str, *, nli_model: str | None = None):
    model_to_use = nli_model or getattr(app_settings, "NLI_MODEL", None)
    if not model_to_use:
        model_to_use = "cross-encoder/nli-deberta-v3-base"
    nli_pipeline = get_nli_pipeline(model_to_use)
    return nli_pipeline(
        f"{premise} [SEP] {hypothesis}",
        top_k=None,
        truncation=True,
        max_length=512,
        padding=True,
    )


SYSTEM_INSTR = """
You are an expert evidence-checking assistant.

Your goal is to determine if each claim is supported by the provided passages,
allowing for synonyms, paraphrases, and implied meanings. Use only the passages
given; do not draw on external knowledge.

For each claim:
- Return a JSON object with a single key "evidence" which is an array of evidence
  objects.
- Each evidence object must have "quote", "location", "label", "chunk_id", and
  "type" keys.
- The "label" for each evidence object must be one of: "entailment" or "contradiction".
If no passage supports or contradicts the claim, return an empty "evidence" array.

Always produce valid JSON with the described structure.
"""


def assess(
    claim: str,
    passages: list[str],
    metadatas: list[dict],
    nli_model: str | None = None,
) -> list[dict]:
    """Run a HF sequence-classification model for NLI.

    Accepts any HF checkpoint string via `nli_model`.
    Returns a list of { quote, chunk_id, type, label, score } dicts,
    keeping only entailment & contradiction above the configured threshold.
    """
    model_to_use = nli_model or "cross-encoder/nli-deberta-v3-base"
    logging.debug(f"[NLI] using model {model_to_use}")
    pipe = get_nli_pipeline(model_to_use)

    input_texts = [f"{text} [SEP] {claim}" for text in passages]
    batch_size = _effective_batch_size()
    try:
        results = pipe(
            input_texts,
            batch_size=batch_size,
            top_k=None,
            truncation=True,
            max_length=512,
            padding=True,
        )
    except Exception as e:
        logging.error(f"[NLI] error during batched pipeline call: {e}")
        # Best-effort fallback: retry on CPU with a tiny batch.
        message = str(e).lower()
        if "mps" in message and "out of memory" in message:
            try:
                cpu_pipe = get_nli_pipeline(model_to_use, device="cpu")
                results = cpu_pipe(
                    input_texts,
                    batch_size=1,
                    top_k=None,
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
            except Exception as exc:
                logging.error(f"[NLI] CPU fallback failed: {exc}")
                return []
        else:
            return []

    evidence = []
    threshold = _threshold()
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

        if ent and ent["score"] >= threshold:
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

        if con and con["score"] >= threshold:
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
