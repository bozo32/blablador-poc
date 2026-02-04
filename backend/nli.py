import os

os.environ["TRANSFORMERS_CACHE"] = str(os.path.expanduser("~/.cache/huggingface"))

import logging
from functools import lru_cache

import torch

from backend.settings import AppSettings, settings as app_settings
from backend.utils import hf_inference_post
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

_log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper().strip()
_log_level = getattr(logging, _log_level_name, logging.INFO)
logging.basicConfig(level=_log_level, format="%(asctime)s %(levelname)s ▶ %(message)s")
for _logger_name in ("urllib3", "huggingface_hub", "transformers"):
    logging.getLogger(_logger_name).setLevel(max(logging.WARNING, _log_level))


def _threshold(settings: AppSettings) -> float:
    try:
        return float(getattr(settings, "NLI_THRESHOLD", 0.5) or 0.5)
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


def _effective_batch_size(settings: AppSettings) -> int:
    bs = int(getattr(settings, "NLI_BATCH_SIZE", 1) or 1)
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


HF_INFERENCE_API_BASE = "https://api-inference.huggingface.co/models"
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def _safe_err(err: Exception) -> str:
    msg = str(err).strip()
    if len(msg) > 240:
        msg = msg[:240] + "…"
    return msg or err.__class__.__name__


def _resolve_hf_model_id(model_name: str | None, *, settings: AppSettings) -> str:
    requested = (model_name or "").strip()
    if not requested:
        requested = (getattr(settings, "NLI_MODEL", "") or "").strip()
    if not requested:
        requested = DEFAULT_NLI_MODEL

    mapping = {
        # Historical default used by older code paths.
        "cross-encoder/nli-deberta-v3-base": "cross-encoder/nli-deberta-v3-base",
        # Current default in settings.py.
        DEFAULT_NLI_MODEL: DEFAULT_NLI_MODEL,
    }
    return mapping.get(requested, requested)


def _hf_remote_requested(
    *, settings: AppSettings, advanced_settings: dict | None
) -> bool:
    if isinstance(advanced_settings, dict) and "hf_remote" in advanced_settings:
        return bool(advanced_settings.get("hf_remote"))
    return bool(getattr(settings, "HF_REMOTE_INFERENCE", False))


def _hf_token(settings: AppSettings) -> str:
    # Prefer current env var value, fall back to settings (env/.env loaded at startup).
    token = (
        os.environ.get("HF_API_TOKEN") or getattr(settings, "HF_API_TOKEN", "") or ""
    ).strip()
    return token


def _assess_remote(
    claim: str,
    passages: list[str],
    metadatas: list[dict],
    *,
    settings: AppSettings,
    nli_model: str | None,
    timeout: int = 30,
) -> list[dict]:
    token = _hf_token(settings)
    if not token:
        raise RuntimeError("HF_API_TOKEN is missing")
    model_id = _resolve_hf_model_id(nli_model, settings=settings)
    url = f"{HF_INFERENCE_API_BASE}/{model_id}"
    input_texts = [f"{text} [SEP] {claim}" for text in passages]
    payload = {
        "inputs": input_texts,
        "options": {"wait_for_model": True},
    }
    raw = hf_inference_post(model_id, token=token, payload=payload, timeout=timeout)

    if isinstance(raw, dict) and raw.get("error"):
        raise RuntimeError(str(raw.get("error")))
    if not isinstance(raw, list):
        raise RuntimeError(f"Unexpected HF inference response type: {type(raw)}")
    if not input_texts:
        return []

    per_input: list[list[dict]] = []
    if raw and isinstance(raw[0], list):
        per_input = raw  # type: ignore[assignment]
    elif raw and isinstance(raw[0], dict):
        if len(input_texts) != 1:
            raise RuntimeError("HF inference returned single output for batched inputs")
        per_input = [raw]  # type: ignore[list-item]
    else:
        raise RuntimeError("HF inference returned empty or malformed output")

    threshold = _threshold(settings)
    evidence: list[dict] = []
    for preds, text, meta in zip(per_input, passages, metadatas):
        if isinstance(preds, dict):
            preds = [preds]
        class_scores = {
            str(p.get("label", "")).lower(): float(p.get("score", 0.0) or 0.0)
            for p in preds
            if isinstance(p, dict)
        }

        ent_score = None
        con_score = None
        for label, score in class_scores.items():
            if label == "entailment":
                ent_score = score
            elif label == "contradiction":
                con_score = score

        if ent_score is None and con_score is None:
            raise RuntimeError("HF inference response missing NLI labels")

        if ent_score is not None and ent_score >= threshold:
            ev_dict = {
                "text": text,
                "label": "entailment",
                "score": ent_score,
                **meta,
            }
            ev_dict["chunk_id"] = meta.get("id") or meta.get("chunk_id")
            ev_dict["type"] = meta.get("type")
            ev_dict["all_scores"] = class_scores
            evidence.append(ev_dict)

        if con_score is not None and con_score >= threshold:
            ev_dict = {
                "text": text,
                "label": "contradiction",
                "score": con_score,
                **meta,
            }
            ev_dict["chunk_id"] = meta.get("id") or meta.get("chunk_id")
            ev_dict["type"] = meta.get("type")
            ev_dict["all_scores"] = class_scores
            evidence.append(ev_dict)

    logging.info("[NLI] used HF Inference API: %s", url)
    return evidence


def assess(
    claim: str,
    passages: list[str],
    metadatas: list[dict],
    nli_model: str | None = None,
    *,
    settings: AppSettings | None = None,
    advanced_settings: dict | None = None,
) -> list[dict]:
    """Run a HF sequence-classification model for NLI.

    Accepts any HF checkpoint string via `nli_model`.
    Returns a list of { quote, chunk_id, type, label, score } dicts,
    keeping only entailment & contradiction above the configured threshold.
    """
    effective_settings = settings or app_settings

    if _hf_remote_requested(
        settings=effective_settings, advanced_settings=advanced_settings
    ):
        token = _hf_token(effective_settings)
        if token:
            try:
                return _assess_remote(
                    claim,
                    passages,
                    metadatas,
                    settings=effective_settings,
                    nli_model=nli_model,
                )
            except Exception as exc:  # noqa: BLE001 - fall back to local
                logging.warning(
                    "[NLI] HF remote inference failed; falling back to local (%s)",
                    _safe_err(exc),
                )

    model_to_use = nli_model or "cross-encoder/nli-deberta-v3-base"
    logging.debug(f"[NLI] using model {model_to_use}")
    pipe = get_nli_pipeline(model_to_use)

    input_texts = [f"{text} [SEP] {claim}" for text in passages]
    batch_size = _effective_batch_size(effective_settings)
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
    threshold = _threshold(effective_settings)
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
