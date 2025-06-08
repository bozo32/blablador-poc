from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline as hf_pipeline,
)

_LOCAL_NLI_MODELS = {}

try:
    for name, path in {
        "deberta-base": "cross-encoder/nli-deberta-v3-base",
        "deberta-large": "microsoft/deberta-v3-large",
    }.items():
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        _LOCAL_NLI_MODELS[name] = hf_pipeline(
            "text-classification",
            model=model,
            tokenizer=tok,
            return_all_scores=True,
            device=0,
        )
except Exception as e:
    print(f"Warning: could not load local NLI models: {e}")
    _LOCAL_NLI_MODELS = {}

assert "text" in first_ev
