import json
from pathlib import Path
from typing import Literal, List

# We’ll track four categories of models
ModelCategory = Literal['embed', 'reranker', 'nli', 'llm']

CACHE_FILE = Path(__file__).parent / "model_cache.json"

def _load_cache() -> dict[ModelCategory, List[str]]:
    """
    Return a dict mapping each category to its list of cached models.
    If the file doesn’t exist yet, initialize empty lists.
    """
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    else:
        # start with empty history for each category
        cache = {cat: [] for cat in ('embed','reranker','nli','llm')}
        _save_cache(cache)
        return cache

def _save_cache(cache: dict[ModelCategory, List[str]]):
    """
    Persist the entire cache dict to disk.
    """
    CACHE_FILE.write_text(json.dumps(cache, indent=2))

def get_models(category: ModelCategory) -> List[str]:
    """
    Retrieve the history list for a given category.
    """
    cache = _load_cache()
    return cache.get(category, [])

def add_model(category: ModelCategory, model: str):
    """
    Add a model name to the history for a category (if not already present),
    then save the updated cache.
    """
    cache = _load_cache()
    lst = cache.setdefault(category, [])
    if model not in lst:
        lst.append(model)
        _save_cache(cache)