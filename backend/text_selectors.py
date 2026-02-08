from __future__ import annotations

"""Small utilities for stable text-span anchoring.

We store quote selectors (exact/prefix/suffix) because:
- character offsets drift when PDFs are reprocessed with different parsers
- reviewers may segment the same citing sentence differently

The selectors are intentionally simple and JSON-friendly so they can be embedded
into judgment provenance or graph edges without tight coupling to one extractor.

This module is currently used as a shared reference implementation for how we
normalize, fingerprint, and (optionally) re-resolve quote selectors.
"""

import hashlib
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Optional, Tuple


_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for matching/fingerprinting.

    Keep this conservative: whitespace collapse + strip.
    (Lowercasing is applied by callers when appropriate.)
    """
    return _WS_RE.sub(" ", str(text or "")).strip()


def fingerprint(text: str) -> str:
    """Stable sha256 fingerprint of normalized text."""
    value = normalize_text(text)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_anchor_quote(
    window_text: str,
    *,
    exact_words: int = 14,
    prefix_words: int = 14,
) -> Dict[str, Optional[str]]:
    """Build a small quote selector near the end of a citation window.

    For in-text citations, the most reliable anchor is often the *preceding*
    phrase right before the citation marker. We therefore take the last N words
    as `exact` and the preceding N words as `prefix`.
    """
    norm = normalize_text(window_text)
    tokens = [t for t in norm.split(" ") if t]
    if not tokens:
        return {"exact": None, "prefix": None, "suffix": None}
    exact_tokens = tokens[-exact_words:]
    prefix_tokens = tokens[-(exact_words + prefix_words) : -exact_words]
    exact = " ".join(exact_tokens).strip() or None
    prefix = " ".join(prefix_tokens).strip() or None
    return {"exact": exact, "prefix": prefix, "suffix": None}


def resolve_quote_selector(
    text: str,
    selector: Dict[str, Any],
    *,
    min_ratio: float = 0.72,
) -> Optional[Tuple[int, int, float]]:
    """Best-effort locate a quote selector in text.

    Strategy:
    1) Exact match on normalized selector.exact.
    2) If not found, fuzzy scan by comparing a sliding window to selector.exact.

    Returns (start, end, confidence) in normalized-text coordinates.
    Callers that need raw offsets should run this on the same normalized string
    they persist.
    """
    hay = normalize_text(text)
    exact = normalize_text(str((selector or {}).get("exact") or ""))
    if not hay or not exact:
        return None

    idx = hay.find(exact)
    if idx >= 0:
        return (idx, idx + len(exact), 1.0)

    # Fuzzy scan. This is intentionally simple (no dependencies) and bounded by
    # the typical citation-window size (a sentence or two).
    best = (0, 0, 0.0)
    step = max(1, len(exact) // 6)
    for start in range(0, max(1, len(hay) - len(exact) + 1), step):
        chunk = hay[start : start + len(exact)]
        ratio = SequenceMatcher(a=chunk, b=exact).ratio()
        if ratio > best[2]:
            best = (start, start + len(exact), float(ratio))
    if best[2] < min_ratio:
        return None
    return best
