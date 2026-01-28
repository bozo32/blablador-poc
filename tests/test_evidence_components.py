"""Helper tests covering evidence component utilities."""

from frontend.components.evidence_card import (
    build_focus_order,
    merge_highlight_spans,
    truncate_snippet,
)


def test_card_truncate_snippet_respects_limit_and_word_boundary() -> None:
    text = "sentence " * 200
    result = truncate_snippet(text, limit=120)
    assert len(result) <= 121  # allow ellipsis
    assert result.endswith("…")
    assert "sentence" in result  # keeps readable chunk


def test_card_truncate_snippet_returns_original_when_short() -> None:
    text = "short snippet"
    assert truncate_snippet(text, limit=200) == text


def test_card_merge_highlight_spans_merges_and_orders() -> None:
    spans = [(0, 10), (5, 15), (30, 40), (35, 50), (60, 50)]
    assert merge_highlight_spans(spans) == [(0, 15), (30, 50)]


def test_card_build_focus_order_prioritizes_pinned_ids() -> None:
    candidates = [
        {"id": "a"},
        {"id": "b"},
        {"id": "c"},
    ]
    order = build_focus_order(candidates, pinned_ids=["b", "missing"])
    assert order == ["b", "a", "c"]


def test_card_build_focus_order_ignores_missing_ids() -> None:
    candidates = [{"foo": "bar"}]
    assert build_focus_order(candidates) == []
