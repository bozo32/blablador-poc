"""Evidence card rendering utilities and keyboard-friendly component helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import streamlit as st
import streamlit.components.v1 as components

ASSET_PATH = Path(__file__).resolve().parent.parent / "assets" / "evidence.css"

HighlightSpan = Tuple[int, int]


def truncate_snippet(text: str, *, limit: int = 600) -> str:
    """Trim snippets to the configured limit without chopping words mid-stream."""
    clean = (text or "").strip()
    if not clean or len(clean) <= limit:
        return clean
    truncated = clean[:limit]
    last_space = truncated.rfind(" ")
    if last_space > limit * 0.7:  # keep most of the sentence intact
        truncated = truncated[:last_space]
    return truncated.rstrip(" ,.;") + "…"


def merge_highlight_spans(spans: Sequence[HighlightSpan]) -> List[HighlightSpan]:
    """Merge overlapping highlight spans to keep markup predictable."""
    normalized: List[HighlightSpan] = []
    for start, end in sorted(spans or [], key=lambda pair: pair[0]):
        if end <= start:
            continue
        if not normalized:
            normalized.append((start, end))
            continue
        prev_start, prev_end = normalized[-1]
        if start <= prev_end:
            normalized[-1] = (prev_start, max(prev_end, end))
        else:
            normalized.append((start, end))
    return normalized


def build_focus_order(
    candidates: Sequence[Dict[str, Any]],
    *,
    pinned_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return candidate IDs ordered for keyboard navigation."""
    pinned = set(pinned_ids or [])
    ordered: List[str] = []
    for bucket in (True, False):
        for candidate in candidates:
            cid = candidate.get("id")
            if not cid:
                continue
            if (cid in pinned) is bucket:
                ordered.append(cid)
    return ordered


def _apply_highlights(snippet: str, spans: Sequence[HighlightSpan]) -> str:
    merged = merge_highlight_spans(spans)
    if not snippet or not merged:
        return snippet
    highlighted = []
    last_index = 0
    for start, end in merged:
        highlighted.append(_escape_html(snippet[last_index:start]))
        highlighted.append(
            (
                '<mark class="evidence-card__highlight">'
                f"{_escape_html(snippet[start:end])}"
                "</mark>"
            )
        )
        last_index = end
    highlighted.append(_escape_html(snippet[last_index:]))
    return "".join(highlighted)


def _escape_html(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_badge(label: str, *, tone: str = "neutral") -> str:
    return '<span class="evidence-badge evidence-badge--{tone}">{label}</span>'.format(
        tone=tone,
        label=_escape_html(label),
    )


def _format_confidence_sparkline(values: Sequence[float]) -> str:
    bars = []
    normalized = [max(0.0, min(1.0, float(val))) for val in values[:12]]
    for val in normalized:
        height = 10 + int(val * 30)
        bars.append(
            '<span class="evidence-card__spark-bar" '
            f'style="height:{height}px"></span>'
        )
    if not bars:
        bars.append('<span class="evidence-card__spark-placeholder">∿</span>')
    return "".join(bars)


def _serialize_metadata(candidate: Dict[str, Any]) -> str:
    meta = candidate.get("metadata") or {}
    parts = []
    if meta.get("page"):
        parts.append(f"Page {meta['page']}")
    if meta.get("section"):
        parts.append(meta["section"])
    if meta.get("provenance"):
        parts.append(meta["provenance"].title())
    if candidate.get("confidence") is not None:
        score = f"{float(candidate['confidence']):.2f}"
        parts.append(f"Confidence {score}")
    if meta.get("ocr_quality"):
        parts.append(f"OCR {meta['ocr_quality']}")
    return " • ".join(parts)


def _describe_actions(lock_state: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    lock_state = lock_state or {}
    rerun_locked = lock_state.get("rerun_inflight", False)
    return {
        "primary_disabled": bool(lock_state.get("accept_locked") or rerun_locked),
        "secondary_disabled": bool(lock_state.get("reject_locked") or rerun_locked),
        "pins_disabled": bool(lock_state.get("pin_locked")),
    }


CardCallback = Callable[[Dict[str, Any]], None]


@dataclass
class CardActionCallbacks:
    """Bundle callbacks for rendering inline evidence actions."""

    accept: Optional[CardCallback] = None
    reject: Optional[CardCallback] = None
    pin: Optional[CardCallback] = None
    share: Optional[CardCallback] = None
    open_pdf: Optional[CardCallback] = None


@dataclass
class EvidenceCardRenderer:
    ui: Any = st
    keyboard_namespace: str = "evidence-card-nav"
    css_path: Path = ASSET_PATH
    callbacks: CardActionCallbacks = field(default_factory=CardActionCallbacks)

    _css_injected: bool = field(default=False, init=False)
    _keyboard_bound: bool = field(default=False, init=False)

    def render_cards(
        self,
        claim_id: str,
        candidates: Sequence[Dict[str, Any]],
        *,
        pinned_ids: Optional[Iterable[str]] = None,
        lock_state: Optional[Dict[str, Any]] = None,
        enable_keyboard: bool = True,
    ) -> None:
        self._inject_styles()
        if enable_keyboard:
            self._inject_keyboard_script()
        for idx, candidate in enumerate(candidates, start=1):
            card_id = candidate.get("id") or f"{claim_id}-candidate-{idx}"
            self._render_card(
                claim_id,
                candidate,
                card_id=card_id,
                lock_state=lock_state,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _render_card(
        self,
        claim_id: str,
        candidate: Dict[str, Any],
        *,
        card_id: str,
        lock_state: Optional[Dict[str, Any]],
    ) -> None:
        label = (candidate.get("label") or "unknown").lower()
        snippet = truncate_snippet(
            candidate.get("text") or candidate.get("snippet", "")
        )
        highlights = candidate.get("highlights") or []
        highlight_spans = [
            (int(span.get("start", 0)), int(span.get("end", 0)))
            for span in highlights
            if span is not None
        ]
        snippet_html = _apply_highlights(snippet, highlight_spans)
        header = self._render_header(candidate, label=label)
        metadata = _serialize_metadata(candidate)
        sparkline = _format_confidence_sparkline(candidate.get("sparkline") or [])
        notes = candidate.get("notes_summary") or ""
        rationale = candidate.get("why_matched") or []
        actions = _describe_actions(lock_state)

        self.ui.markdown(
            f"""
            <div
                class="evidence-card"
                data-evidence-card="true"
                data-card-id="{card_id}"
                data-label="{label}"
            >
                {header}
                <div class="evidence-card__snippet" aria-label="Evidence snippet">
                    {snippet_html}
                </div>
                <div class="evidence-card__sparkline" aria-label="Confidence sparkline">
                    {sparkline}
                </div>
                <div class="evidence-card__meta">{_escape_html(metadata)}</div>
                {self._render_notes(notes)}
                {self._render_rationale(rationale)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        self._render_actions(
            claim_id,
            card_id,
            candidate,
            disabled=actions,
        )

    def _render_header(self, candidate: Dict[str, Any], *, label: str) -> str:
        label_badge = _format_badge(
            label.title(),
            tone="entail"
            if label == "entail"
            else "contrad"
            if label == "contradict"
            else "neutral",
        )
        rank = candidate.get("rank")
        rank_badge = (
            _format_badge(f"Rank {rank}", tone="rank") if rank is not None else ""
        )
        provenance = candidate.get("metadata", {}).get("provenance")
        provenance_badge = (
            _format_badge(provenance.title(), tone="info") if provenance else ""
        )
        reviewer = candidate.get("reviewer_initials")
        reviewer_badge = _format_badge(reviewer, tone="reviewer") if reviewer else ""
        delta = candidate.get("rank_delta")
        delta_badge = ""
        if delta:
            arrow = "↑" if delta < 0 else "↓"
            delta_badge = _format_badge(f"{arrow} {abs(delta)}", tone="delta")
        badges = "".join(
            filter(
                None,
                [
                    label_badge,
                    rank_badge,
                    provenance_badge,
                    reviewer_badge,
                    delta_badge,
                ],
            )
        )
        title = candidate.get("title") or "Supporting evidence"
        return (
            '<div class="evidence-card__header">'
            f"<div><h4>{_escape_html(title)}</h4></div>"
            f'<div class="evidence-card__badges">{badges}</div>'
            "</div>"
        )

    def _render_notes(self, notes: str) -> str:
        if not notes:
            return ""
        return (
            '<div class="evidence-card__notes" aria-label="Reviewer notes">'
            f"{_escape_html(notes)}"
            "</div>"
        )

    def _render_rationale(self, rationale: Sequence[str]) -> str:
        if not rationale:
            return ""
        items = "".join(f"<li>{_escape_html(item)}</li>" for item in rationale if item)
        return (
            '<div class="evidence-card__rationale">'
            "<strong>Why matched</strong>"
            f"<ul>{items}</ul>"
            "</div>"
        )

    def _render_actions(
        self,
        claim_id: str,
        card_id: str,
        candidate: Dict[str, Any],
        *,
        disabled: Dict[str, bool],
    ) -> None:
        cols = self.ui.columns([1, 1, 1, 1, 1])
        action_specs = [
            (
                "Accept",
                "accept",
                self.callbacks.accept,
                not disabled["primary_disabled"],
                True,
            ),
            (
                "Reject",
                "reject",
                self.callbacks.reject,
                not disabled["secondary_disabled"],
                False,
            ),
            ("Pin", "pin", self.callbacks.pin, not disabled["pins_disabled"], False),
            ("Share", "share", self.callbacks.share, True, False),
            ("Open PDF", "open", self.callbacks.open_pdf, True, False),
        ]
        for column, (label, action, callback, enabled, primary) in zip(
            cols, action_specs
        ):
            with column:
                key = f"{claim_id}-{card_id}-{action}"
                self.ui.button(
                    label,
                    key=key,
                    type="primary" if primary else "secondary",
                    disabled=not enabled or callback is None,
                    use_container_width=True,
                    on_click=callback,
                    args=(candidate,),
                )
                self._register_focus_target(
                    card_id,
                    action,
                    primary=primary,
                    label=label,
                )

    def _register_focus_target(
        self, card_id: str, action: str, *, primary: bool, label: str
    ) -> None:
        self.ui.markdown(
            f"""
            <div
                class="evidence-action-proxy"
                data-card-id="{card_id}"
                data-action="{action}"
                data-primary="{str(primary).lower()}"
                data-aria-label="{_escape_html(label)}"
            ></div>
            """,
            unsafe_allow_html=True,
        )

    def _inject_styles(self) -> None:
        if self._css_injected:
            return
        if self.css_path.exists():
            self.ui.markdown(
                f"<style>{self.css_path.read_text()}</style>",
                unsafe_allow_html=True,
            )
        self._css_injected = True

    def _inject_keyboard_script(self) -> None:
        if self._keyboard_bound:
            return
        script = f"""
        <script>
        (function() {{
            if (window['{self.keyboard_namespace}']) {{
                return;
            }}
            window['{self.keyboard_namespace}'] = true;

            function tagButtons() {{
                document
                    .querySelectorAll('.evidence-action-proxy')
                    .forEach(function(proxy) {{
                        var wrapper = proxy.previousElementSibling;
                        if (!wrapper) return;
                        var button = wrapper.querySelector('button');
                        if (!button) return;
                        button.dataset.focusable = 'true';
                        button.dataset.cardId = proxy.dataset.cardId;
                        button.dataset.action = proxy.dataset.action;
                        button.dataset.primaryAction = proxy.dataset.primary;
                        if (proxy.dataset.ariaLabel) {{
                            button.setAttribute('aria-label', proxy.dataset.ariaLabel);
                        }}
                        proxy.remove();
                    }});
            }}

            const observer = new MutationObserver(function() {{
                tagButtons();
            }});
            observer.observe(document.body, {{ childList: true, subtree: true }});
            tagButtons();

            document.addEventListener(
                'keydown',
                function(evt) {{
                    const keys = [
                        'ArrowDown',
                        'ArrowUp',
                        'ArrowLeft',
                        'ArrowRight',
                        'Enter'
                    ];
                    if (keys.indexOf(evt.key) === -1) return;
                    const buttons = Array.from(
                        document.querySelectorAll('button[data-focusable="true"]')
                    );
                    if (!buttons.length) return;
                    let index = buttons.indexOf(document.activeElement);
                    if (index === -1) {{
                        index = 0;
                    }}
                    if (evt.key === 'Enter') {{
                        const active = document.activeElement;
                        if (
                            active &&
                            active.dataset.primaryAction === 'true'
                        ) {{
                            active.click();
                            evt.preventDefault();
                        }}
                        return;
                    }}
                    if (evt.key === 'ArrowDown' || evt.key === 'ArrowRight') {{
                        index = (index + 1) % buttons.length;
                        buttons[index].focus();
                        evt.preventDefault();
                        return;
                    }}
                    if (evt.key === 'ArrowUp' || evt.key === 'ArrowLeft') {{
                        index = index - 1;
                        if (index < 0) {{
                            index = buttons.length - 1;
                        }}
                        buttons[index].focus();
                        evt.preventDefault();
                    }}
                }},
                true
            );
        }})();
        </script>
        """
        components.html(script, height=0)
        self._keyboard_bound = True


__all__ = [
    "EvidenceCardRenderer",
    "CardActionCallbacks",
    "truncate_snippet",
    "merge_highlight_spans",
    "build_focus_order",
]
