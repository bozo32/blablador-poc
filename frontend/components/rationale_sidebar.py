"""Rationale sidebar component for evidence ranking context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import streamlit as st

from frontend.components import evidence_card

CSS_PATH = evidence_card.ASSET_PATH
SIDEBAR_STATE_KEY = "_evidence_sidebar_state"


def build_progress_summary(candidates: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate entail/contrad/neutral counts for the global progress bar."""
    counts = {"entail": 0, "contradict": 0, "neutral": 0}
    for candidate in candidates or []:
        label = (candidate.get("label") or "unknown").lower()
        if label == "entail":
            counts["entail"] += 1
        elif label in {"contradict", "refute"}:
            counts["contradict"] += 1
        else:
            counts["neutral"] += 1
    total = sum(counts.values())
    denom = total or 1
    return {
        **counts,
        "total": total,
        "entail_pct": counts["entail"] / denom,
        "contradict_pct": counts["contradict"] / denom,
    }


def summarize_rank_delta(
    *,
    delta: Optional[int] = None,
    current_rank: Optional[int] = None,
    previous_rank: Optional[int] = None,
) -> str:
    """Return arrow-style rank delta text for sidebar badges."""
    if delta is None and None not in (current_rank, previous_rank):
        delta = (current_rank or 0) - (previous_rank or 0)
    if delta in (None, 0):
        return "—"
    arrow = "↑" if delta < 0 else "↓"
    return f"{arrow}{abs(int(delta))}"


def build_filter_chip_config(filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Describe filter chip states for entail/contrad/neutral/pinned toggles."""
    state = (filters or {}).copy()
    label = (state.get("label") or "").lower() or None
    include_neutral = state.get("include_neutral", True)
    pinned_only = state.get("pinned_only", False)
    chips: List[Dict[str, Any]] = []
    for slug, text in (
        ("entail", "Entail"),
        ("contradict", "Contradict"),
        ("neutral", "Neutral"),
    ):
        chips.append(
            {
                "key": f"label-{slug}",
                "label": text,
                "active": label == slug,
                "payload": {"label": slug},
            }
        )
    chips.append(
        {
            "key": "neutral-toggle",
            "label": "Neutral +",
            "active": include_neutral,
            "payload": {"include_neutral": not include_neutral},
        }
    )
    chips.append(
        {
            "key": "pinned-only",
            "label": "Pinned",
            "active": pinned_only,
            "payload": {"pinned_only": not pinned_only},
        }
    )
    return chips


@dataclass
class SidebarConfig:
    ui: Any = st


def render_rationale_sidebar(
    claim_state: Dict[str, Any],
    *,
    selected_candidate_id: Optional[str] = None,
    config: SidebarConfig = SidebarConfig(),
) -> None:
    """Render sidebar synchronized with pinned/hovered evidence cards."""
    ui = config.ui
    _inject_styles(ui)
    sidebar_state = _ensure_sidebar_state(claim_state)
    candidates = claim_state.get("candidates") or []
    summary = build_progress_summary(candidates)
    pinned_ids = claim_state.get("pinned_ids") or []
    focus_order = claim_state.get("focus_order") or []
    candidate = _select_candidate(
        candidates,
        selected_candidate_id=selected_candidate_id,
        pinned_ids=pinned_ids,
        focus_order=focus_order,
    )

    ui.subheader("Rationale & ranking feedback")
    _render_progress(ui, summary)
    _render_filters(ui, pinned_ids)

    if not candidate:
        ui.info("Hover or pin an evidence card to see detailed rationale.")
        _render_export_controls(ui, claim_state)
        return

    _render_candidate_snapshot(ui, candidate)
    _render_rationale(ui, candidate)
    _render_advanced(ui, claim_state, sidebar_state)
    _render_export_controls(ui, claim_state)


def _inject_styles(ui: Any) -> None:
    key = "_evidence_sidebar_css"
    session_state = st.session_state
    if session_state.get(key):
        return
    if CSS_PATH.exists():
        ui.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)
    session_state[key] = True


def _ensure_sidebar_state(claim_state: Dict[str, Any]) -> Dict[str, Any]:
    root = st.session_state.setdefault(SIDEBAR_STATE_KEY, {})
    claim_id = claim_state.get("claim_id") or "default"
    return root.setdefault(claim_id, {"advanced": False})


def _select_candidate(
    candidates: Sequence[Dict[str, Any]],
    *,
    selected_candidate_id: Optional[str],
    pinned_ids: Iterable[str],
    focus_order: Iterable[str],
) -> Optional[Dict[str, Any]]:
    lookup = {
        candidate.get("id"): candidate
        for candidate in candidates
        if candidate.get("id")
    }
    fallback = list(candidates)
    for candidate_id in [
        selected_candidate_id,
        *pinned_ids,
        *(focus_order or []),
    ]:
        if candidate_id and candidate_id in lookup:
            return lookup[candidate_id]
    return fallback[0] if fallback else None


def _render_progress(ui: Any, summary: Dict[str, float]) -> None:
    entail = summary["entail"]
    contradict = summary["contradict"]
    neutral = summary["neutral"]
    total = summary["total"]
    ui.markdown(
        f"""
        <div class="evidence-progress" aria-label="Global entail vs contradict">
            <div
                class="evidence-progress__segment evidence-progress__segment--entail"
                style="flex: {entail}"
            ></div>
            <div
                class="evidence-progress__segment
                       evidence-progress__segment--contradict"
                style="flex: {contradict}"
            ></div>
            <div
                class="evidence-progress__segment evidence-progress__segment--neutral"
                style="flex: {neutral}"
            ></div>
        </div>
        <div class="evidence-progress__labels">
            <span>{entail} entail</span>
            <span>{contradict} contradict</span>
            <span>{neutral} neutral</span>
            <span>{total} total</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_filters(ui: Any, pinned_ids: Iterable[str]) -> None:
    pins = list(pinned_ids)
    if not pins:
        return
    label = ", ".join(pins[:4])
    if len(pins) > 4:
        label += "…"
    ui.caption(f"Pinned cards: {label}")


def _render_candidate_snapshot(ui: Any, candidate: Dict[str, Any]) -> None:
    title = candidate.get("title") or "Selected evidence"
    ui.markdown(f"**Inspecting:** {title}")
    meta = candidate.get("metadata") or {}
    location_bits = [meta.get("page"), meta.get("section"), meta.get("provenance")]
    location = " • ".join(str(bit) for bit in location_bits if bit)
    if location:
        ui.caption(location)
    rank = candidate.get("rank")
    delta_badge = summarize_rank_delta(delta=candidate.get("rank_delta"))
    score = candidate.get("confidence")
    ui.metric("Rank", rank if rank is not None else "—", delta_badge)
    if score is not None:
        ui.metric("Confidence score", f"{float(score):.2f}")
    token_scores = candidate.get("token_scores")
    if token_scores:
        ui.line_chart(token_scores, height=120)


def _render_rationale(ui: Any, candidate: Dict[str, Any]) -> None:
    rationale = candidate.get("why_matched") or []
    if rationale:
        ui.markdown("#### Why this ranked here")
        for bullet in rationale:
            ui.markdown(f"- {bullet}")
    diversity = candidate.get("diversity_notes") or []
    if diversity:
        with ui.expander("Diversity / demotion notes", expanded=False):
            for note in diversity:
                ui.write(note)
    explanation = candidate.get("label_explanation")
    if explanation:
        ui.info(explanation)


def _render_advanced(
    ui: Any,
    claim_state: Dict[str, Any],
    sidebar_state: Dict[str, Any],
) -> None:
    advanced_default = bool(sidebar_state.get("advanced"))
    advanced = ui.checkbox("Advanced mode", value=advanced_default)
    if advanced != advanced_default:
        sidebar_state["advanced"] = advanced
        _toast(ui, f"Advanced mode {'enabled' if advanced else 'disabled'}", icon="✨")
    if not advanced:
        return
    ui.markdown("#### Rerank configuration")
    run = claim_state.get("run") or {}
    config_payload = run.get("config") or claim_state.get("last_payload", {}).get(
        "config"
    )
    if config_payload:
        ui.code(json.dumps(config_payload, indent=2, ensure_ascii=False))
    else:
        ui.caption("No advanced configuration reported by the backend.")


def _render_export_controls(ui: Any, claim_state: Dict[str, Any]) -> None:
    payload = claim_state.get("last_payload") or {}
    if payload:
        ui.download_button(
            "Download JSON export",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name=f"evidence-{claim_state.get('claim_id','claim')}.json",
            mime="application/json",
        )
    history = claim_state.get("history") or []
    if history:
        ui.download_button(
            "Download rerun history",
            data=json.dumps(history, indent=2, ensure_ascii=False),
            file_name=f"evidence-history-{claim_state.get('claim_id','claim')}.json",
            mime="application/json",
        )


def _toast(ui: Any, message: str, *, icon: str = "ℹ️") -> None:
    toast = getattr(ui, "toast", None)
    if callable(toast):  # pragma: no cover - streamlit runtime only
        toast(message, icon=icon)
    else:  # pragma: no cover - fallback for tests
        ui.write(f"{icon} {message}")


__all__ = [
    "render_rationale_sidebar",
    "build_progress_summary",
    "summarize_rank_delta",
    "build_filter_chip_config",
]
