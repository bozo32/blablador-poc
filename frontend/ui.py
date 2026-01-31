# frontend/ui.py

# === Imports ===
import html
import json
import hashlib
import os
import pathlib
import re
import sys
import tempfile

import graphviz
import pandas as pd
import requests
import streamlit as st

try:  # Optional dependency for timed polling
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - fallback when package missing
    st_autorefresh = None

st.set_page_config(page_title="Citation-Support Checker", layout="wide")

# Add project root to sys.path so `backend`/`frontend` modules import cleanly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend import attachment_queue, claim_queue, clipboard, evidence_store
from frontend.components.evidence_card import CardActionCallbacks, EvidenceCardRenderer
from frontend.components.rationale_sidebar import (
    SidebarConfig,
    build_filter_chip_config,
    build_progress_summary,
    render_rationale_sidebar,
)
from frontend.evidence_api import MAX_LIST_REQUESTS, show_api_error

from backend import utils
from backend.bl_client import BlabladorClient
from backend.model_cache import add_model, get_models
from backend.settings import AppSettings
from backend.utils import list_local_models
from frontend.ingestion_api import (
    get_citation_context,
    get_citation_graph,
    get_document_body,
    get_document,
    list_documents,
    trigger_extraction,
    trigger_resolution,
    upload_pdf,
)
from typing import Any, Dict, List, Optional


# Third-party


# Local application

# === Settings & State Initialization ===
settings = AppSettings()


def init_session_state():
    """Initialize Streamlit session state keys from settings."""
    defaults = {
        "api_url": settings.BACKEND_URL,
        "api_key": "" if settings.API_KEY == "..." else settings.API_KEY,
        "api_base": settings.API_BASE,
        "embed_model": settings.EMBED_MODEL,
        "max_sentences": settings.MAX_SENTENCES,
        "faiss_min_score": settings.FAISS_MIN_SCORE,
        "reranker_model": settings.RERANKER_MODEL,
        "reranker_top_k": settings.RERANKER_TOP_K,
        "nli_model": settings.NLI_MODEL,
        "nli_threshold": settings.NLI_THRESHOLD,
        "selected_model": settings.DEFAULT_LLM_MODEL,
        "seg_cache": {},
        "results": {},
        "started": False,
        "seg_requested": False,
        "pipeline_mode": getattr(settings, "PIPELINE_MODE", "classic"),
        "ingested_docs": [],
        "selected_doc_id": None,
        "active_document": None,
        "citation_selected_index": None,
        "citation_selected_target": None,
        "citation_context_key": None,
        "citation_context": None,
        "citation_context_error": None,
        "citation_last_context_request": None,
        "citation_follow_open": False,
        "citation_graph_key": None,
        "citation_graph": None,
        "citation_graph_error": None,
        "citation_last_graph_request": None,
        "citation_graph_depth": 1,
        "citation_graph_max_nodes": 10,
        "citation_sentence_segments": {},
        "auto_extract_on_upload": True,
        "auto_resolve_on_upload": True,
        "citation_debug": False,
        "show_demo_claims": False,
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


# === Helpers ===

SEGMENT_ID_CLAIM_RE = re.compile(r"^\s*(\d+[a-z])\.\s*(.*)$", re.I)


def to_segment_dict(seg_line: str) -> dict:
    """Parse a segment line (e.g., '1a.

    climate change...') into a dict for API.
    """
    m = SEGMENT_ID_CLAIM_RE.match(seg_line.strip())
    if not m:
        # fallback: assign dummy segment_id, keep claim
        return {"segment_id": "", "claim": seg_line.strip()}
    segment_id, claim = m.groups()
    return {"segment_id": segment_id.strip(), "claim": claim.strip()}


def model_selector(
    label: str, session_key: str, choices: list[str], allow_custom: bool = True
):
    """Return dropdown with optional custom input."""
    current = st.session_state.get(session_key)
    options = choices.copy()
    if allow_custom:
        options.append("Custom...")
    sel = st.selectbox(
        label,
        options,
        index=options.index(current) if current in options else 0,
    )
    if allow_custom and sel == "Custom...":
        custom = st.text_input(f"Custom {label}", value=current or "")
        if custom:
            st.session_state[session_key] = custom
    else:
        st.session_state[session_key] = sel
    return st.session_state[session_key]


def get_responsive_models():
    """Fetch and return only those Blablador models that actually respond."""
    api_key = st.session_state["api_key"]
    base_url = st.session_state["api_base"]
    if not api_key or not base_url:
        return []
    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        ids = [m["id"] for m in resp.json().get("data", [])]
    except BaseException:
        return []
    valid = []
    for m in ids:
        try:
            test = requests.post(
                f"{base_url.rstrip('/')}/v1/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": m, "prompt": "Test", "max_tokens": 1},
                timeout=5,
            )
            if test.status_code == 200:
                valid.append(m)
        except BaseException:
            pass
    return valid


def reset_segmentation():
    """Clear segmentation flags so you can re-run with new settings."""
    for k in list(st.session_state.keys()):
        if k.startswith("done-"):
            del st.session_state[k]
    st.session_state.started = False
    st.session_state.seg_requested = False
    st.session_state.pop("seg_cache", None)
    # force a fresh FAISS build next time
    st.session_state.pop("faiss_started", None)


# Constants for segmentation helper
SEGMENT_PROMPT_TEMPLATE = (
    "You are an expert at breaking sentences into standalone proposition segments.\n"
    "For example, given a sentence A and B cause 1 and 2 you generate 4 segments:\n"
    "    A causes 1\n"
    "    A causes 2\n"
    "    B causes 1\n"
    "    B causes 2\n"
    "You only work with the words provided in the prompting sentence\n"
    "For example, given the sentence A causes B, you generate:\n"
    "    A causes B\n"
    "You stop segmenting when you run out of words in the original sentence.\n"
    "You always keep modifiers (adjectives or adverbs) with what they modify "
    "(nouns or verbs). For example\n"
    "immersion in cold water or snow causes hypothermia\n"
    "becomes\n"
    "    cold water causes hypothermia\n"
    "    snow causes hypothermia\n\n"
    "Some citing sentences directly mention the source.\n"
    "They may take variants on the form '(author name) found that A causes 1.'.\n"
    "In such cases drop the direct mention of the source so that "
    "the sentence becomes:\n"
    "    A causes 1\n\n"
    "Do not output any explanation, commentary, or extra text. "
    "Only list the segments generated from the original sentence. "
    "Your output must end after the last segment.\n\n"
    "List each segment on its own line, numbered {row_idx}a, {row_idx}b, etc., "
    "continuing alphabetically.\n\n"
    "Sentence:\n"
    "{sentence}\n"
    "Segments:\n"
)

SEG_RE = re.compile(r"^\s*\d+[a-z]\.", re.I)


def seg_via_llm(sentence: str, row_idx: int, model: str) -> list[str]:
    prompt = SEGMENT_PROMPT_TEMPLATE.format(row_idx=row_idx, sentence=sentence)
    actual_model = model  # use the model string chosen in the UI

    client = BlabladorClient(
        api_key=st.session_state.get("api_key", ""),
        base_url=st.session_state.get("api_base", ""),
    )
    try:
        text = client.completion(
            prompt,
            model=actual_model,
            temperature=0,
            max_tokens=256,
        )
    except Exception as e:
        st.error(f"Blablador API error: {e}")
        return []
    # Split and extract numbered segments
    lines = [ln.strip() for ln in text.splitlines()]
    segments = [ln for ln in lines if SEG_RE.match(ln)]
    return segments  # (use your fallback logic as before)


def handle_upload():
    """Save uploaded files to a temporary directory and reset segmentation."""
    files = st.session_state.get("uploaded_files", [])
    if not files:
        return
    tmpdir = tempfile.mkdtemp()
    for f in files:
        path = os.path.join(tmpdir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
    st.session_state.data_dir = tmpdir
    st.session_state.results = {}
    reset_segmentation()
    st.success(f"Loaded {len(files)} files")


def refresh_ingested_docs(show_error: bool = True) -> list[dict]:
    api_url = st.session_state.get("api_url", "http://localhost:8000")
    try:
        documents = list_documents(api_url)
    except RuntimeError as exc:
        if show_error:
            st.error(f"Failed to load ingested PDFs: {exc}")
        return []
    st.session_state["ingested_docs"] = documents
    doc_ids = [doc.get("id") for doc in documents if doc.get("id")]
    current = st.session_state.get("selected_doc_id")
    if doc_ids and current not in doc_ids:
        st.session_state["selected_doc_id"] = doc_ids[0]
    return documents


def load_selected_document(show_error: bool = True) -> Optional[dict]:
    doc_id = st.session_state.get("selected_doc_id")
    if not doc_id:
        st.session_state["active_document"] = None
        return None
    api_url = st.session_state.get("api_url", "http://localhost:8000")
    try:
        document = get_document(api_url, doc_id)
    except RuntimeError as exc:
        if show_error:
            st.error(f"Failed to load document details: {exc}")
        return None
    st.session_state["active_document"] = document
    return document


def handle_pdf_upload():
    files = st.session_state.get("uploaded_pdfs") or []
    if not files:
        return
    api_url = st.session_state.get("api_url", "http://localhost:8000")
    uploaded = []
    with st.spinner("Uploading PDFs..."):
        for file in files:
            try:
                uploaded_doc = upload_pdf(api_url, file)
            except RuntimeError as exc:
                st.error(f"Upload failed for {getattr(file, 'name', 'file')}: {exc}")
                continue
            if uploaded_doc:
                uploaded.append(uploaded_doc)
    if uploaded:
        refresh_ingested_docs(show_error=False)
        last_doc = uploaded[-1]
        if last_doc.get("id"):
            st.session_state["selected_doc_id"] = last_doc["id"]
            st.session_state["active_document"] = last_doc
        st.success(f"Uploaded {len(uploaded)} PDF(s).")
        doc_id = last_doc.get("id")
        if doc_id and st.session_state.get("auto_extract_on_upload"):
            with st.spinner("Running extraction..."):
                try:
                    trigger_extraction(api_url, doc_id)
                    document = load_selected_document(show_error=False)
                    st.session_state["active_document"] = document
                    st.success("Extraction complete.")
                except RuntimeError as exc:
                    st.error(f"Extraction failed: {exc}")
                    return
        if doc_id and st.session_state.get("auto_resolve_on_upload"):
            with st.spinner("Resolving references..."):
                try:
                    trigger_resolution(api_url, doc_id)
                    document = load_selected_document(show_error=False)
                    st.session_state["active_document"] = document
                    st.success("Resolution complete.")
                except RuntimeError as exc:
                    st.warning(_resolution_error_message(exc))


def stringify_value(value: object) -> str | int | float | bool | None:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return value


def _resolution_error_message(exc: Exception) -> str:
    raw = str(exc)
    try:
        payload = json.loads(raw)
        detail = payload.get("detail") or raw
    except Exception:
        detail = raw
    if "404" in detail:
        return (
            "Reference resolution service could not find the cited DOI (404). "
            "Continuing with extracted metadata; retry later if needed."
        )
    return detail


def normalize_records(records: list[dict]) -> list[dict]:
    return [
        {key: stringify_value(val) for key, val in record.items()} for record in records
    ]


def render_skeleton(lines: int = 3) -> None:
    if hasattr(st, "skeleton"):
        for _ in range(lines):
            st.skeleton()
    else:
        st.write("Loading...")


def highlight_callout(text: str | None, callout: str | None) -> str:
    if not text:
        return ""
    if not callout:
        return text
    pattern = re.escape(callout)
    return re.sub(
        pattern,
        (
            '<span style="background-color:#fff2b3; padding:2px 4px; '
            'border-radius:4px;">' + callout + "</span>"
        ),
        text,
        count=1,
    )


def inject_citation_styles() -> None:
    if st.session_state.get("citation_styles_loaded"):
        return
    st.markdown(
        """
        <style>
        .citation-sentence {
            font-size: 0.96rem;
            line-height: 1.6;
        }
        .citation-chip {
            display: inline-block;
            padding: 2px 6px;
            margin: 0 2px;
            border-radius: 10px;
            background: #eef3ff;
            color: #1f3a8a;
            border: 1px solid #c9d8ff;
            font-weight: 600;
            font-size: 0.82rem;
            text-decoration: none;
        }
        .citation-chip:hover {
            background: #dce7ff;
        }
        .citation-chip-link {
            text-decoration: none;
        }
        .citation-chip-selected {
            background: rgba(255, 242, 179, 0.55);
            border-color: rgba(160, 120, 0, 0.25);
        }
        .citation-selected {
            background: rgba(255, 242, 179, 0.35);
            border-radius: 10px;
            padding: 8px 10px;
        }
        .citation-divider {
            height: 1px;
            background: #e6e6e6;
            margin: 12px 0 16px;
        }
        .citation-paragraph {
            margin-bottom: 0.9rem;
            padding: 2px 0;
        }
        .citation-sentence-row {
            display: inline;
        }
        .citation-sentence-selected {
            background: rgba(255, 242, 179, 0.35);
            border-radius: 8px;
            padding: 2px 4px;
        }
        .citation-workflow-rail {
            position: sticky;
            top: 0.75rem;
            max-height: calc(100vh - 2rem);
            overflow: auto;
            padding-right: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["citation_styles_loaded"] = True


def _sentence_anchor(sentence: str) -> str:
    digest = hashlib.sha1(sentence.encode("utf-8")).hexdigest()[:10]
    return f"cite-sent-{digest}"


def _sentence_label(sentence: str, words: int = 4) -> str:
    tokens = [tok for tok in (sentence or "").strip().split() if tok]
    if not tokens:
        return "(empty sentence)"
    head = " ".join(tokens[:words])
    suffix = "…" if len(tokens) > words else ""
    return f"{head}{suffix}"


def _citation_href(
    doc_id: Optional[str],
    citation_index: int,
    target_id: Optional[str],
    anchor: str,
) -> str:
    target = normalize_target_id(target_id)
    base = (
        f"?doc={doc_id}&cite={citation_index}" if doc_id else f"?cite={citation_index}"
    )
    if target:
        return f"{base}&target={target}#{anchor}"
    return f"{base}#{anchor}"


def normalize_callout_text(callout: str) -> str:
    text = (callout or "").strip()
    if not text:
        return ""
    text = text.strip("()[]")
    text = re.sub(r"\s*(,|;)?\s*(and|&)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*(,|;)?\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_target_id(target_id: Optional[str]) -> Optional[str]:
    if not target_id:
        return None
    normalized = str(target_id).strip()
    return normalized or None


def split_callout_suffix(callout: str) -> tuple[str, str]:
    raw = (callout or "").strip()
    if not raw:
        return "", ""
    suffix = ""
    match = re.search(r"\s*(and|&)\s*$", raw, flags=re.IGNORECASE)
    if match:
        suffix = " and"
        raw = raw[: match.start()].strip()
    if raw.endswith(")") or raw.endswith("]"):
        suffix = raw[-1] + suffix
        raw = raw[:-1].rstrip()
    return raw, suffix


def format_callout(callout: str) -> dict:
    raw = (callout or "citation").strip()
    base_text, suffix = split_callout_suffix(raw)
    match_text = normalize_callout_text(base_text) or base_text or raw
    tokens = re.findall(r"\d+[a-zA-Z]?", match_text)
    has_letters = bool(re.search(r"[A-Za-z]", match_text))
    if has_letters:
        display = match_text
    elif tokens:
        display = f"({', '.join(tokens)})"
    else:
        display = match_text
    return {
        "raw": raw,
        "display": display,
        "tokens": tokens,
        "match": match_text,
        "suffix": suffix,
    }


def render_sentence_with_callouts(
    sentence: str, callouts: list[dict]
) -> tuple[str, list[dict]]:
    rendered = sentence
    unmatched = []
    for callout in sorted(
        callouts, key=lambda item: len(item.get("raw", "")), reverse=True
    ):
        raw = callout.get("raw") or ""
        display = callout.get("display") or "citation"
        match_text = callout.get("match") or raw
        tokens = callout.get("tokens") or []
        suffix = callout.get("suffix") or ""
        if not match_text:
            continue
        chip = f'<span class="citation-chip">{display}</span>{suffix}'

        tokens_lower = [token.lower() for token in tokens]
        if tokens_lower:
            for bracket_match in re.finditer(r"[\[(][^\])]+[\])]", rendered):
                segment = bracket_match.group(0)
                segment_tokens = [
                    token.lower() for token in re.findall(r"\d+[a-zA-Z]?", segment)
                ]
                if segment_tokens and all(
                    token in segment_tokens for token in tokens_lower
                ):
                    start, end = bracket_match.span()
                    rendered = f"{rendered[:start]}{chip}{rendered[end:]}"
                    break
            else:
                segment = None
        else:
            segment = None

        if tokens_lower and segment is not None:
            continue
        if tokens:
            joined = r"\s*,\s*".join(re.escape(token) for token in tokens)
            pattern = re.compile(rf"[\[(]?\s*{joined}\s*[\])]?")
        else:
            normalized = re.sub(r"\s+", " ", match_text)
            escaped = re.escape(normalized).replace(r"\ ", r"\\s*")
            pattern = re.compile(escaped)

        if pattern.search(rendered):
            rendered = pattern.sub(chip, rendered, count=1)
            continue

        matches = list(re.finditer(r"\d+[a-zA-Z]?", rendered))
        if tokens_lower and matches:
            match_tokens = [m.group(0).lower() for m in matches]
            for idx in range(len(match_tokens) - len(tokens_lower) + 1):
                if match_tokens[idx : idx + len(tokens_lower)] == tokens_lower:
                    start = matches[idx].start()
                    end = matches[idx + len(tokens_lower) - 1].end()
                    rendered = f"{rendered[:start]}{chip}{rendered[end:]}"
                    break
            else:
                unmatched.append(callout)
        else:
            unmatched.append(callout)
    return rendered, unmatched


def _render_sentence_with_citation_chips(
    sentence: str,
    *,
    doc_id: Optional[str],
    citation_indices: list[int],
    anchor: str,
    target_id: Optional[str],
) -> str:
    """Replace in-text citation spans with a single clickable chip.

    Uses heuristic sentence patterns (parenthetical/bracket citations). The chip
    click selects the first citation index for the span.
    """
    if not sentence:
        return ""

    href = _citation_href(
        doc_id,
        citation_indices[0] if citation_indices else 0,
        target_id,
        anchor,
    )

    def make_chip(label: str) -> str:
        cleaned = re.sub(r"\s+", " ", (label or "").strip())
        if not cleaned:
            cleaned = "citation"
        return (
            f'<a class="citation-chip citation-chip-link" href="{html.escape(href)}">'
            f"{html.escape(cleaned)}"
            "</a>"
        )

    # Replace citation clusters like (Author 2018; Author2 2022) or [12, 13].
    span_re = re.compile(r"\([^\)]{2,240}\)|\[[^\]]{2,240}\]")
    spans: list[tuple[int, int, str]] = []
    for match in span_re.finditer(sentence):
        span = match.group(0)
        lowered = span.lower()
        looks_like_citation = (
            any(char.isdigit() for char in lowered)
            or ";" in lowered
            or "et al" in lowered
        )
        if looks_like_citation:
            spans.append((match.start(), match.end(), span))

    if not spans:
        return (
            f"{html.escape(sentence)} {make_chip('citation')}"
            if citation_indices
            else html.escape(sentence)
        )

    chunks: list[str] = []
    cursor = 0
    for start, end, label in spans:
        if start > cursor:
            chunks.append(html.escape(sentence[cursor:start]))
        chunks.append(make_chip(label))
        cursor = end
    if cursor < len(sentence):
        chunks.append(html.escape(sentence[cursor:]))
    return "".join(chunks)


def _render_sentence_with_citation_cluster(
    sentence: str,
    callouts: list[dict],
    *,
    href: Optional[str] = None,
) -> str:
    """Render a sentence with a single in-text citation chip.

    This is a UI helper for cases where TEI sentence nodes are very long
    (sometimes paragraph-like) and where multiple citations appear together.
    We collapse the citation span into a single chip instead of rendering one
    chip per callout.
    """
    if not sentence:
        return ""
    if not callouts:
        return html.escape(sentence)

    chip_inner = "citations"
    if href:
        chip = (
            f'<a class="citation-chip citation-chip-link" href="{html.escape(href)}">'
            f"{chip_inner}</a>"
        )
    else:
        chip = f'<span class="citation-chip">{chip_inner}</span>'

    spans: list[tuple[int, int]] = []
    for callout in callouts:
        for needle in (callout.get("raw"), callout.get("match")):
            if not needle:
                continue
            normalized = re.sub(r"\s+", " ", str(needle)).strip()
            if not normalized:
                continue
            pattern = re.escape(normalized).replace(r"\ ", r"\\s+")
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                spans.append(match.span())
                break

    if not spans:
        base = html.escape(sentence)
        return f"{base} {chip}" if callouts else base

    spans.sort(key=lambda item: item[0])
    cluster_start, cluster_end = spans[0]
    for start, end in spans[1:]:
        if start <= cluster_end + 10:
            cluster_end = max(cluster_end, end)
        else:
            break

    # Expand to include surrounding brackets when present.
    if cluster_start > 0 and sentence[cluster_start - 1] in "([":
        opener = sentence[cluster_start - 1]
        closer = ")" if opener == "(" else "]"
        if cluster_end < len(sentence) and sentence[cluster_end] == closer:
            cluster_start -= 1
            cluster_end += 1

    focus_sentence = sentence
    focus_offset = 0
    if len(sentence) > 480:
        before = sentence[:cluster_start]
        after = sentence[cluster_end:]
        left_boundary = 0
        for m in re.finditer(r"[.!?]\s", before):
            left_boundary = m.end()
        right_boundary = len(sentence)
        m_after = re.search(r"\s[.!?]", after)
        if m_after:
            right_boundary = cluster_end + m_after.start() + 2
        focus_sentence = sentence[left_boundary:right_boundary].strip()
        focus_offset = left_boundary

    rel_start = max(0, cluster_start - focus_offset)
    rel_end = max(rel_start, cluster_end - focus_offset)
    return (
        html.escape(focus_sentence[:rel_start])
        + chip
        + html.escape(focus_sentence[rel_end:])
    )


# === Attachment Helpers ===
ATTACHMENT_CSS_PATH = (
    pathlib.Path(__file__).resolve().parent / "assets" / "attachment_panel.css"
)
ATTACHMENT_STATUS_LABELS = {
    "pending": "Pending",
    "converting": "Converting",
    "parsing": "Parsing",
    "matched": "Matched",
    "error": "Error",
}


def inject_attachment_panel_styles() -> None:
    """Load CSS for the attachment queue workspace."""
    if st.session_state.get("attachment_panel_styles_loaded"):
        return
    if ATTACHMENT_CSS_PATH.exists():
        st.markdown(
            f"<style>{ATTACHMENT_CSS_PATH.read_text()}</style>",
            unsafe_allow_html=True,
        )
    st.session_state["attachment_panel_styles_loaded"] = True


def prepare_attachment_workspace() -> None:
    """Ensure attachment queue state and claim registry are hydrated."""
    attachment_queue.init_attachment_queue_state()
    claim_queue.sync_claims_from_results(st.session_state.get("results"))
    if not claim_queue.get_claim_records() and st.session_state.get("show_demo_claims"):
        claim_queue.ensure_demo_claims()
    attachment_queue.sync_backend_state()
    attachment_queue.ensure_open_when_activity()
    attachment_queue.collapse_when_idle()


def render_attachment_workspace() -> None:
    """Render claim drop zones, modal fallback, and queue panel."""
    inject_attachment_panel_styles()
    prepare_attachment_workspace()
    if attachment_queue.has_inflight_jobs():
        if st_autorefresh:
            st_autorefresh(interval=5000, key="attachment-autopoll")
        else:
            st.caption(
                "Attachments are processing in the background. "
                "Use the queue panel to refresh statuses."
            )
    st.subheader("Evidence attachments & queue")
    st.caption(
        "Drop cited PDFs onto claims, then monitor their status in the queue panel."
    )
    render_claim_cards_grid()
    render_attachment_modal()
    render_attachment_queue_panel()


def render_claim_cards_grid() -> None:
    claims = claim_queue.get_claim_records()
    if not claims:
        st.info(
            "No claims available yet. Run segmentation or use the demo claims to"
            " prototype the attachment flow."
        )
        return
    active_target = attachment_queue.get_active_drop_target()
    column_count = min(3, max(1, len(claims)))
    columns = st.columns(column_count)
    for idx, claim in enumerate(claims):
        column = columns[idx % column_count]
        with column:
            render_claim_card(claim, active_target)


def render_claim_card(claim: dict, active_target: Optional[str]) -> None:
    claim_id = claim.get("id") or f"claim-{hash(claim.get('claim', 'claim'))}"
    classes = ["attachment-card"]
    if active_target and active_target != claim_id:
        classes.append("attachment-card--dimmed")
    if active_target == claim_id:
        classes.append("attachment-card--focused")
    class_attr = " ".join(classes)
    st.markdown(
        f'<div class="{class_attr}" data-claim-id="{claim_id}">',
        unsafe_allow_html=True,
    )
    claim_title = html.escape(claim.get("claim", "Untitled claim"))
    st.markdown(
        f"<p class='attachment-claim-title'>{claim_title}</p>",
        unsafe_allow_html=True,
    )
    callout = claim.get("callout") or "Unlabeled citation"
    st.caption(f"{callout} • Claim ID: {claim_id}")
    dropzone_html = """
    <div class="attachment-dropzone">
        <strong>Drop PDF</strong>
        <span>Supports drag, drop, or keyboard navigation</span>
    </div>
    """
    st.markdown(dropzone_html, unsafe_allow_html=True)
    drop_key = f"attachment-drop-{claim_id}"
    uploaded = st.file_uploader(
        f"Drop PDF for {callout}",
        type=["pdf"],
        accept_multiple_files=True,
        key=drop_key,
        label_visibility="collapsed",
    )
    if uploaded:
        attachment_queue.handle_drop(
            claim_id,
            uploaded,
            doc_id=claim.get("doc_id"),
            reference_hint=_build_reference_hint_from_claim(claim),
            source="dropzone",
        )
        attachment_queue.clear_active_drop_target()
        st.session_state.pop(drop_key, None)
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Attach via modal", key=f"open-modal-{claim_id}"):
            attachment_queue.open_attachment_modal(claim_id)
    with action_cols[1]:
        if active_target == claim_id:
            if st.button("Clear highlight", key=f"clear-highlight-{claim_id}"):
                attachment_queue.clear_active_drop_target()
        else:
            if st.button("Highlight drop zone", key=f"focus-highlight-{claim_id}"):
                attachment_queue.set_active_drop_target(claim_id)
    attached = attachment_queue.get_claim_attachment(claim_id)
    if attached:
        status = attached.get("status", "pending")
        pill = ATTACHMENT_STATUS_LABELS.get(status, status.title())
        st.markdown(
            f"<span class='attachment-status-pill {status}'>{pill}</span>",
            unsafe_allow_html=True,
        )
        st.caption(_describe_attachment(attached, claim))
        if status in {"pending", "converting", "parsing"}:
            render_skeleton(1)
        if st.button("Detach file", key=f"detach-{claim_id}"):
            attachment_queue.detach_attachment(claim_id, attached.get("id"))
    else:
        st.caption("No attachment assigned yet.")
    timeline = (attached or {}).get("history") or claim_queue.get_timeline(claim_id)
    if timeline:
        with st.expander("Attachment timeline", expanded=False):
            for entry in timeline:
                detail = entry.get("detail")
                if isinstance(detail, dict):
                    summary = ", ".join(
                        f"{k}: {v}" for k, v in detail.items() if v is not None
                    )
                elif isinstance(detail, str):
                    summary = detail.strip()
                else:
                    summary = ""
                event_label = entry.get("event", "update").title()
                detail_text = summary or "No detail"
                timestamp = entry.get("at") or "unknown"
                st.markdown(f"- **{event_label}** — {detail_text} ({timestamp})")
    st.caption("Use the modal button for keyboard-only uploads.")
    st.markdown("</div>", unsafe_allow_html=True)


def _build_reference_hint_from_claim(claim: dict) -> dict:
    return {
        "callout": claim.get("callout"),
        "reference_id": claim.get("reference_id") or claim.get("id"),
        "author": claim.get("author"),
        "year": claim.get("year"),
        "doi": (claim.get("reference_hint") or {}).get("doi"),
    }


def _describe_attachment(item: dict, claim: Optional[dict]) -> str:
    filename = item.get("filename")
    size = format_filesize(item.get("size"))
    claim_text = (claim or {}).get("claim")
    target = (
        claim_text[:42] + "…" if claim_text and len(claim_text) > 45 else claim_text
    )
    status = item.get("status", "pending")
    label = ATTACHMENT_STATUS_LABELS.get(status, status.title())
    base = f"{filename} ({size}) • {label}"
    if target:
        return f"{base} for {target}"
    return base


def format_filesize(num_bytes: Optional[int]) -> str:
    if not num_bytes:
        return "unknown"
    if num_bytes < 1024:
        return f"{num_bytes} B"
    kb = num_bytes / 1024
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024
    return f"{mb:.1f} MB"


def _render_queue_summary(summary: dict) -> None:
    status_line = _format_queue_summary(summary)
    if status_line:
        st.caption(status_line)
    converting = summary.get("converting", 0)
    pending = summary.get("pending", 0)
    if converting:
        st.info(
            (
                f"{converting} attachment{'s' if converting != 1 else ''} "
                "converting before parsing completes."
            ),
            icon="⏳",
        )
    elif pending:
        st.caption(
            "Pending uploads will move into converting automatically as soon as "
            "preprocessing starts."
        )


def _format_queue_summary(summary: dict) -> str:
    ordered = ["pending", "converting", "parsing", "matched", "error"]
    parts = []
    for status in ordered:
        label = ATTACHMENT_STATUS_LABELS.get(status, status.title())
        count = summary.get(status, 0)
        if status == "error" and count == 0:
            continue
        parts.append(f"{label}: {count}")
    return " • ".join(parts)


def _claim_option_index(options: List[dict], claim_id: Optional[str]) -> int:
    for idx, option in enumerate(options):
        if option.get("id") == claim_id:
            return idx
    return 0


def render_attachment_modal() -> None:
    claim_id = attachment_queue.get_modal_claim_id()
    if not claim_id:
        return
    claim = claim_queue.get_claim_record(claim_id) or {}
    st.markdown('<div class="attachment-modal">', unsafe_allow_html=True)
    modal_claim_title = html.escape(claim.get("claim", "Untitled claim"))
    st.markdown(
        f"<strong>Keyboard-friendly uploader</strong><br>Claim: {modal_claim_title}",
        unsafe_allow_html=True,
    )
    modal_files = st.file_uploader(
        "Attach PDF via modal",
        type=["pdf"],
        accept_multiple_files=True,
        key="attachment-modal-uploader",
    )
    if modal_files:
        attachment_queue.handle_drop(
            claim_id,
            modal_files,
            doc_id=claim.get("doc_id"),
            reference_hint=_build_reference_hint_from_claim(claim),
            source="modal",
        )
        st.session_state["attachment-modal-uploader"] = None
    if st.button("Close modal", key="close-attachment-modal"):
        attachment_queue.close_attachment_modal()
    st.markdown("</div>", unsafe_allow_html=True)


def render_attachment_queue_panel() -> None:
    snapshot = attachment_queue.get_queue_snapshot()
    summary_label = attachment_queue.summarize_chip_label()
    if not snapshot.get("panel_open"):
        if attachment_queue.has_inflight_jobs():
            st.info(
                "Attachments are still processing in the background. "
                "Open the queue to monitor progress."
            )
        if st.button(
            summary_label,
            key="queue-summary-chip",
            type="secondary",
            help="Reopen the queue panel",
        ):
            attachment_queue.toggle_panel(True)
        return
    st.markdown("### Attachment queue panel")
    _render_queue_summary(snapshot.get("summary", {}))
    if st.button("Collapse queue panel", key="queue-collapse"):
        attachment_queue.toggle_panel(False)
        return
    items = attachment_queue.get_queue_items()
    if not items:
        st.info("Queue is empty. Drop a PDF from any claim to populate the panel.")
        return
    attention_needed = sum(1 for entry in items if entry.get("ambiguous_matches"))
    if attention_needed:
        st.warning(
            (
                f"{attention_needed} attachment{'s' if attention_needed != 1 else ''} "
                "need manual assignment."
            ),
            icon="⚠",
        )
    for item in items:
        render_queue_item(item)


def render_queue_item(item: dict) -> None:
    status = item.get("status", "pending")
    pill = ATTACHMENT_STATUS_LABELS.get(status, status.title())
    claim = claim_queue.get_claim_record(item.get("claim_id"))
    st.markdown(
        f'<div class="queue-item queue-item--{status}">',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='queue-item-header'><strong>{item.get('filename')}</strong>"
        f"<span class='attachment-status-pill {status}'>{pill}</span></div>",
        unsafe_allow_html=True,
    )
    source_label = item.get("source", "dropzone")
    size_label = format_filesize(item.get("size"))
    st.caption(f"Source: {source_label} • Size: {size_label}")
    if claim:
        st.caption(f"Attached to: {claim.get('claim')}")
    ambiguous = item.get("ambiguous_matches") or []
    needs_assignment = ambiguous or not item.get("claim_id")
    if ambiguous:
        suggestion = ambiguous[0]
        suggestion_label = (
            suggestion.get("callout") or suggestion.get("id") or "top suggestion"
        )
        suggestion_score = suggestion.get("score")
        st.warning(
            (
                "Multiple possible claims detected. Review the suggestion or "
                "choose manually."
            ),
            icon="⚠",
        )
        if suggestion_score is not None:
            st.caption(
                f"Top suggestion: {suggestion_label} (score {suggestion_score:.2f})"
            )
        else:
            st.caption(f"Top suggestion: {suggestion_label}")
        if suggestion.get("id") and st.button(
            f"Accept {suggestion_label}",
            key=f"queue-accept-{item['id']}",
        ):
            attachment_queue.attach_to_claim(
                suggestion["id"],
                item["id"],
                via="auto-suggestion",
                score=suggestion.get("score"),
            )
            st.experimental_rerun()
    if needs_assignment:
        options = claim_queue.get_claim_options()
        if options:
            default_idx = _claim_option_index(options, item.get("claim_id"))
            selected_option = st.selectbox(
                "Assign to claim",
                options,
                index=default_idx,
                format_func=lambda option: option["label"],
                key=f"queue-select-{item['id']}",
            )
            if st.button("Assign to claim", key=f"queue-assign-{item['id']}"):
                attachment_queue.attach_to_claim(
                    selected_option["id"], item["id"], via="manual"
                )
                st.experimental_rerun()
        else:
            st.info(
                "No claims available yet. Generate claims before assigning attachments."
            )
    if status == "error":
        if st.button("Retry parse", key=f"queue-retry-{item['id']}"):
            attachment_queue.retry_attachment(item["id"])
            st.experimental_rerun()
    history = item.get("history") or []
    if history:
        with st.expander("Timeline", expanded=False):
            for event in history:
                detail = event.get("detail") or ""
                st.markdown(
                    f"- {event.get('at')}: {event.get('event')} {detail}".strip()
                )
    backend_details = item.get("backend_details")
    if backend_details:
        with st.expander("Diagnostics", expanded=False):
            st.json(backend_details)
    st.markdown("</div>", unsafe_allow_html=True)


def render_evidence_panel() -> None:
    st.subheader("Ranked evidence preview")
    claims = claim_queue.get_claim_records()
    if not claims:
        st.info("Run segmentation to populate claims before ranking evidence.")
        return
    options = claim_queue.get_claim_options()
    claim_ids = [opt.get("id") for opt in options if opt.get("id")]
    if not claim_ids:
        st.info("Claims are missing identifiers; rerun segmentation if needed.")
        return
    label_map = {opt["id"]: opt["label"] for opt in options if opt.get("id")}
    store = evidence_store.EvidenceStore()
    active_claim = store.active_claim_id() or claim_ids[0]
    selected_claim = st.selectbox(
        "Focus claim",
        claim_ids,
        index=claim_ids.index(active_claim) if active_claim in claim_ids else 0,
        format_func=lambda cid: label_map.get(cid, cid),
        key="evidence-claim-select",
    )
    if selected_claim != active_claim:
        claim_queue.set_active_claim(selected_claim)
        store = evidence_store.EvidenceStore()
    claim_record = claim_queue.get_claim_record(selected_claim) or {}
    claim_text_override = (claim_record.get("claim") or "").strip()
    initial_claim_text = claim_text_override or None
    state = store.sync_for_claim(
        selected_claim,
        claim_text=initial_claim_text,
    )
    metadata_claim_text = (state.get("metadata") or {}).get("claim_text") or ""
    active_claim_text = (metadata_claim_text or claim_text_override).strip()
    active_claim_text_payload = active_claim_text or None
    rerun_state = state.get("rerun", {})
    summary = build_progress_summary(state.get("candidates") or [])
    layout_main, layout_sidebar = st.columns([2, 1], gap="large")

    def _claim_text_missing_error() -> Optional[str]:
        last_error = state.get("last_error") or ""
        if not last_error:
            return None
        normalized = last_error.lower()
        if "claim_text" in normalized:
            return last_error
        return None

    def _build_action_callback(action: str):
        def _callback(candidate: Dict[str, Any]) -> None:
            candidate_id = candidate.get("id")
            if not candidate_id:
                return
            if action == "accept":
                store.accept_candidate(selected_claim, candidate_id)
            elif action == "reject":
                store.reject_candidate(selected_claim, candidate_id)
            elif action == "pin":
                store.toggle_pin(selected_claim, candidate_id)
            elif action == "share":
                store.prepare_share_link(selected_claim, candidate_id)
            elif action == "open":
                store.open_candidate_pdf(selected_claim, candidate_id)

        return _callback

    with layout_main:
        callbacks = CardActionCallbacks(
            accept=_build_action_callback("accept"),
            reject=_build_action_callback("reject"),
            pin=_build_action_callback("pin"),
            share=_build_action_callback("share"),
            open_pdf=_build_action_callback("open"),
        )
        renderer = EvidenceCardRenderer(ui=layout_main, callbacks=callbacks)

        def _render_claim_header() -> None:
            claim_text = (claim_record.get("claim") or "Untitled claim").strip()
            callout = claim_record.get("callout") or "Unlabeled citation"
            metadata = claim_record.get("metadata") or state.get("metadata") or {}
            unsaved = metadata.get("unsaved_edits") or metadata.get(
                "has_unsaved_changes"
            )
            badge = "Unsaved edits" if unsaved else "Synced"
            badge_color = "#f97316" if unsaved else "#10b981"
            st.markdown(
                f"""
                <div class=\"claim-reminder\">
                    <div><strong>{callout}</strong></div>
                    <div>{claim_text}</div>
                    <span
                        class=\"claim-reminder__badge\"
                        style=\"background:{badge_color};\"
                    >
                        {badge}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        def _render_status_messages() -> None:
            if state.get("stale"):
                st.warning(
                    state.get("stale_reason") or "Ranking looks stale. Request a rerun."
                )
            if state.get("last_error"):
                st.error(f"Evidence fetch failed: {state['last_error']}")
            missing_claim_text_reason = _claim_text_missing_error()
            if missing_claim_text_reason:
                # Remediation flow: surface the missing claim_text warning so reviewers
                # can resend the cached or edited text before rerunning evidence.
                st.warning(
                    (
                        "Evidence fetch needs the claim text before it can run. "
                        "Press 'Send claim text' to resubmit the cached text."
                    ),
                    icon="⚠️",
                )
                if st.button(
                    "Send claim text",
                    key=f"send-claim-text-{selected_claim}",
                    disabled=not active_claim_text_payload,
                ):
                    store.sync_for_claim(
                        selected_claim,
                        claim_text=active_claim_text_payload,
                        force=True,
                    )
                    st.experimental_rerun()
            status = rerun_state.get("status")
            if status in {"queued", "running"}:
                st.info("Evidence rerun in progress…", icon="🔁")
            history = state.get("history") or []
            if history:
                latest = history[0]
                log_url = latest.get("log_url") or latest.get("artifact_path")
                if log_url:
                    st.caption(f"Latest rerun logs: {log_url}")

        def _render_filter_chips() -> None:
            chips = build_filter_chip_config(state.get("filters"))
            chip_cols = st.columns(len(chips))
            for column, chip in zip(chip_cols, chips):
                button_type = "primary" if chip["active"] else "secondary"
                if column.button(
                    chip["label"],
                    key=f"filter-{chip['key']}-{selected_claim}",
                    type=button_type,
                ):
                    store.apply_filter(
                        selected_claim,
                        claim_text=active_claim_text_payload,
                        **chip["payload"],
                    )
                    st.experimental_rerun()

        def _render_rerun_controls() -> None:
            meta = (state.get("last_payload") or {}).get("meta") or {}
            remaining = meta.get("remaining_candidates", 0)
            inflight = state.get("inflight_fetches", 0)
            btn_cols = st.columns([1, 1, 1])
            with btn_cols[0]:
                disabled = rerun_state.get("inflight") or state.get("is_loading")
                if st.button(
                    "Request rerun",
                    key=f"rerun-{selected_claim}",
                    disabled=disabled,
                ):
                    store.queue_rerun(
                        selected_claim,
                        claim_text=active_claim_text_payload,
                        note="manual rerun",
                    )
                    st.experimental_rerun()
            with btn_cols[1]:
                load_disabled = (
                    state.get("is_loading")
                    or inflight >= MAX_LIST_REQUESTS
                    or remaining <= 0
                )
                label = (
                    f"Load more ({max(0, remaining)})"
                    if remaining not in (None, 0)
                    else "Load more"
                )
                if st.button(
                    label,
                    key=f"load-more-{selected_claim}",
                    disabled=load_disabled,
                ):
                    store.load_more(
                        selected_claim,
                        claim_text=active_claim_text_payload,
                    )
                    st.experimental_rerun()
            with btn_cols[2]:
                if st.button(
                    "Refresh list",
                    key=f"refresh-{selected_claim}",
                    disabled=state.get("is_loading"),
                ):
                    store.sync_for_claim(
                        selected_claim,
                        claim_text=active_claim_text_payload,
                        force=True,
                    )
                    st.experimental_rerun()
            with st.expander("Advanced rerun controls", expanded=False):
                note = st.text_input(
                    "Optional note",
                    key=f"rerun-note-{selected_claim}",
                )
                advanced_blob = st.text_area(
                    "Advanced settings (JSON)",
                    key=f"rerun-advanced-{selected_claim}",
                    height=120,
                )
                if st.button(
                    "Queue rerun with settings",
                    key=f"advanced-rerun-{selected_claim}",
                    disabled=rerun_state.get("inflight"),
                ):
                    payload = {}
                    if advanced_blob.strip():
                        try:
                            payload = json.loads(advanced_blob)
                        except json.JSONDecodeError as exc:
                            st.error(f"Advanced settings JSON invalid: {exc}")
                            payload = None
                    if payload is not None:
                        store.queue_rerun(
                            selected_claim,
                            claim_text=active_claim_text_payload,
                            note=note or None,
                            advanced_settings=payload,
                        )
                        st.experimental_rerun()

        def _render_progress_glance() -> None:
            st.markdown(
                f"""
                <div
                    class=\"evidence-progress\"
                    aria-label=\"Global entail vs contradict\"
                >
                    <div
                        class=\"evidence-progress__segment
                               evidence-progress__segment--entail\"
                        style=\"flex:{summary['entail']}\"
                    ></div>
                    <div
                        class=\"evidence-progress__segment
                               evidence-progress__segment--contradict\"
                        style=\"flex:{summary['contradict']}\"
                    ></div>
                    <div
                        class=\"evidence-progress__segment
                               evidence-progress__segment--neutral\"
                        style=\"flex:{summary['neutral']}\"
                    ></div>
                </div>
                <div class=\"evidence-progress__labels\">
                    <span>{summary['entail']} entail</span>
                    <span>{summary['contradict']} contradict</span>
                    <span>{summary['neutral']} neutral</span>
                    <span>{summary['total']} total</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        def _render_evidence_lists() -> None:
            candidates = state.get("candidates") or []
            if not candidates:
                st.info(
                    "Evidence will appear once attachments finish matching this claim."
                )
                return
            filter_state = state.get("filters", {})
            pinned_ids = state.get("pinned_ids") or []
            pinned_set = set(pinned_ids)
            pinned_only = filter_state.get("pinned_only", False)
            include_neutral = filter_state.get("include_neutral", True)
            if pinned_only:
                pinned_candidates = candidates
                primary_candidates: list[dict] = []
                neutral_candidates: list[dict] = []
            else:
                pinned_candidates = [c for c in candidates if c.get("id") in pinned_set]
                remaining = [c for c in candidates if c.get("id") not in pinned_set]
                neutral_candidates = [
                    c
                    for c in remaining
                    if (c.get("label") or "").lower() not in {"entail", "contradict"}
                ]
                primary_candidates = [
                    c for c in remaining if c not in neutral_candidates
                ]
            if pinned_candidates:
                st.markdown("#### Pinned")
                renderer.render_cards(
                    selected_claim,
                    pinned_candidates,
                    pinned_ids=pinned_ids,
                    lock_state=state.get("lock_state"),
                )
            if primary_candidates:
                st.markdown("#### Ranked evidence")
                renderer.render_cards(
                    selected_claim,
                    primary_candidates,
                    pinned_ids=pinned_ids,
                    lock_state=state.get("lock_state"),
                )
            if neutral_candidates and include_neutral:
                with st.expander(
                    f"Neutral candidates ({len(neutral_candidates)})",
                    expanded=False,
                ):
                    renderer.render_cards(
                        selected_claim,
                        neutral_candidates,
                        pinned_ids=pinned_ids,
                        lock_state=state.get("lock_state"),
                    )

        def _render_share_panel() -> None:
            share_state = state.get("share_target")
            if not share_state:
                return
            st.markdown("#### Share evidence card")
            clipboard.render_copy_to_clipboard(
                "Copy share payload",
                share_state.get("payload"),
                key=f"share-{selected_claim}",
                toast="Evidence payload copied",
            )
            if st.button(
                "Clear share context",
                key=f"clear-share-{selected_claim}",
            ):
                state["share_target"] = None
                st.experimental_rerun()

        def _render_pdf_notice() -> None:
            pdf_jump = state.get("pdf_jump")
            if not pdf_jump:
                return
            viewer = pdf_jump.get("viewer") or {}
            page_label = viewer.get("page") or viewer.get("page_number") or "?"
            fragment = viewer.get("fragment") or viewer.get("coordinates") or ""
            st.info(
                f"PDF span ready on page {page_label}. {fragment}",
                icon="📄",
            )
            if st.button(
                "Clear PDF hint",
                key=f"clear-pdf-{selected_claim}",
            ):
                state["pdf_jump"] = None
                st.experimental_rerun()

        _render_claim_header()
        _render_status_messages()
        _render_filter_chips()
        _render_rerun_controls()
        _render_progress_glance()
        _render_evidence_lists()
        _render_share_panel()
        _render_pdf_notice()

    focus_order = state.get("focus_order") or []
    with layout_sidebar:
        render_rationale_sidebar(
            state,
            selected_candidate_id=focus_order[0] if focus_order else None,
            config=SidebarConfig(ui=layout_sidebar),
        )


def format_reference_summary(reference: dict, resolution: dict) -> str:
    title = (resolution or {}).get("title") or (reference or {}).get("raw_reference")
    year = (resolution or {}).get("year")
    doi = (resolution or {}).get("doi") or (reference or {}).get("doi")
    parts = []
    if title:
        parts.append(title)
    if year:
        parts.append(f"({year})")
    summary = " ".join(parts).strip()
    if doi:
        summary = (
            f"{summary} DOI: https://doi.org/{doi}"
            if summary
            else f"DOI: https://doi.org/{doi}"
        )
    return summary


def format_grobid_authors(authors: object) -> str:
    if isinstance(authors, str):
        return authors.strip()
    if isinstance(authors, list):
        names = []
        for author in authors:
            if isinstance(author, str):
                name = author.strip()
            elif isinstance(author, dict):
                name = (
                    author.get("full_name")
                    or author.get("name")
                    or " ".join(
                        part
                        for part in [
                            author.get("given_name") or author.get("first_name"),
                            author.get("surname") or author.get("last_name"),
                        ]
                        if part
                    ).strip()
                )
            else:
                name = ""
            if name:
                names.append(name)
        return "; ".join(names)
    return ""


def select_grobid_value(grobid: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = grobid.get(key)
        if value:
            return str(value).strip()
    return None


def build_citing_bibliography_summary(
    reference: dict, resolution: dict
) -> tuple[str, bool]:
    grobid = (reference or {}).get("grobid") or (resolution or {}).get("grobid") or {}
    title = select_grobid_value(grobid, ["title", "title_full", "title_main"])
    year = select_grobid_value(grobid, ["year", "published_year", "date"])
    authors = format_grobid_authors(grobid.get("authors") or grobid.get("author"))
    has_consolidated = bool(title or year or authors)
    if has_consolidated:
        prefix_parts = []
        if authors:
            prefix_parts.append(authors)
        if year:
            prefix_parts.append(f"({year})")
        prefix = " ".join(prefix_parts).strip()
        if title:
            if prefix:
                return f"{prefix} — {title}", True
            return title, True
        return prefix, True
    fallback = (reference or {}).get("raw_reference") or ""
    return fallback, False


def format_candidate_label(source_label: str, candidate: dict) -> str:
    title = (candidate or {}).get("title") or "Untitled"
    doi = (candidate or {}).get("doi")
    if doi:
        return f"{source_label}: {title} (DOI: {doi})"
    return f"{source_label}: {title}"


def build_resolution_candidates(resolution: dict) -> dict:
    candidates = {}
    for source, label in (
        ("grobid", "GROBID"),
        ("crossref", "Crossref"),
        ("openalex", "OpenAlex"),
    ):
        candidate = (resolution or {}).get(source)
        if candidate:
            candidates[source] = format_candidate_label(label, candidate)
    return candidates


def build_citation_graphviz(graph_data: dict) -> graphviz.Digraph:
    graph = graphviz.Digraph()
    graph.attr(rankdir="LR", bgcolor="transparent")
    nodes = graph_data.get("nodes") or []
    edges = graph_data.get("edges") or []
    root_id = graph_data.get("root_id")

    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        if node.get("kind") == "stub":
            label = "data unavailable"
        else:
            label_parts = [node.get("label") or "Untitled work"]
            year = node.get("year")
            if year:
                label_parts.append(str(year))
            label = "\n".join(label_parts)
        if node.get("kind") == "stub":
            graph.node(
                node_id,
                label,
                style="dashed",
                color="#999999",
                fontcolor="#777777",
            )
        elif node_id == root_id:
            graph.node(
                node_id,
                label,
                style="filled",
                fillcolor="#e6eefb",
                color="#2f5d9f",
            )
        else:
            graph.node(node_id, label, style="rounded", color="#4b4f5a")

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        relation = edge.get("relation")
        if relation == "cited_by":
            graph.edge(source, target, color="#2f5d9f")
        elif relation == "references":
            graph.edge(source, target, color="#6b4e3d")
        else:
            graph.edge(source, target, color="#666666")

    return graph


# === UI Drawing ===
def color_tokens(
    text: str, token_scores: List[float], color_pos="green", color_neg="red"
):
    """Color words according to token salience."""
    tokens = text.split()
    colored = []
    for token, score in zip(tokens, token_scores):
        # Clamp/normalize score as needed, e.g., [-1, 1]
        if score > 0:
            color = color_pos
        elif score < 0:
            color = color_neg
        else:
            color = "black"
        # Make the intensity proportional to abs(score)
        intensity = min(1.0, abs(score))
        alpha = 0.4 + 0.6 * intensity  # Range [0.4, 1.0]
        span = f'<span style="color:{color};opacity:{alpha};">{token}</span>'
        colored.append(span)
    # Join back, preserving spaces
    return " ".join(colored)


def draw_sidebar():
    init_session_state()
    with st.sidebar:
        st.header("Pipeline Mode")
        mode = st.selectbox(
            "Choose pipeline",
            ["classic", "hybrid"],
            index=0 if st.session_state["pipeline_mode"] == "classic" else 1,
            help="Classic = fast, Hybrid = exhaustive",
            key="pipeline_mode_selectbox",
        )
        st.session_state["pipeline_mode"] = mode
        st.header("PDF Ingestion")
        st.checkbox(
            "Auto-run extraction",
            key="auto_extract_on_upload",
            help="Run extraction immediately after upload.",
        )
        st.checkbox(
            "Auto-run reference resolution",
            key="auto_resolve_on_upload",
            help="Resolve references after extraction completes.",
        )
        st.checkbox(
            "Seed sample claims when workspace is empty",
            key="show_demo_claims",
            help=(
                "Populate demo claims only when no segmentation results exist. "
                "Leave unchecked for a clean workspace once your Blablador token "
                "is configured."
            ),
        )
        st.checkbox(
            "Debug callouts",
            key="citation_debug",
            help="Show raw sentence + callout strings for troubleshooting.",
        )
        st.file_uploader(
            "PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="uploaded_pdfs",
            on_change=handle_pdf_upload,
        )
        if st.button("Refresh ingested PDFs"):
            refresh_ingested_docs()
        docs = st.session_state.get("ingested_docs") or []
        if docs:
            doc_ids = [doc.get("id") for doc in docs if doc.get("id")]
            current = st.session_state.get("selected_doc_id")
            if doc_ids and current not in doc_ids:
                st.session_state["selected_doc_id"] = doc_ids[0]
            if doc_ids:
                st.selectbox(
                    "Active document",
                    doc_ids,
                    format_func=lambda doc_id: next(
                        (
                            f"{doc.get('filename')} ({doc_id[:8]})"
                            for doc in docs
                            if doc.get("id") == doc_id
                        ),
                        doc_id,
                    ),
                    key="selected_doc_id",
                    on_change=load_selected_document,
                )
        else:
            st.caption("No PDFs ingested yet.")
        st.header("Upload your data")
        st.file_uploader(
            "CSV & TEI files",
            type=["csv", "xml"],
            accept_multiple_files=True,
            key="uploaded_files",
            on_change=handle_upload,
        )

        st.header("Source & API Configuration")
        st.text_input(
            "Blablador API Key",
            key="api_key",
            on_change=lambda: st.session_state.pop("available_models", None),
        )
        st.text_input(
            "Blablador Base URL",
            key="api_base",
            on_change=lambda: st.session_state.pop("available_models", None),
        )
        st.text_input("Citation API URL", key="api_url")

        st.header("LLM for Rationale")
        with st.spinner("Fetching models..."):
            if "available_models" not in st.session_state:
                st.session_state.available_models = get_responsive_models()
        model_selector(
            "LLM model",
            "selected_model",
            st.session_state.available_models,
            allow_custom=False,
        )

        st.header("Embedding Model")
        embed_choices = get_models("embed") or list_local_models()
        if settings.EMBED_MODEL not in embed_choices:
            embed_choices.insert(0, settings.EMBED_MODEL)
        model_selector("Embedding model", "embed_model", embed_choices)
        add_model("embed", st.session_state.embed_model)

        st.header("FAISS Retrieval")
        st.number_input(
            "Max initial sentences",
            min_value=1,
            key="max_sentences",
            on_change=reset_segmentation,
        )
        st.slider(
            "FAISS min similarity",
            0.0,
            1.0,
            key="faiss_min_score",
            on_change=reset_segmentation,
        )

        st.header("Reranker")
        rerank_choices = get_models("reranker") or [settings.RERANKER_MODEL]
        model_selector("Reranker model", "reranker_model", rerank_choices)
        add_model("reranker", st.session_state.reranker_model)
        st.number_input(
            "Reranker top-K",
            min_value=1,
            key="reranker_top_k",
        )

        st.header("NLI Model")
        nli_choices = get_models("nli") or [settings.NLI_MODEL]
        model_selector("NLI model", "nli_model", nli_choices)
        add_model("nli", st.session_state.nli_model)
        st.slider(
            "NLI confidence threshold",
            0.0,
            1.0,
            key="nli_threshold",
            on_change=reset_segmentation,
        )

        st.button(
            "Start segmentation",
            on_click=lambda: (
                st.session_state.__setitem__("started", True),
                st.session_state.__setitem__("seg_requested", True),
            ),
        )


def draw_ingestion_panel():
    # Restore selected document from query params (clicking citation chips
    # navigates with ?doc=...&cite=...).
    try:
        params = st.query_params  # type: ignore[attr-defined]
        param_doc = params.get("doc")
        if isinstance(param_doc, list):
            param_doc = param_doc[0] if param_doc else None
        if param_doc and param_doc != st.session_state.get("selected_doc_id"):
            st.session_state["selected_doc_id"] = str(param_doc)
            load_selected_document(show_error=False)
    except Exception:
        pass
    st.subheader("PDF Ingestion")
    docs = st.session_state.get("ingested_docs")
    if docs is None:
        docs = refresh_ingested_docs(show_error=True)
    if not docs:
        # One more attempt in case we landed here via a chip click.
        docs = refresh_ingested_docs(show_error=True)
        if not docs:
            st.info("Upload a PDF from the sidebar to begin.")
            return
    doc_id = st.session_state.get("selected_doc_id")
    if not doc_id:
        st.info("Select a PDF to view details.")
        return
    document = st.session_state.get("active_document")
    if not document or document.get("id") != doc_id:
        document = load_selected_document(show_error=False)
    if not document:
        st.info("Select a PDF to view details.")
        return

    col_action, col_status = st.columns([1, 3])
    with col_action:
        if st.button("Run Extraction"):
            api_url = st.session_state.get("api_url", "http://localhost:8000")
            with st.spinner("Running extraction..."):
                try:
                    trigger_extraction(api_url, doc_id)
                    document = load_selected_document(show_error=False)
                    st.success("Extraction complete.")
                except RuntimeError as exc:
                    st.error(f"Extraction failed: {exc}")
        if st.button("Resolve References"):
            api_url = st.session_state.get("api_url", "http://localhost:8000")
            with st.spinner("Resolving references..."):
                try:
                    trigger_resolution(api_url, doc_id)
                    document = load_selected_document(show_error=False)
                    st.success("Resolution complete.")
                except RuntimeError as exc:
                    st.warning(_resolution_error_message(exc))
    with col_status:
        st.write(
            {
                "Filename": document.get("filename"),
                "Uploaded": document.get("uploaded_at"),
                "Status": document.get("status"),
                "Size (bytes)": document.get("size_bytes"),
            }
        )

    extraction = document.get("extraction") or {}
    extraction_data = extraction.get("data") or {}
    resolution = document.get("resolution") or {}
    resolution_data = resolution.get("data") or []

    with st.expander("Metadata", expanded=True):
        metadata = extraction_data.get("metadata") or {}
        if metadata:
            rows = [
                {"Field": key, "Value": stringify_value(value)}
                for key, value in metadata.items()
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.info(
                "No metadata available yet. Run extraction to populate this section."
            )

    with st.expander("Citations"):
        citations = extraction_data.get("citations") or []
        if citations:
            st.dataframe(
                pd.DataFrame(normalize_records(citations)),
                width="stretch",
            )
        else:
            st.info("No citations extracted yet.")

    with st.expander("Bibliography"):
        references = extraction_data.get("references") or []
        if references:
            st.dataframe(
                pd.DataFrame(normalize_records(references)),
                width="stretch",
            )
        else:
            st.info("No bibliography entries extracted yet.")

    with st.expander("Resolution Results"):
        if resolution_data:
            st.dataframe(
                pd.DataFrame(normalize_records(resolution_data)),
                width="stretch",
            )
        else:
            st.info("No resolved references yet. Run resolution after extraction.")

    st.divider()
    st.subheader("Citation Context")

    api_url = st.session_state.get("api_url", "http://localhost:8000")
    try:
        body_payload = get_document_body(api_url, doc_id)
    except RuntimeError as exc:
        st.info("Run extraction to populate the document text.")
        st.caption(str(exc))
        return
    paragraphs = body_payload.get("paragraphs") or []
    if not paragraphs:
        st.info("Run extraction to populate the document text.")
        return

    def select_citation(index: int, target_id: str | None) -> None:
        normalized_target = normalize_target_id(target_id)
        st.session_state["citation_selected_index"] = index
        st.session_state["citation_selected_target"] = normalized_target
        st.session_state["citation_context_key"] = None
        st.session_state["citation_context"] = None
        st.session_state["citation_context_error"] = None
        st.session_state["citation_last_context_request"] = None
        st.session_state["citation_follow_open"] = False
        st.session_state["citation_graph_key"] = None
        st.session_state["citation_graph"] = None
        st.session_state["citation_graph_error"] = None
        st.session_state["citation_last_graph_request"] = None

    def load_citation_context(request: dict) -> None:
        st.session_state["citation_context_error"] = None
        st.session_state["citation_context"] = None
        st.session_state["citation_context_key"] = (
            request["doc_id"],
            request["citation_index"],
            request.get("target_id"),
        )
        st.session_state["citation_last_context_request"] = request
        placeholder = st.empty()
        with placeholder:
            render_skeleton(3)
        try:
            response = get_citation_context(
                request["api_url"],
                request["doc_id"],
                request["citation_index"],
                target_id=request.get("target_id"),
            )
        except RuntimeError as exc:
            st.session_state["citation_context_error"] = str(exc)
            show_api_error(f"Failed to load citation context: {exc}")
            placeholder.empty()
            return
        placeholder.empty()
        st.session_state["citation_context"] = response.get("context")

    def load_citation_graph(request: dict) -> None:
        st.session_state["citation_graph_error"] = None
        st.session_state["citation_graph"] = None
        st.session_state["citation_graph_key"] = (
            request["doc_id"],
            request["target_id"],
            request.get("doi"),
            request["depth"],
            request["max_nodes"],
        )
        st.session_state["citation_last_graph_request"] = request
        placeholder = st.empty()
        with placeholder:
            render_skeleton(4)
        try:
            response = get_citation_graph(
                request["api_url"],
                request["doc_id"],
                request["target_id"],
                request["depth"],
                request["max_nodes"],
                doi=request.get("doi"),
            )
        except RuntimeError as exc:
            st.session_state["citation_graph_error"] = str(exc)
            show_api_error(f"Failed to load citation graph: {exc}")
            placeholder.empty()
            return
        placeholder.empty()
        st.session_state["citation_graph"] = response

    inject_citation_styles()
    main_col, rail_col = st.columns([7, 5], gap="large")

    def _read_query_params() -> dict:
        try:
            raw = st.query_params  # type: ignore[attr-defined]
            return {key: raw.get(key) for key in raw.keys()}
        except Exception:
            return st.experimental_get_query_params()

    params = _read_query_params()
    param_doc = params.get("doc")
    param_cite = params.get("cite")
    param_target = params.get("target")
    if isinstance(param_doc, list):
        param_doc = param_doc[0] if param_doc else None
    if isinstance(param_cite, list):
        param_cite = param_cite[0] if param_cite else None
    if isinstance(param_target, list):
        param_target = param_target[0] if param_target else None

    if param_doc and param_doc != st.session_state.get("selected_doc_id"):
        st.session_state["selected_doc_id"] = str(param_doc)
        load_selected_document(show_error=False)
    if param_cite is not None:
        try:
            select_citation(
                int(str(param_cite)), str(param_target) if param_target else None
            )
        except ValueError:
            pass

    with main_col:
        st.markdown("#### Document text")
        st.caption("Click an in-text citation chip to inspect context.")

        selected_index = st.session_state.get("citation_selected_index")
        for para in paragraphs:
            segments = para.get("segments") or []
            para_citations = set(para.get("citation_indices") or [])
            highlight = selected_index is not None and selected_index in para_citations
            wrapper_class = (
                "citation-paragraph citation-selected"
                if highlight
                else "citation-paragraph"
            )

            parts: list[str] = []
            for seg in segments:
                seg_type = seg.get("type")
                if seg_type == "text":
                    parts.append(html.escape(seg.get("text") or ""))
                    continue
                if seg_type != "citation":
                    continue
                cite_index = seg.get("citation_index")
                if cite_index is None:
                    continue
                cite_index = int(cite_index)
                anchor = f"cite-idx-{cite_index}"
                target_id = seg.get("target_id")
                label = seg.get("label") or seg.get("callout") or "citation"
                href = _citation_href(doc_id, cite_index, target_id, anchor)
                parts.append(f'<a id="{anchor}"></a>')
                parts.append(
                    f'<a class="citation-chip citation-chip-link" '
                    f'href="{html.escape(href)}">{html.escape(str(label))}</a>'
                )

            rendered = " ".join(part for part in parts if part)
            st.markdown(
                f'<div class="{wrapper_class}">{rendered}</div>',
                unsafe_allow_html=True,
            )

    with rail_col:
        st.markdown('<div class="citation-workflow-rail">', unsafe_allow_html=True)
        st.markdown("#### Workflow")
        st.caption("Click a citation chip to populate this panel.")

        selected_index = st.session_state.get("citation_selected_index")
        if selected_index is None:
            st.info("Click an in-text citation chip to inspect and parse.")
        else:
            selected_target = normalize_target_id(
                st.session_state.get("citation_selected_target")
            )
            anchor = f"cite-idx-{int(selected_index)}"
            st.markdown(f"[Jump to text](#{anchor})")

            context_request = {
                "api_url": api_url,
                "doc_id": doc_id,
                "citation_index": selected_index,
                "target_id": selected_target,
            }
            current_key = (
                context_request["doc_id"],
                context_request["citation_index"],
                context_request.get("target_id"),
            )
            if st.session_state.get("citation_context_key") != current_key:
                load_citation_context(context_request)
            context = st.session_state.get("citation_context") or {}

            if context:
                st.markdown("**Context**")
                st.write(context.get("previous_sentence") or "")
                st.write(
                    context.get("citing_sentence") or context.get("sentence") or ""
                )
                st.write(context.get("next_sentence") or "")
            if st.session_state.get("citation_debug"):
                st.caption("Raw context payload")
                st.json(context)

            active_panel = st.session_state.get("workflow_active_panel")
            with st.expander("Parsing", expanded=active_panel == "parsing"):
                model = st.session_state.get("selected_model")
                ta_key = f"citation-segments-{selected_index}"
                stored = st.session_state.get("citation_sentence_segments", {}).get(
                    str(selected_index),
                    [],
                )
                st.session_state.setdefault(ta_key, "\n".join(stored))
                if st.button(
                    "Segment sentence",
                    key=f"segment-sentence-{selected_index}",
                    disabled=not model,
                ):
                    st.session_state["workflow_active_panel"] = "parsing"
                    seg_source = (
                        context.get("citing_sentence") or context.get("sentence") or ""
                    )
                    st.caption("Parsing input")
                    st.write(seg_source)
                    segments = seg_via_llm(seg_source, int(selected_index) + 1, model)
                    st.session_state.setdefault("citation_sentence_segments", {})[
                        str(selected_index)
                    ] = segments
                    st.session_state[ta_key] = "\n".join(segments)

                seg_text = st.text_area(
                    "Parsed claims (one per line)",
                    key=ta_key,
                    height=160,
                )
                if st.button("Save claims", key=f"save-claims-{selected_index}"):
                    st.session_state["workflow_active_panel"] = "parsing"
                    lines = [ln.strip() for ln in seg_text.splitlines() if ln.strip()]
                    st.session_state.setdefault("citation_sentence_segments", {})[
                        str(selected_index)
                    ] = lines
                    primary_callout = context.get("callout") or "citation"
                    reference_hint = {
                        "callout": primary_callout,
                        "reference_id": selected_target,
                    }
                    saved = 0
                    for idx, line in enumerate(lines):
                        parsed = to_segment_dict(line)
                        segment_id = parsed.get("segment_id") or f"seg-{idx+1}"
                        claim_text = parsed.get("claim") or line
                        claim_id = f"cite:{doc_id}:{selected_index}:{segment_id}"
                        claim_queue.register_claim(
                            claim_id,
                            claim=claim_text,
                            callout=primary_callout,
                            doc_id=doc_id,
                            reference_id=selected_target,
                            reference_hint=reference_hint,
                        )
                        saved += 1
                    if saved:
                        st.success(f"Saved {saved} claim(s) to the workspace.")

            with st.expander("Retrieving", expanded=active_panel == "retrieving"):
                if not selected_target:
                    st.info("No target ID available for this citation.")
                else:
                    reference = context.get("reference") or {}
                    resolution = context.get("resolution") or {}
                    summary = format_reference_summary(reference, resolution)
                    if summary:
                        st.markdown("**Reference summary**")
                        st.markdown(summary)
                    with st.expander("Retrieval instructions", expanded=False):
                        claim_queue.render_retrieval_instructions(
                            api_url=api_url,
                            doc_id=doc_id,
                            reference_id=selected_target,
                            key_prefix=str(selected_index),
                        )

        st.divider()
        st.markdown("#### Citation Graph")
        selected_target = normalize_target_id(
            st.session_state.get("citation_selected_target")
        )
        if selected_target is None:
            st.info("Select a citation callout to view the graph.")
        else:
            context_snapshot = st.session_state.get("citation_context") or {}
            resolution_entry = context_snapshot.get("resolution") or {}
            reference_entry = context_snapshot.get("reference") or {}
            resolved_identifier = (
                resolution_entry.get("openalex_id")
                or resolution_entry.get("openalex_work_id")
                or resolution_entry.get("doi")
                or reference_entry.get("doi")
            )
            if resolved_identifier:
                st.caption(f"Resolved identifier: {resolved_identifier}")
            else:
                st.caption("No DOI resolved for this citation yet.")
            st.caption(f"Target ID: {selected_target}")
            st.slider(
                "Depth",
                min_value=1,
                max_value=3,
                key="citation_graph_depth",
            )
            st.number_input(
                "Node cap",
                min_value=5,
                max_value=50,
                key="citation_graph_max_nodes",
            )
            graph_request = {
                "api_url": api_url,
                "doc_id": doc_id,
                "target_id": selected_target,
                "doi": resolved_identifier,
                "depth": int(st.session_state.get("citation_graph_depth", 1)),
                "max_nodes": int(st.session_state.get("citation_graph_max_nodes", 10)),
            }
            if st.button("Load citation graph", key="citation-graph-load"):
                load_citation_graph(graph_request)
            graph_key = (
                graph_request["doc_id"],
                graph_request["target_id"],
                graph_request.get("doi"),
                graph_request["depth"],
                graph_request["max_nodes"],
            )
            graph_data = None
            if st.session_state.get("citation_graph_key") == graph_key:
                graph_data = st.session_state.get("citation_graph")
            if graph_data:
                nodes = graph_data.get("nodes") or []
                if not nodes:
                    st.info("No data available for this citation graph.")
                else:
                    graph = build_citation_graphviz(graph_data)
                    st.graphviz_chart(graph)
            else:
                st.caption("Load the citation graph to explore references.")
            if st.session_state.get("citation_graph_error"):
                error_message = st.session_state.get("citation_graph_error", "")
                if "OpenAlex request failed (404)" in error_message:
                    st.caption(
                        "OpenAlex could not find this work. "
                        "Check that reference resolution populated a DOI or OpenAlex "
                        "ID."
                    )
                if st.button("Retry graph", key="citation-graph-retry"):
                    last_request = st.session_state.get("citation_last_graph_request")
                    if last_request:
                        load_citation_graph(last_request)

        st.markdown("</div>", unsafe_allow_html=True)


def draw_main():
    st.title("Citation-Support Checker")
    draw_ingestion_panel()
    render_attachment_workspace()
    render_evidence_panel()
    st.divider()
    if not st.session_state.started:
        st.info("Configure settings then click Start segmentation.")
        return
    import glob

    folder = st.session_state.get("data_dir", "")
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        st.error("No CSV file found in the folder.")
        return
    df = utils.read_csv(csv_files[0])
    # Clean up whitespace for robust row IDs
    df["tei_file"] = df["tei_file"].astype(str).str.strip()
    df["tei_xml_id"] = df["tei_xml_id"].astype(str).str.strip()
    df["row_id"] = df["tei_file"] + "::" + df["tei_xml_id"]
    st.write("Loaded rows:", len(df))
    st.write("Folder path:", folder)
    # (Optional, but good for debugging:) After building the DataFrame, add:
    st.write("All DataFrame row_id values:", list(df["row_id"]))

    # ==== Begin Streamlit form for segment evaluation UI ====
    if st.session_state.seg_requested:
        # 1) Populate seg_cache if empty
        if not st.session_state.get("seg_cache"):
            st.session_state["seg_cache"] = {}
            with st.spinner("Segmenting sentences…"):
                for i, row in enumerate(df.itertuples(index=False), 1):
                    row_id = row.row_id.strip()
                    segments = seg_via_llm(
                        row.tei_sentence, i, st.session_state["selected_model"]
                    )
                    st.session_state["seg_cache"][row_id] = segments
        # Now, use a Streamlit form with one expander per sentence/segment.
        with st.form("sentence_segment_evaluation_form"):
            for i, row in enumerate(df.itertuples(index=False), 1):
                row_id = row.row_id.strip()
                expanded = True if i == 1 else False
                with st.expander(
                    f"Row {row_id}: {row.tei_sentence}",
                    expanded=expanded,
                ):
                    seg_text = st.text_area(
                        f"Candidate segments (edit if needed) - Row {row_id}",
                        "\n".join(
                            st.session_state.get("seg_cache", {}).get(row_id, [])
                        ),
                        key=f"ta-{row_id}",
                        height=120,
                    )
                    st.session_state[f"edited-{row_id}"] = [
                        ln.strip() for ln in seg_text.splitlines() if ln.strip()
                    ]
            submitted = st.form_submit_button("Submit all choices")

        # If form submitted, process all user inputs together
        if submitted:
            api_url = st.session_state.get("api_url", "http://localhost:8000")
            with st.spinner("Running citation-support for all rows…"):
                for i, row in enumerate(df.itertuples(index=False), 1):
                    row_id = row.row_id.strip()
                    segments = seg_via_llm(
                        row.tei_sentence, i, st.session_state["selected_model"]
                    )
                    st.session_state["seg_cache"][row_id] = segments

                    # Coerce possible NaN → "" for the 'tei_target' field
                    tid = getattr(row, "tei_target", "")
                    if pd.isna(tid):
                        tid = ""

                    payload = {
                        "folder": st.session_state["data_dir"],
                        "row_id": row_id,
                        "citing_title": getattr(row, "cited_author", ""),
                        "citing_id": tid,
                        "original_sentence": row.tei_sentence,
                        "segments": [
                            to_segment_dict(seg)
                            for seg in st.session_state.get(
                                f"edited-{row_id}", segments
                            )
                        ],
                        "settings": {
                            "embed_model": st.session_state["embed_model"],
                            "max_sentences": st.session_state["max_sentences"],
                            "faiss_min_score": st.session_state["faiss_min_score"],
                            "nli_model": st.session_state["nli_model"],
                            "llm_model": st.session_state["selected_model"],
                            "nli_threshold": st.session_state["nli_threshold"],
                            "api_key": st.session_state["api_key"],
                            "base_url": st.session_state["api_base"],
                            "reranker_model": st.session_state.get("reranker_model"),
                            "reranker_top_k": st.session_state.get("reranker_top_k"),
                            "pipeline_mode": st.session_state["pipeline_mode"],
                        },
                    }
                    try:
                        response = requests.post(f"{api_url}/segment", json=payload)
                        st.session_state["results"][row_id] = response.json()
                    except BaseException:
                        st.session_state["results"][row_id] = response.text

            # --- Results/Assessment UI ---
            # After the segmentation form
            if st.session_state.get("results"):
                # Show results/assessment forms for rows with results
                for row_id in sorted(st.session_state["results"].keys()):
                    result = st.session_state["results"].get(row_id)
                    if not (result and isinstance(result, dict)):
                        continue

                    st.markdown(f"### Results for Row {row_id}")
                    original_sentence = result.get("original_sentence", "")
                    st.markdown(f"**Original sentence:** {original_sentence}")

                    # Per-row form for all segments in this row
                    with st.form(f"assessment_form_row_{row_id}"):
                        segs = result.get("segments", [])
                        all_troll_pay_items = []
                        for seg_idx, seg in enumerate(segs):
                            seg_id = seg.get("segment_id", "")
                            claim = seg.get("claim", "")
                            f"exp_{row_id}_{seg_id}"

                            with st.expander(
                                f"Row {row_id} Segment {seg_id}: {claim}",
                                expanded=(
                                    seg_idx == 0
                                    and not st.session_state.get(
                                        f"done_row_{row_id}", False
                                    )
                                ),
                            ):
                                evidence = seg.get("evidence", [])
                                N = settings.NLI_CANDIDATES_SHOWN
                                support = sorted(
                                    [
                                        ev
                                        for ev in evidence
                                        if ev.get("label") == "entailment"
                                    ],
                                    key=lambda ev: ev.get("score", 0),
                                    reverse=True,
                                )[:N]
                                contradiction = sorted(
                                    [
                                        ev
                                        for ev in evidence
                                        if ev.get("label") == "contradiction"
                                    ],
                                    key=lambda ev: ev.get("score", 0),
                                    reverse=True,
                                )[:N]

                                st.write("**Supporting Evidence:**")
                                support_checked = []
                                for i, ev in enumerate(support):
                                    eid = ev["id"]
                                    salience = ev.get("token_scores")
                                    text = ev.get("text", "")
                                    # Compose a two-line HTML block.
                                    section_path = (
                                        ev.get("section_path")
                                        or ev.get("section_head")
                                        or ""
                                    )
                                    if salience and settings.SHOW_SALIENCE:
                                        coloured_text = color_tokens(text, salience)
                                    else:
                                        coloured_text = text
                                    if section_path:
                                        html_block = (
                                            "<div><strong>Location:</strong> "
                                            f"{section_path}</div>"
                                            f"<div>{coloured_text}</div>"
                                        )
                                    else:
                                        html_block = coloured_text
                                    st.markdown(html_block, unsafe_allow_html=True)
                                    key = f"support_cb_{row_id}_{seg_id}_{eid}"
                                    checked = st.session_state.get(key, False)
                                    cb = st.checkbox("Select", value=checked, key=key)
                                    if cb:
                                        support_checked.append(eid)
                                seg["user_selected_support"] = support_checked

                                st.write("**Contradicting Evidence:**")
                                contra_checked = []
                                for i, ev in enumerate(contradiction):
                                    eid = ev["id"]
                                    salience = ev.get("token_scores")
                                    text = ev.get("text", "")
                                    # Compose a two-line HTML block.
                                    section_path = (
                                        ev.get("section_path")
                                        or ev.get("section_head")
                                        or ""
                                    )
                                    if salience and settings.SHOW_SALIENCE:
                                        coloured_text = color_tokens(text, salience)
                                    else:
                                        coloured_text = text
                                    if section_path:
                                        html_block = (
                                            "<div><strong>Location:</strong> "
                                            f"{section_path}</div>"
                                            f"<div>{coloured_text}</div>"
                                        )
                                    else:
                                        html_block = coloured_text
                                    st.markdown(html_block, unsafe_allow_html=True)
                                    key = f"contradict_cb_{row_id}_{seg_id}_{eid}"
                                    checked = st.session_state.get(key, False)
                                    cb = st.checkbox("Select", value=checked, key=key)
                                    if cb:
                                        contra_checked.append(eid)
                                seg["user_selected_contradiction"] = contra_checked

                                # --- Collect troll pay evidence per segment ---
                                troll_pay_items = []
                                for ev in evidence:
                                    all_scores = ev.get("all_scores", {})
                                    label_scores = sorted(
                                        all_scores.items(), key=lambda x: -x[1]
                                    )
                                    for i, (label1, score1) in enumerate(label_scores):
                                        if label1 not in {
                                            "entailment",
                                            "contradiction",
                                        }:
                                            continue
                                        for j, (label2, score2) in enumerate(
                                            label_scores
                                        ):
                                            if j == i:
                                                continue
                                            if label2 not in {
                                                "entailment",
                                                "contradiction",
                                            }:
                                                continue
                                            if (
                                                abs(score1 - score2)
                                                <= settings.TROLL_PAY_MARGIN
                                            ):
                                                troll_pay_items.append((seg_id, ev))
                                                break
                                        else:
                                            continue
                                        break
                                all_troll_pay_items.extend(troll_pay_items)

                        # After all segments, show one troll-pay expander per row.
                        if all_troll_pay_items:
                            # Deduplicate on (seg_id, evidence id)
                            seen_keys = set()
                            deduped_troll_pay_items = []
                            for seg_id, ev in all_troll_pay_items:
                                eid = ev["id"]
                                k = (seg_id, eid)
                                if k not in seen_keys:
                                    seen_keys.add(k)
                                    deduped_troll_pay_items.append((seg_id, ev))

                            # Find the most ambiguous item by smallest score margin.
                            def ambiguity(ev):
                                scores = sorted(
                                    ev.get("all_scores", {}).values(), reverse=True
                                )
                                if len(scores) >= 2:
                                    return abs(scores[0] - scores[1])
                                return 1.0  # Not ambiguous if only one score

                            most_ambiguous = None
                            min_margin = 1.0
                            for seg_id, ev in deduped_troll_pay_items:
                                margin = ambiguity(ev)
                                if margin < min_margin:
                                    min_margin = margin
                                    most_ambiguous = (seg_id, ev)
                            if most_ambiguous:
                                seg_id, ev = most_ambiguous
                                with st.expander(
                                    "🧌 tax! Help us on this ambiguous case",
                                    expanded=True,
                                ):
                                    st.write(
                                        "This candidate was difficult for the model "
                                        "to classify. Please help by labeling it:"
                                    )
                                    eid = ev["id"]
                                    text = ev.get("text", "")
                                    section_path = (
                                        ev.get("section_path")
                                        or ev.get("section_head")
                                        or ""
                                    )
                                    label = (
                                        f"**Row {row_id} Segment {seg_id}**<br>"
                                        f"**Section:** {section_path}<br>{text}"
                                        if section_path
                                        else text
                                    )
                                    radio_key = f"troll_radio_{row_id}_{seg_id}_{eid}"
                                    prev = st.session_state.get(radio_key, "I dunno")
                                    choice = st.radio(
                                        label,
                                        options=[
                                            "Entails",
                                            "Neutral",
                                            "Contradicts",
                                            "I dunno",
                                        ],
                                        index=(
                                            [
                                                "Entails",
                                                "Neutral",
                                                "Contradicts",
                                                "I dunno",
                                            ].index(prev)
                                            if prev
                                            in [
                                                "Entails",
                                                "Neutral",
                                                "Contradicts",
                                                "I dunno",
                                            ]
                                            else 3
                                        ),
                                        key=radio_key,
                                        horizontal=True,
                                    )
                                    # Assign to proper segment (optional)
                                    for seg in segs:
                                        if seg.get("segment_id", "") == seg_id:
                                            troll_selected = seg.get(
                                                "user_selected_trollpay", {}
                                            )
                                            troll_selected[eid] = choice
                                            seg[
                                                "user_selected_trollpay"
                                            ] = troll_selected

                        submitted = st.form_submit_button("Submit Assessment")
                        if submitted:
                            st.session_state[f"done_row_{row_id}"] = True
                            st.success("Assessment saved!")

            # After the form, show raw JSON per row (unchanged)
            for row_id in sorted(st.session_state["results"].keys()):
                result = st.session_state["results"].get(row_id)
                if result:
                    st.markdown("**Raw Result JSON (includes your selections):**")
                    st.json(result)
        else:
            # Only show *one* instructional message if no results yet
            st.info(
                "Click 'Submit all choices' above to search for candidate cited "
                "sentences."
            )
    # ==== End Streamlit form for segment evaluation UI ====

    # Only prebuild FAISS indices for classic pipeline mode
    if (
        st.session_state.seg_requested
        and "faiss_started" not in st.session_state
        and st.session_state.get("pipeline_mode", "classic") == "classic"
    ):
        # now kick off backend index build in background
        with st.spinner("Building FAISS index in background…"):
            api_url = st.session_state.get("api_url", "http://localhost:8000")
            try:
                import math

                max_sent = st.session_state["max_sentences"]
                if pd.isna(max_sent) or (
                    isinstance(max_sent, float) and math.isnan(max_sent)
                ):
                    max_sent = 256
                min_score = st.session_state["faiss_min_score"]
                if pd.isna(min_score) or (
                    isinstance(min_score, float) and math.isnan(min_score)
                ):
                    min_score = 0.2
                body = {
                    "folder": str(st.session_state["data_dir"]),
                    "embed_model": st.session_state["embed_model"],
                    "max_chunks": int(max_sent),
                    "faiss_min_score": float(min_score),
                }
                prebuild_resp = requests.post(f"{api_url}/prebuild", json=body)
                prebuild_resp.raise_for_status()
                st.success("Backend indexing completed.")
            except requests.HTTPError as e:
                # Show detailed server error response
                try:
                    err_body = e.response.json()
                except Exception:
                    err_body = e.response.text
                st.error(
                    "Failed to start backend indexing "
                    f"(HTTP {e.response.status_code}): {e}\n"
                    f"Response body:\n{err_body}"
                )
            except Exception as e:
                st.error(f"Failed to start backend indexing: {e}")
        st.session_state["faiss_started"] = True

    # ------------------------
    # offer download once all rows are done
    if all(
        st.session_state.get(f"done_row_{row_id.strip()}", False)
        for row_id in df["row_id"]
    ):
        import json

        results_json = json.dumps(st.session_state["results"], indent=2)
        st.download_button(
            "Download all results as JSON",
            data=results_json,
            file_name="citation_support_results.json",
            mime="application/json",
        )


def run_prebuild():
    """Trigger the backend FAISS prebuild process."""
    api_url = st.session_state.get("api_url", "http://localhost:8000")
    # (somewhere you collect these from sliders/text inputs)
    max_chunks = st.session_state["max_chunks"]
    faiss_min_score = st.session_state["faiss_min_score"]

    # Coerce NaN → 0 (or whatever default you prefer) before building JSON:
    if pd.isna(max_chunks):
        max_chunks = None
    if pd.isna(faiss_min_score):
        faiss_min_score = 0.0

    payload = {
        "folder": st.session_state["data_dir"],
        "max_chunks": max_chunks,
        "faiss_min_score": float(faiss_min_score),
        "embed_model": st.session_state["embed_model"],
        "api_key": st.session_state["api_key"],
        "base_url": st.session_state["base_url"],
    }
    requests.post(f"{api_url}/prebuild", json=payload)


def main():
    draw_sidebar()
    draw_main()


if __name__ == "__main__":
    main()
