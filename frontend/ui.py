# frontend/ui.py

# === Imports ===
import html
import inspect
import json
import hashlib
import os
import pathlib
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from uuid import uuid4

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

from frontend import (
    attachment_queue,
    claim_queue,
    clipboard,
    evidence_store,
    judgment_api,
    judgment_store,
    workflow_api,
)
from frontend.components import chase_queue as chase_queue_component
from frontend.components import chasing_panel
from frontend.components import live_surfing_panel
from frontend.state_keys import (
    SCOPE_DRAFT_PROJECT_ID,
    SCOPE_DRAFT_UID,
    WORKSPACE_ACTIVE_TAB,
    WORKSPACE_DENSE_MODE,
    WORKSPACE_SETTINGS_OPEN,
    WORKSPACE_TAB_DOCUMENT,
    WORKSPACE_TAB_REVIEW,
    WORKSPACE_TAB_GRAPH,
    WORKSPACE_TAB_LABELS,
    canonical_segments_key,
)
from frontend.components.evidence_card import CardActionCallbacks, EvidenceCardRenderer
from frontend.components.rationale_sidebar import (
    build_filter_chip_config,
    build_progress_summary,
)
from frontend.evidence_api import MAX_LIST_REQUESTS, show_api_error

from backend import utils
from backend.bl_client import BlabladorClient
from backend.model_cache import add_model, get_models
from backend.settings import AppSettings
from backend.utils import list_local_models
from frontend.ingestion_api import (
    auto_place_claim_source,
    confirm_claims,
    get_claim_span_context,
    get_claim_status,
    get_span_bundle,
    compact_span_graph,
    set_span_cite_role,
    get_citation_context,
    get_citation_graph,
    get_document_body,
    get_document,
    list_documents,
    trigger_extraction,
    trigger_resolution,
    upload_pdf,
)
from frontend.opinion_api import (
    append_follow,
    list_follows_by_doc,
    resolve_span_id,
)
from frontend import ledger_api
from frontend import project_api
from frontend import scope_lock
from typing import Any, Dict, List, Optional


# Third-party


# Local application

# === Settings & State Initialization ===
settings = AppSettings()


DEFAULT_BACKEND_URL = (settings.BACKEND_URL or "http://localhost:8000").strip()
RUNTIME_STAMP_CACHE_TTL_SECONDS = 15


def get_api_url() -> str:
    """Return a usable backend URL even if the widget is blank."""
    url = (st.session_state.get("api_url") or "").strip()
    if not url:
        url = DEFAULT_BACKEND_URL
        st.session_state["api_url"] = url
    return url


def init_session_state():
    """Initialize Streamlit session state keys from settings."""
    defaults = {
        "api_url": (settings.BACKEND_URL or "http://localhost:8000"),
        "api_key": "" if settings.API_KEY == "..." else (settings.API_KEY or ""),
        "api_base": (settings.API_BASE or "http://localhost:8070"),
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
        "ingested_docs": None,
        "_ingested_docs_loaded": False,
        "selected_doc_id": "",
        "active_document": None,
        "citation_selected_index": None,
        "citation_selected_target": None,
        "citation_selected_sentence_id": None,
        "selected_callout_tuple": None,
        "citation_context_key": None,
        "citation_context": None,
        "citation_context_error": None,
        "citation_last_context_request": None,
        "citation_follow_open": False,
        "citation_graph_key": None,
        "citation_graph": None,
        "citation_graph_error": None,
        "citation_last_graph_request": None,
        "followed_citations": [],
        "citation_context_cache": {},
        "citation_parsing_inputs": {},
        "citation_graph_depth": 1,
        "citation_graph_max_nodes": 10,
        # Reviewer-scoped accepted segmentations for citing spans. Structure:
        # { reviewer_state: { "{doc}::{cite_idx}::{target_id}": ["1a. ...", ...] } }
        "citation_segments_by_reviewer": {},
        "citation_sentence_segments": {},
        "auto_extract_on_upload": True,
        "auto_resolve_on_upload": True,
        "citation_debug": False,
        "show_demo_claims": False,
        # Workspace shell (Phase 08)
        WORKSPACE_DENSE_MODE: False,
        WORKSPACE_SETTINGS_OPEN: False,
        WORKSPACE_ACTIVE_TAB: WORKSPACE_TAB_DOCUMENT,
        "workspace_upload_mode": "source",
        # Phase 08-07: named execution profiles for evidence reruns.
        "execution_profile": "Fast/Local",
        # Phase 08-08: optional HF Inference API toggle for NLI.
        "hf_remote": bool(getattr(settings, "HF_REMOTE_INFERENCE", False)),
        # Phase 09: document ledger (backend graph).
        "ledger_payload": None,
        "ledger_editor_doc": None,
        "ledger_editor_side": None,
        # Phase 09: project shell + export/import.
        "project_meta": None,
        "project_export_blob": None,
        "project_import_confirm": False,
        # Scope plumbing (Phase 10-04.5): /ingest endpoints are project-scoped.
        "project_id": "",
        SCOPE_DRAFT_UID: "",
        SCOPE_DRAFT_PROJECT_ID: "",
        "scope_new_uid": "",
        "scope_new_project_id": "",
        "scope_users_error": None,
        "scope_projects_last_uid": "",
        "scope_projects_error": None,
        # Phase 10-04.5: unified Intake drop + inbox.
        "intake_inbox": [],
        "intake_blobs": {},
        "intake_dropzone_nonce": 0,
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


def get_project_id() -> str:
    """Return current project scope for X-Project-Id."""
    return scope_lock.get_applied_project_id()


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


def _rerun() -> None:
    """Streamlit rerun helper across versions."""
    # Streamlit auto-reruns after callbacks; explicit rerun in that context
    # logs a warning and is ignored.
    try:
        for frame in inspect.stack():
            fn = str(frame.function or "")
            filename = str(frame.filename or "").replace("\\", "/")
            if fn in {"call_callback", "_call_callbacks"} and (
                "streamlit/runtime/state" in filename
                or "streamlit/runtime/scriptrunner" in filename
            ):
                return
    except Exception:
        pass

    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    experimental = getattr(st, "experimental_rerun", None)
    if callable(experimental):
        experimental()
        return
    raise RuntimeError("Streamlit rerun is unavailable in this version")


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
    "Hard constraint: every content word in your segments must appear in the "
    "Sentence (case-insensitive). Do not invent facts, entities, or new words. "
    "If you cannot comply, output exactly one segment that repeats the Sentence "
    "verbatim.\n\n"
    "List each segment on its own line, numbered {row_idx}a, {row_idx}b, etc., "
    "continuing alphabetically.\n\n"
    "Sentence:\n"
    "{sentence}\n"
    "Segments:\n"
)

SEG_RE = re.compile(r"^\s*\d+[a-z]\.", re.I)


def _segment_locally(sentence: str, row_idx: int) -> list[str]:
    raw = (sentence or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip()]
    if not parts:
        parts = [raw]
    segments: list[str] = []
    for idx, part in enumerate(parts):
        letter = chr(ord("a") + idx)
        segments.append(f"{row_idx}{letter}. {part}")
    return segments


def seg_via_llm(sentence: str, row_idx: int, model: str) -> list[str]:
    prompt = SEGMENT_PROMPT_TEMPLATE.format(row_idx=row_idx, sentence=sentence)
    actual_model = model  # use the model string chosen in the UI

    api_key = (st.session_state.get("api_key") or "").strip()
    base_url = (st.session_state.get("api_base") or "").strip()
    if not api_key or not base_url or not actual_model:
        return _segment_locally(sentence, row_idx)

    client = BlabladorClient(api_key=api_key, base_url=base_url)
    try:
        text = client.completion(
            prompt,
            model=actual_model,
            temperature=0,
            max_tokens=256,
        )
    except Exception as e:
        st.warning(f"Segmentation API error: {e}. Falling back locally.")
        return _segment_locally(sentence, row_idx)
    # Split and extract numbered segments
    lines = [ln.strip() for ln in text.splitlines()]
    segments = [ln for ln in lines if SEG_RE.match(ln)]
    return segments or _segment_locally(sentence, row_idx)


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
    api_url = get_api_url()
    previous = st.session_state.get("ingested_docs") or []
    try:
        documents = list_documents(api_url, project_id=get_project_id())
    except RuntimeError as exc:
        if show_error:
            st.error(f"Failed to load ingested PDFs: {exc}")
        return list(previous)

    # If the backend temporarily returns an empty list (startup, transient error),
    # keep the previous list so we don't blank the workspace.
    if not documents and previous:
        return list(previous)

    st.session_state["ingested_docs"] = documents
    return documents


def load_selected_document(show_error: bool = True) -> Optional[dict]:
    doc_id = st.session_state.get("selected_doc_id")
    if not doc_id:
        st.session_state["active_document"] = None
        return None
    api_url = get_api_url()
    try:
        document = get_document(api_url, doc_id, project_id=get_project_id())
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
    api_url = get_api_url()
    project_id = get_project_id()
    uploaded = []
    with st.spinner("Uploading PDFs..."):
        for file in files:
            try:
                uploaded_doc = upload_pdf(
                    api_url,
                    file,
                    project_id=project_id,
                    user_id=_active_reviewer_uid(),
                    auto_process=bool(st.session_state.get("auto_extract_on_upload")),
                )
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
            st.session_state["selected_doc_choice"] = last_doc["id"]
            st.session_state["active_document"] = last_doc
        st.success(f"Uploaded {len(uploaded)} PDF(s).")
        doc_id = last_doc.get("id")
        if doc_id and st.session_state.get("auto_extract_on_upload"):
            with st.spinner("Running extraction..."):
                try:
                    trigger_extraction(
                        api_url,
                        doc_id,
                        project_id=project_id,
                        user_id=_active_reviewer_uid(),
                    )
                    document = load_selected_document(show_error=False)
                    st.session_state["active_document"] = document
                    st.success("Extraction complete.")
                except RuntimeError as exc:
                    st.error(f"Extraction failed: {exc}")
                    return
        if doc_id and st.session_state.get("auto_resolve_on_upload"):
            with st.spinner("Resolving references..."):
                try:
                    trigger_resolution(
                        api_url,
                        doc_id,
                        project_id=project_id,
                        user_id=_active_reviewer_uid(),
                    )
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
    st.markdown(
        """
        <style>
        .citation-sentence {
            font-size: 0.96rem;
            line-height: 1.6;
        }
        .citation-chip {
            display: inline-block !important;
            padding: 2px 6px !important;
            margin: 0 2px !important;
            border-radius: 10px !important;
            background: rgba(39, 128, 227, 0.08) !important;
            color: #1f6fc7 !important;
            border: 1px solid rgba(39, 128, 227, 0.22) !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            text-decoration: none !important;
        }
        .citation-chip:hover {
            background: rgba(39, 128, 227, 0.12);
        }
        .citation-chip-link {
            text-decoration: underline;
            text-decoration-thickness: 1px;
            text-underline-offset: 2px;
            text-decoration-color: rgba(39, 128, 227, 0.35);
            transition: background 120ms ease, text-decoration-color 120ms ease;
        }
        .citation-chip-link:hover {
            text-decoration-color: rgba(39, 128, 227, 0.75);
        }
        .citation-chip-link:visited {
            color: #1f6fc7;
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
        <script>
        function getCenterPane() {
            var panes = document.querySelectorAll('div[data-testid="stVerticalBlockBorderWrapper"]');
            if (panes && panes.length >= 2) return panes[1];
            return null;
        }
        // Save center-pane scroll position before page unloads
        window.addEventListener('click', function(e) {
            if (e.target.closest('.citation-chip-link')) {
                var pane = getCenterPane();
                if (pane) {
                    sessionStorage.setItem('citationCenterScrollPos', pane.scrollTop || 0);
                }
            }
        });
        // Restore center-pane scroll position after page loads
        window.addEventListener('load', function() {
            var pos = sessionStorage.getItem('citationCenterScrollPos');
            if (pos !== null) {
                setTimeout(function() {
                    var pane = getCenterPane();
                    if (pane) {
                        pane.scrollTop = parseInt(pos);
                    }
                    sessionStorage.removeItem('citationCenterScrollPos');
                }, 50);
            }
            // Check for hash and scroll to anchor
            if (window.location.hash) {
                var id = window.location.hash.substring(1);
                var el = document.getElementById(id);
                if (el) {
                    setTimeout(function() {
                        el.scrollIntoView();
                    }, 100);
                }
            }
        });
        """,
        unsafe_allow_html=True,
    )


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
    span_key: Optional[str] = None,
) -> str:
    target = normalize_target_id(target_id)
    scope_uid = scope_lock.get_applied_uid()
    scope_project = scope_lock.get_applied_project_id()
    base = (
        f"?doc={doc_id}&cite={citation_index}" if doc_id else f"?cite={citation_index}"
    )
    if scope_uid:
        base += f"&uid={scope_uid}"
    if scope_project:
        base += f"&project={scope_project}"
    if span_key:
        base += f"&sid={span_key}"
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
            f'<a class="citation-chip citation-chip-link" '
            f'href="{html.escape(href)}" target="_self">'
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
            f'<a class="citation-chip citation-chip-link" '
            f'href="{html.escape(href)}" target="_self">'
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
    "parsing": "Extracting",
    "matched": "Matched",
    "error": "Error",
}


# === Evidence Review (Phase 06) Styles ===
EVIDENCE_REVIEW_CSS_PATH = (
    pathlib.Path(__file__).resolve().parent / "assets" / "evidence_review.css"
)


# === Workspace Shell (Phase 08) Styles ===
WORKSPACE_CSS_PATH = (
    pathlib.Path(__file__).resolve().parent / "assets" / "workspace.css"
)


# === Judgment (Phase 07) Styles ===
JUDGMENT_CSS_PATH = pathlib.Path(__file__).resolve().parent / "assets" / "judgment.css"


RAIL_DEBUG_LOG_PATH = pathlib.Path("/tmp/rail_debug.log")


def _rail_debug_log(event: str, **payload: Any) -> None:
    try:
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        rec: Dict[str, Any] = {"ts": ts, "event": str(event)}
        if payload:
            rec["payload"] = payload
        with RAIL_DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=True) + "\n")
    except Exception:
        pass


def inject_evidence_review_styles() -> None:
    if EVIDENCE_REVIEW_CSS_PATH.exists():
        st.markdown(
            f"<style>{EVIDENCE_REVIEW_CSS_PATH.read_text()}</style>",
            unsafe_allow_html=True,
        )


def inject_judgment_styles() -> None:
    if JUDGMENT_CSS_PATH.exists():
        st.markdown(
            f"<style>{JUDGMENT_CSS_PATH.read_text()}</style>",
            unsafe_allow_html=True,
        )


def inject_workspace_styles(*, dense: bool) -> None:
    """Load base workspace CSS plus optional dense overrides."""
    if not WORKSPACE_CSS_PATH.exists():
        return

    raw = WORKSPACE_CSS_PATH.read_text()
    splitter = "/* === Dense Mode === */"
    base_css = raw
    dense_css = ""
    if splitter in raw:
        base_css, dense_css = raw.split(splitter, 1)
        dense_css = splitter + dense_css
    css = base_css + (dense_css if dense else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        /* Reduce spacing in right pane chase queue */
        .citation-workflow-rail {
            gap: 0.25rem !important;
        }
        .citation-workflow-rail > div {
            margin-bottom: 0.25rem !important;
            padding: 0.25rem !important;
        }
        div[data-testid="stHorizontalBlockGap"] > div:has(> .citation-workflow-rail) {
            gap: 0.25rem !important;
        }
        /* Make each column scroll independently - target Streamlit's column structure */
        section[data-testid="stVerticalBlock"] {
            overflow-y: auto !important;
            max-height: calc(100vh - 120px) !important;
        }
        /* Don't scroll the top area */
        header, div[data-testid="stHeader"], div[data-testid="stToolbar"] {
            position: sticky !important;
            top: 0 !important;
            z-index: 100 !important;
            background: var(--ws-app-bg) !important;
        }
        /* Make sure main content area scrolls */
        div[data-testid="stAppViewContainer"] > div {
            max-height: calc(100vh - 50px) !important;
            overflow-y: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_attachment_panel_styles() -> None:
    """Load CSS for the attachment queue workspace."""
    if ATTACHMENT_CSS_PATH.exists():
        st.markdown(
            f"<style>{ATTACHMENT_CSS_PATH.read_text()}</style>",
            unsafe_allow_html=True,
        )


def prepare_attachment_workspace() -> None:
    """Ensure attachment queue state and claim registry are hydrated."""
    attachment_queue.init_attachment_queue_state()
    claim_queue.sync_claims_from_results(st.session_state.get("results"))
    if not claim_queue.get_claim_records() and st.session_state.get("show_demo_claims"):
        claim_queue.ensure_demo_claims()
    if not st.session_state.get("_source_bin_loaded"):
        attachment_queue.sync_backend_state()
        st.session_state["_source_bin_loaded"] = True
    attachment_queue.ensure_open_when_activity()
    attachment_queue.collapse_when_idle()


def render_attachment_workspace() -> None:
    """Render claim drop zones, modal fallback, and queue panel."""
    inject_attachment_panel_styles()
    prepare_attachment_workspace()
    if attachment_queue.has_inflight_jobs():
        st.caption(
            (
                "Attachments are processing in the background. "
                "Use the queue panel to refresh statuses."
            )
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


def _get_background_state(api_url: str) -> Optional[dict]:
    url = (api_url or "").rstrip("/")
    if not url:
        return None
    try:
        response = requests.get(f"{url}/background/state", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _cmdf_snippet(text: str, *, words: int = 4) -> str:
    tokens = [tok for tok in re.findall(r"[A-Za-z0-9']+", str(text or "")) if tok]
    if not tokens:
        return ""
    n = min(int(words), len(tokens))
    if n <= 0:
        return ""
    if len(tokens) <= n:
        return " ".join(tokens)
    mid = len(tokens) // 2
    start = max(0, min(len(tokens) - n, mid - (n // 2)))
    return " ".join(tokens[start : start + n])


def _local_path_for_attachment(attachment_id: str) -> Optional[str]:
    if not attachment_id:
        return None
    snapshot = attachment_queue.get_queue_snapshot()
    for item in (snapshot.get("items") or {}).values():
        if item.get("attachment_id") != attachment_id:
            continue
        local_path = (item.get("local_path") or "").strip()
        if local_path and os.path.exists(local_path):
            return local_path
    return None


class _LocalUploadFile:
    def __init__(
        self,
        path: str,
        *,
        name: Optional[str] = None,
        content_type: str = "application/pdf",
    ):
        self.path = path
        self.name = name or os.path.basename(path)
        self.type = content_type

    def getbuffer(self):
        with open(self.path, "rb") as handle:
            return handle.read()


def _source_bin_status(item: dict) -> tuple[str, str]:
    status = (item.get("status") or "").strip().lower()
    claim_id = (item.get("claim_id") or "").strip()
    if status in {"", "none"}:
        return "Not started", "not-started"
    if status in {"pending", "converting", "parsing"}:
        return "Processing", "processing"
    if status == "error":
        return "Failed", "failed"
    if status == "matched":
        if claim_id:
            return "Placed", "placed"
        return "Needs placement", "needs-placement"
    return status.title(), "processing"


def _status_progress(status: str) -> Optional[float]:
    normalized = (status or "").strip().lower()
    if normalized == "pending":
        return 0.15
    if normalized == "converting":
        return 0.45
    if normalized == "parsing":
        return 0.75
    if normalized == "matched":
        return 1.0
    return None


def _render_source_bin_row(item: dict) -> None:
    queue_item_id = item.get("id")
    if not queue_item_id:
        return

    filename = str(item.get("filename") or "attachment.pdf")
    size_label = format_filesize(item.get("size"))
    archived = bool(item.get("archived"))
    status_label, status_class = _source_bin_status(item)
    claim_id = item.get("claim_id")
    claim_record = claim_queue.get_claim_record(str(claim_id)) if claim_id else None
    claim_label = (claim_record or {}).get("callout") if claim_record else None
    claim_text = (claim_record or {}).get("claim") if claim_record else None

    classes = "source-row source-row--compact" + (
        " source-row--archived" if archived else ""
    )
    st.markdown(f'<div class="{classes}">', unsafe_allow_html=True)

    header = st.columns([4, 1], gap="small")
    with header[0]:
        st.markdown(
            f"<div class='source-row__filename'>{html.escape(filename)}</div>",
            unsafe_allow_html=True,
        )
    with header[1]:
        st.markdown(
            (
                f"<span class='source-status-pill {status_class}'>"
                f"{html.escape(status_label)}"
                "</span>"
            ),
            unsafe_allow_html=True,
        )

    meta_bits = [size_label]
    if claim_label:
        meta_bits.append(f"Assigned: {claim_label}")
    st.caption(" • ".join(bit for bit in meta_bits if bit))

    progress = _status_progress(str(item.get("status") or ""))
    if progress is not None and progress < 1.0:
        st.progress(progress)

    relation_key = f"source-relation::{queue_item_id}"
    target_doc_key = f"source-target-doc::{queue_item_id}"
    assign_key = f"source-assign::{queue_item_id}"
    row = st.columns([3, 2], gap="small")

    with row[0]:
        st.radio(
            "Relationship",
            ["is cited by", "cites"],
            key=relation_key,
            horizontal=False,
            label_visibility="collapsed",
        )

        ledger_payload = _ledger_fetch(get_api_url(), force=False)
        ledger_rows = [
            r
            for r in (ledger_payload.get("rows") or [])
            if isinstance(r, dict) and r.get("ingest_id")
        ]

        def _target_doc_label(row_data: dict) -> str:
            short = str(row_data.get("short") or "").strip()
            title = str(row_data.get("title") or "").strip()
            words = [w for w in title.split() if w][:2]
            title_part = " ".join(words)
            if short and title_part:
                return f"{short} {title_part}..."
            if short:
                return short
            return title or "Document"

        target_options = [
            (str(r.get("ingest_id")), _target_doc_label(r))
            for r in ledger_rows
            if str(r.get("ingest_id") or "").strip()
        ]
        option_values = [v for v, _ in target_options]
        label_lookup = {v: lbl for v, lbl in target_options}
        num_by_ingest = {
            str(r.get("ingest_id")): int(r.get("num") or 0)
            for r in ledger_rows
            if str(r.get("ingest_id") or "").strip()
        }

        st.selectbox(
            "All placed documents",
            option_values,
            key=target_doc_key,
            format_func=lambda v: label_lookup.get(str(v), str(v)),
            label_visibility="collapsed",
        )

        if st.button(":material/check_circle:", key=f"source-place-submit::{queue_item_id}", use_container_width=True):
            relation = str(st.session_state.get(relation_key) or "is cited by")
            target_ingest_id = str(st.session_state.get(target_doc_key) or "").strip()
            if not target_ingest_id:
                st.info("Select a placed document first.")
            else:
                api_url = get_api_url()
                target_num = int(num_by_ingest.get(target_ingest_id) or 0)
                source_ingest_id = str(
                    (item.get("backend_details") or {}).get("source_ingest_id")
                    or item.get("doc_id")
                    or ""
                ).strip()
                if not source_ingest_id:
                    src_attachment_id = str(item.get("attachment_id") or queue_item_id)
                    try:
                        acting_user = str(_active_reviewer_uid() or "").strip()
                        if not acting_user:
                            raise RuntimeError(
                                "promote-ingest requires active reviewer identity"
                            )
                        promote_url = (
                            f"{str(api_url).rstrip('/')}/attachments/"
                            f"{src_attachment_id}/promote-ingest"
                        )
                        resp = requests.post(
                            promote_url,
                            headers={
                                "X-Project-Id": get_project_id(),
                                "X-User-Id": acting_user,
                            },
                            timeout=30,
                        )
                        resp.raise_for_status()
                        promoted = (resp.json() or {}).get("attachment") or {}
                        source_ingest_id = str(
                            promoted.get("source_ingest_id")
                            or promoted.get("doc_id")
                            or ""
                        ).strip()
                        if promoted:
                            item["backend_details"] = promoted
                    except Exception:
                        source_ingest_id = ""

                source_num = int(num_by_ingest.get(source_ingest_id) or 0)
                if source_num <= 0:
                    st.info(
                        "This stray document is not yet in the ledger as a placeable work. "
                        "Wait for processing, then retry placement."
                    )
                elif target_num <= 0:
                    st.info("Selected target is not a placeable ledger work.")
                else:
                    try:
                        st.session_state["ledger_payload"] = ledger_api.place_relation(
                            api_url,
                            source_num=source_num,
                            target_num=target_num,
                            relation=relation,
                            canonical=False,
                            reviewer_uid=_active_reviewer_uid(),
                            project_id=get_project_id(),
                            user_id=_active_reviewer_uid(),
                        )
                    except RuntimeError as exc:
                        st.error(str(exc))
                    else:
                        st.caption(
                            "Placement suggestion recorded for review: "
                            f"{label_lookup.get(target_ingest_id, target_ingest_id)}"
                        )
                        try:
                            callout = st.session_state.get("selected_callout_tuple") or {}
                            active_doc = str(st.session_state.get("selected_doc_id") or "").strip()
                            callout_doc = str(callout.get("doc_id") or "").strip()
                            ref_id = str(callout.get("target_id") or "").strip()
                            callout_matches_target = bool(
                                target_ingest_id
                                and (active_doc == target_ingest_id or callout_doc == target_ingest_id)
                            )
                            if (
                                relation == "is cited by"
                                and source_ingest_id
                                and target_ingest_id
                                and ref_id
                                and callout_matches_target
                            ):
                                st.session_state["ledger_payload"] = ledger_api.place_reference(
                                    api_url,
                                    citing_doc_id=target_ingest_id,
                                    reference_id=ref_id,
                                    cited_ingest_id=source_ingest_id,
                                    canonical=False,
                                    reviewer_uid=_active_reviewer_uid(),
                                    project_id=get_project_id(),
                                    user_id=_active_reviewer_uid(),
                                )
                        except RuntimeError as exc:
                            st.warning(str(exc))
                        st.session_state.pop("citation_context_cache", None)
                        st.session_state.pop("citation_context", None)
                        _ledger_fetch(api_url, force=True)
                        _rerun()

    def _on_assign_change() -> None:
        selected_claim_id = st.session_state.get(assign_key)
        if not selected_claim_id:
            return
        callout_tuple = st.session_state.get("selected_callout_tuple") or {}
        claim_rec = claim_queue.get_claim_record(str(selected_claim_id)) or {}

        is_global_source = (
            not str(item.get("claim_id") or "").strip()
            and not str(item.get("doc_id") or "").strip()
            and not str(item.get("target_id") or "").strip()
        )

        doc_id = claim_rec.get("doc_id") or st.session_state.get("selected_doc_id")
        citation_index = callout_tuple.get("citation_index")
        target_id = callout_tuple.get("target_id")

        if is_global_source:
            api_url = get_api_url()
            src_attachment_id = str(item.get("attachment_id") or queue_item_id)
            reviewer_uid = str(_active_reviewer_uid() or "").strip()
            if not reviewer_uid:
                st.warning("Select an active reviewer before assigning sources.")
                return
            headers = {
                "X-Project-Id": get_project_id(),
                "X-User-Id": reviewer_uid,
                "X-Reviewer-Uid": reviewer_uid,
            }

            # Ensure source_ingest_id exists when possible (helps graph auto-place).
            try:
                details = item.get("backend_details") or {}
                if not str(details.get("source_ingest_id") or "").strip():
                    promote_url = (
                        f"{str(api_url).rstrip('/')}/attachments/"
                        f"{src_attachment_id}/promote-ingest"
                    )
                    resp = requests.post(
                        promote_url,
                        headers=headers,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    promoted = (resp.json() or {}).get("attachment") or {}
                    if promoted:
                        item["backend_details"] = promoted
            except Exception:
                # Best-effort; clone still works without promote.
                pass

            payload = {
                "claim_id": str(selected_claim_id),
                "doc_id": str(doc_id) if doc_id else None,
                "citation_index": int(citation_index)
                if citation_index is not None
                else None,
                "target_id": str(target_id) if target_id else None,
                "reference_hint": {"reference_id": str(target_id)}
                if target_id
                else None,
                "claim_text": (claim_rec.get("claim") or "").strip() or None,
            }
            payload = {k: v for k, v in payload.items() if v is not None}

            try:
                resp = requests.post(
                    f"{str(api_url).rstrip('/')}/attachments/{src_attachment_id}/clone",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
            except Exception as exc:
                st.warning(f"Assign failed: {exc}")
                return
        else:
            attachment_queue.place_attachment(
                str(queue_item_id),
                str(selected_claim_id),
                doc_id=doc_id,
                citation_index=citation_index,
                target_id=target_id,
                via="manual",
            )

        _rerun()

    with row[1]:
        options = claim_queue.get_claim_options()
        if not options:
            st.caption("No claims yet.")
        else:
            claim_ids = [option.get("id") for option in options if option.get("id")]
            label_lookup = {option["id"]: option["label"] for option in options}
            assign_options = [None] + claim_ids
            default_idx = 0
            current_claim = str(claim_id) if claim_id else None
            if current_claim in claim_ids:
                default_idx = assign_options.index(current_claim)
            if assign_key in st.session_state:
                st.selectbox(
                    "Assign",
                    assign_options,
                    key=assign_key,
                    format_func=lambda v: "Assign/Re-place…"
                    if v is None
                    else label_lookup.get(str(v), str(v)),
                    on_change=_on_assign_change,
                    label_visibility="collapsed",
                )
            else:
                st.selectbox(
                    "Assign",
                    assign_options,
                    index=default_idx,
                    key=assign_key,
                    format_func=lambda v: "Assign/Re-place…"
                    if v is None
                    else label_lookup.get(str(v), str(v)),
                    on_change=_on_assign_change,
                    label_visibility="collapsed",
                )

    st.markdown("</div>", unsafe_allow_html=True)


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
                "converting before extraction completes."
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


def _format_ledger_bracket(nums: List[int]) -> str:
    cleaned = [int(n) for n in (nums or []) if n]
    if not cleaned:
        return "[]"
    shown = cleaned[:6]
    inside = ", ".join(str(n) for n in shown)
    if len(cleaned) > len(shown):
        inside = inside + ", ..."
    return f"[{inside}]"


def _ledger_fetch(api_url: str, *, force: bool = False) -> dict:
    payload = st.session_state.get("ledger_payload")
    if force or not isinstance(payload, dict):
        payload = None
    if payload is None or force:
        try:
            payload = ledger_api.get_ledger(api_url, project_id=get_project_id())
        except RuntimeError as exc:
            payload = {"rows": [], "options": [], "error": str(exc)}
        st.session_state["ledger_payload"] = payload
    return payload or {}


def _ledger_rows_cached() -> list[dict[str, Any]]:
    payload = st.session_state.get("ledger_payload") or {}
    rows = payload.get("rows") if isinstance(payload, dict) else []
    return rows if isinstance(rows, list) else []


def render_documents_panel(*, max_rows: Optional[int] = None) -> None:
    api_url = get_api_url()
    controls = st.columns([2, 1], gap="small")
    with controls[0]:
        st.text_input(
            "Search",
            key="docs-search",
            placeholder="Search author, title, DOI, or #",
            label_visibility="collapsed",
        )
    with controls[1]:
        if st.button("Refresh", key="ledger-refresh", use_container_width=True):
            _ledger_fetch(api_url, force=True)

    st.toggle(
        "Show placeholders",
        key="docs-show-placeholders",
        help="Show documents that only exist as bibliography entries (no PDF yet).",
    )

    st.caption("Upload PDFs via Upload documents (top of left pane).")

    payload = _ledger_fetch(api_url, force=False)
    if payload.get("error"):
        st.caption(str(payload.get("error")))

    rows = payload.get("rows") or []
    if not rows:
        st.caption(
            "No ledger entries yet. Upload a citing document and run extraction."
        )
        return

    options = payload.get("options") or []
    option_nums = [int(opt.get("num")) for opt in options if opt.get("num")]
    option_by_num = {int(opt.get("num")): opt for opt in options if opt.get("num")}

    ingest_status_by_id: dict[str, dict[str, str]] = {}
    for doc in (st.session_state.get("ingested_docs") or []):
        if not isinstance(doc, dict):
            continue
        iid = str(doc.get("id") or "").strip()
        if not iid:
            continue
        extraction = doc.get("extraction") or {}
        body_extraction = doc.get("body_extraction") or {}
        resolution = doc.get("resolution") or {}
        ingest_status_by_id[iid] = {
            "extraction_status": str(extraction.get("status") or "").strip().lower(),
            "body_extraction_status": str(body_extraction.get("status") or "").strip().lower(),
            "resolution_status": str(resolution.get("status") or "").strip().lower(),
        }

    q = (st.session_state.get("docs-search") or "").strip().lower()
    show_placeholders = bool(st.session_state.get("docs-show-placeholders"))
    filtered = []
    for row in rows:
        if not show_placeholders and not bool(row.get("anchored")):
            continue
        hay = " ".join(
            str(x or "")
            for x in [
                row.get("num"),
                row.get("short"),
                row.get("title"),
                row.get("apa"),
                row.get("doi"),
            ]
        ).lower()
        if q and q not in hay and (not q.startswith("#") or q[1:] not in hay):
            continue
        filtered.append(row)

    if not show_placeholders:
        hidden = sum(1 for r in rows if not bool(r.get("anchored")))
        if hidden:
            st.caption(f"Hiding {hidden} placeholder(s).")

    selected_num = st.session_state.get("documents_selected_num")
    if selected_num is None and filtered:
        selected_num = int(filtered[0].get("num") or 0)
        st.session_state["documents_selected_num"] = selected_num

    def _open_process_doc(ingest_id: str, *, reprocess: bool = False) -> None:
        st.session_state["selected_doc_id"] = str(ingest_id)
        doc = load_selected_document(show_error=False) or {}
        if not doc:
            return
        extraction = doc.get("extraction") or {}
        resolution = doc.get("resolution") or {}
        extraction_done = (extraction.get("status") or "").strip().lower() == "complete"
        resolution_done = (resolution.get("status") or "").strip().lower() == "complete"

        if reprocess:
            extraction_done = False
            resolution_done = False

        if not extraction_done:
            with st.spinner("Extracting..."):
                try:
                    trigger_extraction(
                        api_url,
                        str(ingest_id),
                        project_id=get_project_id(),
                        user_id=_active_reviewer_uid(),
                        force=bool(reprocess),
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                    return
            doc = load_selected_document(show_error=False) or doc
        if not resolution_done:
            with st.spinner("Resolving references..."):
                try:
                    trigger_resolution(
                        api_url,
                        str(ingest_id),
                        project_id=get_project_id(),
                        user_id=_active_reviewer_uid(),
                        force=bool(reprocess),
                    )
                except RuntimeError:
                    # Resolution is optional; don't block opening.
                    pass
            doc = load_selected_document(show_error=False) or doc
        refresh_ingested_docs(show_error=False)
        _ledger_fetch(api_url, force=True)

    filtered_total = len(filtered)
    if max_rows is not None and filtered_total > int(max_rows):
        st.caption(f"Showing {int(max_rows)} of {filtered_total} documents")
        filtered = filtered[: int(max_rows)]

    seen_docs = {}
    deduplicated = []
    for row in filtered:
        short = str(row.get("short") or "").lower().strip()
        title = str(row.get("title") or "").lower().strip()
        if not short:
            short = str(row.get("title") or "").lower().strip()
        doc_key = short or title
        if not doc_key:
            continue
        existing = seen_docs.get(doc_key)
        if existing is None:
            seen_docs[doc_key] = row
            deduplicated.append(row)
            continue

        existing_score = int(bool(existing.get("ingest_id"))) + int(
            str(existing.get("status") or "").strip().lower() == "green"
        )
        new_score = int(bool(row.get("ingest_id"))) + int(
            str(row.get("status") or "").strip().lower() == "green"
        )
        existing_score += int(bool(existing.get("extracted"))) + int(
            bool(existing.get("resolved"))
        )
        new_score += int(bool(row.get("extracted"))) + int(bool(row.get("resolved")))
        if new_score > existing_score:
            seen_docs[doc_key] = row
            try:
                idx = deduplicated.index(existing)
                deduplicated[idx] = row
            except Exception:
                pass

    for row in deduplicated:
        try:
            num = int(row.get("num"))
        except Exception:
            continue
        status = str(row.get("status") or "orange").strip().lower()
        status_icon = (
            "check_circle" if status == "green" else "running_with_errors"
        )
        short = str(row.get("short") or f"Document {num}")
        apa = str(row.get("apa") or short)
        ingest_id = row.get("ingest_id")
        extracted = bool(row.get("extracted"))
        resolved = bool(row.get("resolved"))

        incoming = row.get("incoming") or []
        outgoing = row.get("outgoing") or []
        incoming_live = (
            row.get("incoming_live")
            if row.get("incoming_live") is not None
            else incoming
        )
        outgoing_live = (
            row.get("outgoing_live")
            if row.get("outgoing_live") is not None
            else outgoing
        )
        suggested_pending = bool(
            (row.get("incoming_suggested_live") or [])
            or (row.get("outgoing_suggested_live") or [])
        )

        deg_html = (
            f"<span class='doc-degree'>"
            f"<sup>{len(incoming_live)}</sup><sub>{len(outgoing_live)}</sub>"
            f"</span>"
        )

        title = row.get("title") or ""
        title_hint = f" - {title}" if title else ""
        ingest_snapshot = ingest_status_by_id.get(str(ingest_id or "").strip()) or {}
        extraction_status = _ledger_canonical_stage(
            row,
            ingest_snapshot,
            stage="extraction",
            field="status",
        )
        extraction_error = _ledger_canonical_stage(
            row,
            ingest_snapshot,
            stage="extraction",
            field="error",
        )
        body_status = _ledger_canonical_stage(
            row,
            ingest_snapshot,
            stage="body_extraction",
            field="status",
        )
        body_error = _ledger_canonical_stage(
            row,
            ingest_snapshot,
            stage="body_extraction",
            field="error",
        )
        resolution_status = _ledger_canonical_stage(
            row,
            ingest_snapshot,
            stage="resolution",
            field="status",
        )
        resolution_error = _ledger_canonical_stage(
            row,
            ingest_snapshot,
            stage="resolution",
            field="error",
        )
        extracted = extracted or extraction_status == "complete"
        resolved = resolved or resolution_status == "complete"
        chips = []
        if (
            extraction_status == "running"
            or body_status == "running"
            or resolution_status == "running"
        ):
            chips.append("clock_loader_10")
        elif body_status == "error":
            chips.append("running_with_errors")
        elif extraction_status == "error" or resolution_status == "error":
            chips.append("running_with_errors")
        elif suggested_pending:
            chips.append("running_with_errors")
        elif extracted and resolved:
            chips.append("check_circle")
        else:
            chips.append("incomplete_circle")

        tooltip_lines = [apa]
        if extraction_status == "error" and extraction_error:
            tooltip_lines.append(f"Extraction error: {extraction_error}")
        if body_status == "error" and body_error:
            tooltip_lines.append(f"Body error: {body_error}")
        if resolution_status == "error" and resolution_error:
            tooltip_lines.append(f"Resolution error: {resolution_error}")

        row_icon = chips[0] if chips else status_icon
        cols = st.columns([0.7, 6.3, 0.9, 0.7], gap="small")
        with cols[0]:
            st.markdown(f":material/{row_icon}:")
        with cols[1]:
            selected = bool(
                int(st.session_state.get("documents_selected_num") or 0) == int(num)
            )
            label = f"{short}{title_hint}"
            if selected:
                label = f"[selected] {label}"
            if st.button(
                label,
                key=f"ledger-open-{int(num)}",
                use_container_width=True,
                help="\n".join(tooltip_lines),
            ):
                st.session_state["documents_selected_num"] = int(num)
                if ingest_id:
                    st.session_state["selected_doc_id"] = str(ingest_id)
                    load_selected_document(show_error=False)
                payload = {"docnum": str(int(num))}
                if ingest_id:
                    payload["doc"] = str(ingest_id)
                scope_uid = scope_lock.get_applied_uid()
                scope_project = scope_lock.get_applied_project_id()
                if scope_uid:
                    payload["uid"] = scope_uid
                if scope_project:
                    payload["project"] = scope_project
                try:
                    st.query_params.clear()  # type: ignore[attr-defined]
                    for key, value in payload.items():
                        st.query_params[key] = value  # type: ignore[attr-defined]
                except Exception:
                    try:
                        st.experimental_set_query_params(**payload)
                    except Exception:
                        pass
                _rerun()
        with cols[2]:
            st.markdown(deg_html, unsafe_allow_html=True)
        with cols[3]:
            st.markdown(":material/open_in_browser:")

    # Editor for currently selected document.
    editor_doc = int(st.session_state.get("documents_selected_num") or 0)
    selected_row = next((r for r in rows if int(r.get("num") or 0) == editor_doc), None)
    if not selected_row:
        return

    st.divider()
    st.markdown(f"**Selected: #{editor_doc}**")

    ingest_id = selected_row.get("ingest_id")
    if ingest_id:
        selected_snapshot = ingest_status_by_id.get(str(ingest_id or "").strip()) or {}
        extraction_status = _ledger_canonical_stage(
            selected_row,
            selected_snapshot,
            stage="extraction",
            field="status",
        )
        resolution_status = _ledger_canonical_stage(
            selected_row,
            selected_snapshot,
            stage="resolution",
            field="status",
        )
        if extraction_status == "error":
            if st.button(
                "Retry extraction",
                key=f"doc-editor-retry-extract-{editor_doc}",
                use_container_width=True,
            ):
                with st.spinner("Retrying extraction..."):
                    trigger_extraction(
                        api_url,
                        str(ingest_id),
                        project_id=get_project_id(),
                        user_id=_active_reviewer_uid(),
                    )
                refresh_ingested_docs(show_error=False)
                _ledger_fetch(api_url, force=True)
        if resolution_status == "error":
            if st.button(
                "Retry resolution",
                key=f"doc-editor-retry-resolve-{editor_doc}",
                use_container_width=True,
            ):
                with st.spinner("Retrying resolution..."):
                    try:
                        trigger_resolution(
                            api_url,
                            str(ingest_id),
                            project_id=get_project_id(),
                            user_id=_active_reviewer_uid(),
                        )
                    except RuntimeError as exc:
                        st.error(str(exc))
                refresh_ingested_docs(show_error=False)
                _ledger_fetch(api_url, force=True)

        if st.button(
            "Open", key=f"doc-editor-open-{editor_doc}", use_container_width=True
        ):
            _open_process_doc(str(ingest_id), reprocess=False)
        if st.button(
            "Reprocess + open",
            key=f"doc-editor-reprocess-{editor_doc}",
            use_container_width=True,
        ):
            _open_process_doc(str(ingest_id), reprocess=True)

    assigned_key = f"ledger-assigned::{editor_doc}"
    st.session_state.setdefault(assigned_key, bool(selected_row.get("assigned")))
    if st.toggle("Assigned to workflow", key=assigned_key):
        pass
    if st.button(
        "Save assignment",
        key=f"ledger-assign-save-{editor_doc}",
        use_container_width=True,
    ):
        try:
            st.session_state["ledger_payload"] = ledger_api.set_assigned(
                api_url,
                int(editor_doc),
                bool(st.session_state.get(assigned_key)),
                project_id=get_project_id(),
                user_id=_active_reviewer_uid(),
            )
        except RuntimeError as exc:
            st.error(str(exc))

    def _format_opt(num: int) -> str:
        opt = option_by_num.get(int(num)) or {}
        short = str(opt.get("short") or f"#{num}")
        return f"#{int(num)} {short}"

    # Linking is rarely needed now that uploads auto-merge into the tree.
    if bool(st.session_state.get("docs-show-placeholders")):
        with st.expander("Links (advanced)", expanded=False):
            st.caption("Edit citation links in the local graph.")

            last_key = "_docs_editor_last"
            prev = st.session_state.get(last_key)
            if prev != editor_doc:
                st.session_state.pop(f"ledger-links::incoming::{editor_doc}", None)
                st.session_state.pop(f"ledger-links::outgoing::{editor_doc}", None)
                st.session_state[last_key] = editor_doc

            in_key = f"ledger-links::incoming::{editor_doc}"
            out_key = f"ledger-links::outgoing::{editor_doc}"
            incoming_default = [
                int(n) for n in (selected_row.get("incoming") or []) if n
            ]
            outgoing_default = [
                int(n) for n in (selected_row.get("outgoing") or []) if n
            ]

            st.markdown("**Cited by**")
            incoming_chosen = st.multiselect(
                "Incoming",
                options=[n for n in option_nums if int(n) != int(editor_doc)],
                default=incoming_default,
                format_func=_format_opt,
                key=in_key,
                label_visibility="collapsed",
            )
            if st.button(
                "Save cited-by",
                key=f"ledger-in-apply-{editor_doc}",
                use_container_width=True,
            ):
                try:
                    st.session_state["ledger_payload"] = ledger_api.set_incoming(
                        api_url,
                        int(editor_doc),
                        [int(n) for n in incoming_chosen],
                        project_id=get_project_id(),
                        user_id=_active_reviewer_uid(),
                    )
                except RuntimeError as exc:
                    st.error(str(exc))

            st.markdown("**Cites**")
            outgoing_chosen = st.multiselect(
                "Outgoing",
                options=[n for n in option_nums if int(n) != int(editor_doc)],
                default=outgoing_default,
                format_func=_format_opt,
                key=out_key,
                label_visibility="collapsed",
            )
            if st.button(
                "Save cites",
                key=f"ledger-out-apply-{editor_doc}",
                use_container_width=True,
            ):
                try:
                    st.session_state["ledger_payload"] = ledger_api.set_outgoing(
                        api_url,
                        int(editor_doc),
                        [int(n) for n in outgoing_chosen],
                        project_id=get_project_id(),
                        user_id=_active_reviewer_uid(),
                    )
                except RuntimeError as exc:
                    st.error(str(exc))


def _project_fetch_meta(api_url: str, *, force: bool = False) -> dict:
    if not scope_lock.has_applied_scope():
        return {}
    meta = st.session_state.get("project_meta")
    if force or not isinstance(meta, dict):
        meta = None
    if meta is None or force:
        try:
            meta = project_api.get_meta(
                project_id=get_project_id(),
                user_id=_active_reviewer_uid(),
            )
        except project_api.ProjectApiError as exc:
            meta = {"name": "default", "error": str(exc)}
        st.session_state["project_meta"] = meta
    return meta or {}


def _normalize_reviewer_name(value: str | None) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    return text


def _project_reviewers(meta: dict) -> list[str]:
    raw = meta.get("reviewers")
    if not isinstance(raw, list):
        return []
    return [
        _normalize_reviewer_name(item) for item in raw if _normalize_reviewer_name(item)
    ]


def _project_active_reviewer_uid(meta: dict) -> Optional[str]:
    active = _normalize_reviewer_name(meta.get("active_reviewer_uid"))
    return active or None


def _active_reviewer_uid() -> Optional[str]:
    value = scope_lock.get_applied_uid()
    return value or None


def _reviewer_state_suffix(reviewer_uid: Optional[str]) -> str:
    """Stable, safe suffix for session_state widget keys.

    Streamlit widget state is keyed only by the widget key string.
    If we reuse the same key across reviewers, switching Current user will
    keep the prior user's draft inputs (notes/verdict/etc) in the UI.

    We *do not* want to wipe stored work when changing users; we only want
    the UI drafts to be independent per reviewer.
    """
    norm = _normalize_reviewer_name(reviewer_uid)
    return (norm or "default").casefold()


def _normalize_cited_work_id(value: Optional[str]) -> Optional[str]:
    from frontend.citation_anchors import normalize_cited_work_id

    return normalize_cited_work_id(value)


def _build_anchor_quote(window_text: str) -> dict:
    from frontend.citation_anchors import build_anchor_quote

    return build_anchor_quote(window_text)


def _maybe_attach_citation_anchor(provenance: dict) -> dict:
    """Attach stable citation anchoring fields to a provenance dict.

    This keeps cross-reviewer agreement meaningful even when:
    - the PDF is reprocessed (citation indices / target IDs may drift)
    - reviewers segment the citing sentence differently
    """
    from frontend.citation_anchors import maybe_attach_citation_anchor

    return maybe_attach_citation_anchor(
        provenance=provenance,
        context=st.session_state.get("citation_context"),
    )


def _applied_scope_badge() -> str:
    uid = scope_lock.get_applied_uid()
    project_id = scope_lock.get_applied_project_id()
    if uid and project_id:
        return f"Scope: uid={uid} • project={project_id}"
    return "Scope: unapplied"


def _runtime_stamp_token(*, stamp: str, git_sha: str, image_tag: str) -> str:
    value = str(stamp or "").strip() or "unknown"
    sha = str(git_sha or "").strip()
    tag = str(image_tag or "").strip()
    if sha:
        value = f"{value}@{sha[:12]}"
    if tag:
        value = f"{value} ({tag})"
    return value


def _ui_runtime_stamp_token() -> str:
    return _runtime_stamp_token(
        stamp=str(getattr(settings, "UI_RUNTIME_STAMP", "") or ""),
        git_sha=str(getattr(settings, "UI_GIT_SHA", "") or ""),
        image_tag=str(getattr(settings, "UI_IMAGE_TAG", "") or ""),
    )


def _api_runtime_stamp_token(*, api_url: str) -> str:
    now = float(time.time())
    cache = st.session_state.get("_api_runtime_stamp_cache")
    if isinstance(cache, dict):
        age = now - float(cache.get("fetched_at") or 0.0)
        if str(cache.get("api_url") or "") == str(api_url or "") and age <= float(
            RUNTIME_STAMP_CACHE_TTL_SECONDS
        ):
            cached_value = str(cache.get("token") or "").strip()
            if cached_value:
                return cached_value

    token = "unavailable"
    try:
        resp = requests.get(
            f"{str(api_url).rstrip('/')}/scope/observability/runtime-stamp",
            timeout=3,
        )
        resp.raise_for_status()
        raw_payload = resp.json()
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        token = _runtime_stamp_token(
            stamp=str(payload.get("api_runtime_stamp") or ""),
            git_sha=str(payload.get("api_git_sha") or ""),
            image_tag=str(payload.get("api_image_tag") or ""),
        )
    except Exception:
        token = "unavailable"

    st.session_state["_api_runtime_stamp_cache"] = {
        "api_url": str(api_url or ""),
        "fetched_at": now,
        "token": token,
    }
    return token


def _runtime_stamp_line() -> str:
    ui_token = _ui_runtime_stamp_token()
    api_token = _api_runtime_stamp_token(api_url=get_api_url())
    return f"Runtime: ui={ui_token} • api={api_token}"


def _render_scope_runtime_block() -> None:
    st.caption(_applied_scope_badge())
    st.caption(_runtime_stamp_line())


def _normalize_scope_user_ids(payload: Any) -> list[str]:
    raw_users: list[Any] = []
    if isinstance(payload, dict):
        candidate = payload.get("users")
        if isinstance(candidate, list):
            raw_users = candidate
    elif isinstance(payload, list):
        raw_users = payload

    normalized: list[str] = []
    seen: set[str] = set()
    for row in raw_users:
        if isinstance(row, dict):
            candidate = row.get("user_id") or row.get("id")
        else:
            candidate = row
        user_id = str(candidate or "").strip()
        if not user_id or user_id in seen:
            continue
        normalized.append(user_id)
        seen.add(user_id)
    return normalized


def _clear_unapplied_scope_workspace_state() -> None:
    for key in (
        "selected_doc_id",
        "active_document",
        "citation_selected_index",
        "citation_selected_target",
        "citation_selected_sentence_id",
        "selected_callout_tuple",
        "pending_citation_selection",
        "workflow_active_citation",
        "citation_context_key",
        "citation_context",
        "citation_context_error",
        "citation_last_context_request",
        "citation_graph_key",
        "citation_graph",
        "citation_graph_error",
        "citation_last_graph_request",
        "citation_context_cache",
        "graph_nav_contexts_cache",
    ):
        st.session_state.pop(key, None)
    st.session_state["selected_doc_id"] = ""
    st.session_state["active_document"] = None


def _invalidate_scope_cached_state() -> None:
    st.session_state["project_meta"] = None
    st.session_state["ingested_docs"] = None
    st.session_state["_ingested_docs_loaded"] = False
    st.session_state.pop("ledger_payload", None)
    st.session_state["selected_doc_id"] = ""
    st.session_state["active_document"] = None
    st.session_state.pop("_followed_citations_cache", None)
    st.session_state.pop("graph_nav_contexts_cache", None)


def render_scope_selector_block() -> None:
    scope_lock.ensure_seeded()

    def _read_scope_params() -> tuple[str, str]:
        try:
            raw = st.query_params  # type: ignore[attr-defined]
            uid_raw = raw.get("uid")
            project_raw = raw.get("project")
        except Exception:
            try:
                raw2 = st.experimental_get_query_params()
                uid_raw = raw2.get("uid")
                project_raw = raw2.get("project")
            except Exception:
                uid_raw = None
                project_raw = None
        if isinstance(uid_raw, list):
            uid_raw = uid_raw[0] if uid_raw else None
        if isinstance(project_raw, list):
            project_raw = project_raw[0] if project_raw else None
        return str(uid_raw or "").strip(), str(project_raw or "").strip()

    qp_uid, qp_project = _read_scope_params()
    st.session_state.setdefault("scope_bootstrap_error", None)
    scope_lock.sync_from_backend(preferred_user=qp_uid)
    sync_error = str(st.session_state.get("scope_sync_error") or "").strip()
    if sync_error and qp_uid and not scope_lock.has_applied_scope():
        st.session_state["scope_bootstrap_error"] = (
            f"Unable to sync scope session for {qp_uid}: {sync_error}"
        )
        _clear_unapplied_scope_workspace_state()

    if not scope_lock.has_applied_scope():
        if qp_uid and qp_project:
            st.session_state[SCOPE_DRAFT_UID] = qp_uid
            st.session_state[SCOPE_DRAFT_PROJECT_ID] = qp_project
            try:
                project_api.select_project(user_id=qp_uid, project_id=qp_project)
                scope_lock.apply_draft_scope(persist_backend=True)
                st.session_state["scope_bootstrap_error"] = None
            except Exception as exc:
                st.session_state["scope_bootstrap_error"] = (
                    "Failed to apply scope from URL parameters; "
                    "open Scope selector and click Apply/Switch with valid membership "
                    f"(details: {exc})"
                )
                _clear_unapplied_scope_workspace_state()

    NEW_USER_OPTION = "__scope_new_user__"
    NEW_PROJECT_OPTION = "__scope_new_project__"

    def _project_label(pid: str, pname: str) -> str:
        pid_text = str(pid or "").strip()
        name_text = str(pname or "").strip()
        if name_text and name_text != pid_text:
            return f"{name_text} ({pid_text})"
        if len(pid_text) >= 32 and "-" in pid_text:
            return f"Untitled ({pid_text[:8]})"
        return pid_text

    st.markdown("**Scope selector**")
    st.caption("Set draft scope and click Apply/Switch to unlock activity.")

    draft_uid = scope_lock.get_draft_uid()
    draft_project_id = scope_lock.get_draft_project_id()

    user_options: list[str] = []
    try:
        user_listing = project_api.list_users()
        user_options = _normalize_scope_user_ids(user_listing)
        if isinstance(user_listing, (dict, list)):
            st.session_state["scope_users_error"] = None
        else:
            st.session_state["scope_users_error"] = "invalid user-list payload"
    except project_api.ProjectApiError as exc:
        st.session_state["scope_users_error"] = str(exc)
    except Exception as exc:
        st.session_state["scope_users_error"] = str(exc)

    for fallback_uid in (
        scope_lock.get_applied_uid(),
        draft_uid,
        str(st.session_state.get("scope_new_uid") or "").strip(),
    ):
        if fallback_uid and fallback_uid not in user_options:
            user_options.append(fallback_uid)

    user_select_options = [""] + sorted(set(user_options)) + [NEW_USER_OPTION]
    if draft_uid and draft_uid in user_options:
        user_index = user_select_options.index(draft_uid)
    elif draft_uid:
        st.session_state["scope_new_uid"] = draft_uid
        user_index = user_select_options.index(NEW_USER_OPTION)
    else:
        user_index = 0

    selected_user = st.selectbox(
        "User ID",
        options=user_select_options,
        index=user_index,
        format_func=lambda value: (
            "Select user..."
            if value == ""
            else "+ New user..."
            if value == NEW_USER_OPTION
            else str(value)
        ),
    )
    if selected_user == NEW_USER_OPTION:
        st.text_input(
            "New user ID",
            key="scope_new_uid",
            placeholder="reviewer-a",
            help="Draft only until Apply/Switch.",
        )
        st.session_state[SCOPE_DRAFT_UID] = str(
            st.session_state.get("scope_new_uid") or ""
        ).strip()
    else:
        st.session_state[SCOPE_DRAFT_UID] = str(selected_user or "").strip()
        if selected_user:
            st.session_state["scope_new_uid"] = ""

    draft_uid = scope_lock.get_draft_uid()
    draft_project_id = scope_lock.get_draft_project_id()

    project_options: list[str] = []
    project_labels: dict[str, str] = {}
    active_project_id = ""
    if draft_uid:
        try:
            listing = project_api.list_projects(user_id=draft_uid)
            projects = listing.get("projects") or []
            project_options = [
                str((row or {}).get("project_id") or "").strip()
                for row in projects
                if str((row or {}).get("project_id") or "").strip()
            ]
            for row in projects:
                if not isinstance(row, dict):
                    continue
                pid = str((row or {}).get("project_id") or "").strip()
                if not pid:
                    continue
                pname = str((row or {}).get("name") or "").strip()
                project_labels[pid] = _project_label(pid, pname)
            active_project_id = str(listing.get("active_project_id") or "").strip()
            st.session_state["scope_projects_error"] = None
            st.session_state["scope_projects_last_uid"] = draft_uid
        except project_api.ProjectApiError as exc:
            st.session_state["scope_projects_error"] = str(exc)

    if not draft_project_id and active_project_id:
        st.session_state[SCOPE_DRAFT_PROJECT_ID] = active_project_id
        draft_project_id = active_project_id

    if draft_uid:
        select_options = [""] + sorted(set(project_options)) + [NEW_PROJECT_OPTION]
        if draft_project_id and draft_project_id in project_options:
            select_index = select_options.index(draft_project_id)
        elif draft_project_id:
            st.session_state["scope_new_project_id"] = draft_project_id
            select_index = select_options.index(NEW_PROJECT_OPTION)
        else:
            select_index = 0
        selected_project = st.selectbox(
            "Project ID",
            options=select_options,
            index=select_index,
            format_func=lambda v: (
                "Select project..."
                if not v
                else "+ New project..."
                if str(v) == NEW_PROJECT_OPTION
                else project_labels.get(str(v), str(v))
            ),
        )
        if str(selected_project) == NEW_PROJECT_OPTION:
            st.text_input(
                "New project ID",
                key="scope_new_project_id",
                placeholder="project-a",
                help="Draft only until Apply/Switch.",
            )
            st.session_state[SCOPE_DRAFT_PROJECT_ID] = str(
                st.session_state.get("scope_new_project_id") or ""
            ).strip()
        else:
            st.session_state[SCOPE_DRAFT_PROJECT_ID] = str(selected_project or "").strip()
            if selected_project:
                st.session_state["scope_new_project_id"] = ""
        explicit_new_project_intent = str(selected_project) == NEW_PROJECT_OPTION
    else:
        st.selectbox(
            "Project ID",
            options=[""],
            index=0,
            format_func=lambda _value: "Select a user first",
            disabled=True,
        )
        st.session_state[SCOPE_DRAFT_PROJECT_ID] = ""
        explicit_new_project_intent = False

    draft_uid = scope_lock.get_draft_uid()
    draft_project_id = scope_lock.get_draft_project_id()
    applied_uid = scope_lock.get_applied_uid()
    applied_project_id = scope_lock.get_applied_project_id()
    has_applied = scope_lock.has_applied_scope()
    is_changed = (draft_uid, draft_project_id) != (applied_uid, applied_project_id)

    action_label = "Switch" if has_applied else "Apply"
    if st.button(
        action_label,
        key="scope-apply-switch",
        disabled=not bool(draft_uid and draft_project_id and is_changed),
        use_container_width=True,
    ):
        try:
            try:
                project_api.select_project(user_id=draft_uid, project_id=draft_project_id)
                scope_lock.apply_draft_scope(persist_backend=True)
            except project_api.ProjectApiError:
                if not explicit_new_project_intent:
                    raise
                created = project_api.create_project(
                    user_id=draft_uid,
                    project_id=draft_project_id,
                )
                created_project_id = str(
                    created.get("active_project_id")
                    or ((created.get("project") or {}).get("project_id"))
                    or ""
                ).strip()
                if not created_project_id:
                    raise project_api.ProjectApiError(
                        "project creation did not return project_id"
                    )
                st.session_state[SCOPE_DRAFT_PROJECT_ID] = created_project_id
                st.success(f"Created project {created_project_id}")
                scope_lock.apply_draft_scope(persist_backend=True)
        except (ValueError, project_api.ProjectApiError) as exc:
            st.warning(str(exc))
        else:
            st.session_state["scope_bootstrap_error"] = None
            _invalidate_scope_cached_state()
            _rerun()

    scope_err = st.session_state.get("scope_projects_error")
    if scope_err:
        st.caption(f"Project list unavailable: {scope_err}")

    users_err = st.session_state.get("scope_users_error")
    if users_err:
        st.caption(f"User list unavailable: {users_err}")

    bootstrap_err = str(st.session_state.get("scope_bootstrap_error") or "").strip()
    if bootstrap_err:
        st.warning(bootstrap_err)

    _render_scope_runtime_block()


def render_project_panel() -> None:
    """Project-level controls (export/import) shown in left pane."""
    api_url = get_api_url()
    meta = _project_fetch_meta(api_url, force=False)
    name_default = str(meta.get("name") or "default")

    st.markdown(
        "<div class='ws-pane-header'>"
        f"<div class='ws-pane-header__title'>{html.escape(name_default)}</div>"
        "<div class='ws-pane-header__meta'>Export + import</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ws-pane-body">', unsafe_allow_html=True)

    if meta.get("error"):
        st.caption(str(meta.get("error")))

    name_key = "project-name"
    st.session_state.setdefault(name_key, name_default)
    name_row = st.columns([3, 1], gap="small")
    with name_row[0]:
        st.text_input("Name", key=name_key, label_visibility="collapsed")
    with name_row[1]:
        if st.button("Save", key="project-name-save", use_container_width=True):
            try:
                st.session_state["project_meta"] = project_api.put_meta(
                    st.session_state.get(name_key),
                    project_id=get_project_id(),
                    user_id=_active_reviewer_uid(),
                )
            except project_api.ProjectApiError as exc:
                st.error(str(exc))

    st.markdown("**Current applied scope**")
    _render_scope_runtime_block()

    st.divider()
    st.markdown("**Current user**")
    st.caption("Scopes all judgment saves/loads.")

    reviewers = _project_reviewers(meta)
    active_uid = _project_active_reviewer_uid(meta)
    reviewer_select_key = "project-active-reviewer"
    reviewer_add_key = "project-add-reviewer"

    if reviewer_select_key not in st.session_state:
        st.session_state[reviewer_select_key] = (
            active_uid if active_uid in reviewers else None
        )

    def _on_reviewer_change() -> None:
        chosen = _normalize_reviewer_name(st.session_state.get(reviewer_select_key))
        chosen_value = chosen or None
        if chosen_value == active_uid:
            return
        try:
            st.session_state["project_meta"] = project_api.put_meta(
                {"active_reviewer_uid": chosen_value},
                project_id=get_project_id(),
                user_id=_active_reviewer_uid(),
            )
        except project_api.ProjectApiError as exc:
            st.error(str(exc))
            st.session_state[reviewer_select_key] = active_uid
            return
        _rerun()

    select_options: list[Optional[str]] = [None] + reviewers
    st.selectbox(
        "Current user",
        select_options,
        key=reviewer_select_key,
        on_change=_on_reviewer_change,
        format_func=lambda v: "Select reviewer…" if v is None else str(v),
        label_visibility="collapsed",
    )

    add_row = st.columns([3, 1], gap="small")
    with add_row[0]:
        st.text_input(
            "Add reviewer",
            key=reviewer_add_key,
            placeholder="Add reviewer name…",
            label_visibility="collapsed",
        )

    def _on_add_reviewer_click() -> None:
        raw = _normalize_reviewer_name(st.session_state.get(reviewer_add_key))
        if not raw:
            return

        lookup = {name.casefold(): name for name in reviewers}
        chosen = lookup.get(raw.casefold()) or raw
        updated = list(reviewers)
        if chosen.casefold() not in lookup:
            updated.append(chosen)

        try:
            st.session_state["project_meta"] = project_api.put_meta(
                {"reviewers": updated, "active_reviewer_uid": chosen},
                project_id=get_project_id(),
                user_id=_active_reviewer_uid(),
            )
        except project_api.ProjectApiError as exc:
            st.session_state["project_add_reviewer_error"] = str(exc)
            return

        st.session_state.pop("project_add_reviewer_error", None)
        st.session_state[reviewer_select_key] = chosen
        st.session_state[reviewer_add_key] = ""

    with add_row[1]:
        st.button(
            "Add",
            key="project-add-reviewer-btn",
            use_container_width=True,
            on_click=_on_add_reviewer_click,
        )

    add_err = str(st.session_state.get("project_add_reviewer_error") or "").strip()
    if add_err:
        st.error(add_err)

    if not reviewers:
        st.info("Add a reviewer name to enable saving judgments.")

    st.divider()
    st.markdown("**Reset local drafts**")
    st.caption(
        "Clears local (browser-session) segmentation drafts and claim drafts. "
        "Does not delete saved judgments. Also compacts span-graph assertion history "
        "(backend) to reduce duplicate noise."
    )

    def _clear_segmentation_state(*, reviewer: Optional[str]) -> None:
        # Reviewer-scoped segmentation drafts live in session_state.
        if reviewer is None:
            st.session_state["citation_segments_by_reviewer"] = {}
        else:
            reviewer_state = _reviewer_state_suffix(reviewer)
            segs = st.session_state.get("citation_segments_by_reviewer")
            if isinstance(segs, dict):
                segs.pop(reviewer_state, None)

        # Clear any per-citation text-area caches and revision keys.
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and key.startswith("citation-segments-"):
                if reviewer is None:
                    st.session_state.pop(key, None)
                else:
                    suffix = f"::{_reviewer_state_suffix(reviewer)}"
                    if key.endswith(suffix):
                        st.session_state.pop(key, None)
            if isinstance(key, str) and key.startswith("segments-rev::"):
                if reviewer is None:
                    st.session_state.pop(key, None)
                else:
                    if f"::{_reviewer_state_suffix(reviewer)}" in key:
                        st.session_state.pop(key, None)

            # Chasing panel mode toggles (suggestions vs own) are scoped keys.
            if isinstance(key, str) and "segments-mode::" in key:
                if reviewer is None:
                    st.session_state.pop(key, None)
                else:
                    if f"::{_reviewer_state_suffix(reviewer)}" in key:
                        st.session_state.pop(key, None)

        # Legacy storage (pre-multi-user). Keep the dict but clear contents so it
        # doesn't seed future sessions via setdefault calls.
        legacy = st.session_state.get("citation_sentence_segments")
        if isinstance(legacy, dict):
            legacy.clear()

        # Claim drafts are stored in the local claim queue registry.
        for key in (
            claim_queue.CLAIM_REGISTRY_KEY,
            claim_queue.CLAIM_REGISTRY_ORDER_KEY,
            claim_queue.CLAIM_TIMELINE_KEY,
        ):
            st.session_state.pop(key, None)

        # Clear broader workspace state so the UI is truly empty.
        for key in (
            "selected_doc_id",
            "active_document",
            "citation_selected_index",
            "citation_selected_target",
            "citation_selected_sentence_id",
            "selected_callout_tuple",
            "pending_citation_selection",
            "workflow_active_citation",
            "followed_citations",
            "citation_context_key",
            "citation_context",
            "citation_context_error",
            "citation_last_context_request",
            "citation_follow_open",
            "citation_graph_key",
            "citation_graph",
            "citation_graph_error",
            "citation_last_graph_request",
            "citation_context_cache",
            "citation_parsing_inputs",
            "chase_intent",
            "chase_queue_open",
            "chase_queue_selected",
        ):
            st.session_state.pop(key, None)
        st.session_state[WORKSPACE_ACTIVE_TAB] = WORKSPACE_TAB_DOCUMENT

        # Clear evidence store session cache.
        st.session_state.pop("_evidence_store_state", None)
        st.session_state.pop("_evidence_store_instance", None)

        # Clear Surfing (live) UI state.
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and (
                key.startswith("surf_live_") or key == "surf-live"
            ):
                st.session_state.pop(key, None)

        # Clear backend-stored callout segmentation drafts so reload doesn't repopulate.
        try:
            if reviewer is None:
                target_reviewer = None
            else:
                target_reviewer = reviewer

            payload = judgment_api.list_judgments(doc_id=None, include_drafts=True)
            for j in payload.get("judgments") or []:
                if not isinstance(j, dict):
                    continue
                claim_id = str(j.get("claim_id") or "")
                if not claim_id.startswith("callout:"):
                    continue
                if str(j.get("status") or "") != "draft":
                    continue
                reviewer_uid = str(j.get("reviewer_uid") or "default")
                if target_reviewer and reviewer_uid != str(target_reviewer):
                    continue
                # Overwrite with empty draft.
                judgment_api.put_judgment(
                    claim_id,
                    {
                        "status": "draft",
                        "verdict": None,
                        "notes": None,
                        "span_selectors": None,
                    },
                    reviewer_uid=reviewer_uid,
                )
        except Exception:
            pass

    reset_cols = st.columns([1, 1], gap="small")
    with reset_cols[0]:
        if st.button(
            "Clear my drafts",
            key="project-clear-drafts-mine",
            use_container_width=True,
            disabled=not bool(active_uid),
        ):
            _clear_segmentation_state(reviewer=active_uid)
            try:
                compact_span_graph(get_api_url(), dry_run=False, aggressive=True)
            except Exception:
                pass
            _rerun()
    with reset_cols[1]:
        if st.button(
            "Clear all drafts",
            key="project-clear-drafts-all",
            use_container_width=True,
        ):
            _clear_segmentation_state(reviewer=None)
            try:
                compact_span_graph(get_api_url(), dry_run=False, aggressive=True)
            except Exception:
                pass
            _rerun()

    with st.expander("Maintenance", expanded=False):
        st.caption("Backend maintenance helpers (safe to rerun).")
        if st.button(
            "Compact span graph now",
            key="project-compact-span-graph",
            use_container_width=True,
        ):
            try:
                result = compact_span_graph(
                    get_api_url(),
                    dry_run=False,
                    aggressive=True,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.json(result)

        st.divider()
        st.markdown("**Danger zone**")
        st.caption(
            "Wipes Postgres spine tables, MinIO objects, and local data directories."
        )
        wipe_confirm = st.text_input(
            "Type WIPE to confirm",
            key="project-wipe-confirm",
            placeholder="WIPE",
        )
        if st.button(
            "Wipe everything",
            key="project-wipe-everything",
            use_container_width=True,
            disabled=str(wipe_confirm or "").strip() != "WIPE",
        ):
            try:
                result = project_api.wipe_everything(confirm=wipe_confirm)
            except project_api.ProjectApiError as exc:
                st.error(str(exc))
            else:
                st.json(result)
                # Reset local UI state.
                for key in list(st.session_state.keys()):
                    st.session_state.pop(key, None)
                _rerun()

    export_cols = st.columns([1, 1], gap="small")
    with export_cols[0]:
        if st.button(
            "Export project",
            key="project-export-prepare",
            use_container_width=True,
        ):
            try:
                st.session_state["project_export_blob"] = project_api.export_zip(
                    project_id=get_project_id(),
                    user_id=_active_reviewer_uid(),
                )
            except project_api.ProjectApiError as exc:
                st.error(str(exc))
    with export_cols[1]:
        blob = st.session_state.get("project_export_blob")
        filename = (
            f"{(st.session_state.get(name_key) or 'project').strip() or 'project'}.zip"
        )
        st.download_button(
            "Download",
            data=blob or b"",
            file_name=filename,
            mime="application/zip",
            use_container_width=True,
            disabled=not bool(blob),
        )

    st.divider()
    st.markdown("**Import project**")
    st.caption(
        "Import overwrites local data stores. A backup zip is created automatically."
    )
    uploaded = st.file_uploader(
        "Import .zip",
        type=["zip"],
        key="project-import-uploader",
        label_visibility="collapsed",
    )
    st.toggle(
        "Confirm overwrite local data",
        key="project_import_confirm",
        help=(
            "This replaces ingestion/attachments/graph/judgments/etc. "
            "Restart backend+UI after import."
        ),
    )
    if st.button(
        "Import",
        key="project-import-run",
        use_container_width=True,
        disabled=not bool(uploaded)
        or not bool(st.session_state.get("project_import_confirm")),
    ):
        try:
            data = uploaded.getvalue() if uploaded else b""
            result = project_api.import_zip(
                data,
                overwrite=True,
                project_id=get_project_id(),
                user_id=_active_reviewer_uid(),
            )
        except project_api.ProjectApiError as exc:
            st.error(str(exc))
        else:
            backup_path = (result or {}).get("backup_zip")
            st.success("Project imported. Restart backend + Streamlit.")
            if backup_path:
                st.caption(f"Backup written: {backup_path}")

    st.markdown("</div>", unsafe_allow_html=True)


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
            _rerun()
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
                _rerun()
        else:
            st.info(
                "No claims available yet. Generate claims before assigning attachments."
            )
    if status == "error":
        if st.button("Retry parse", key=f"queue-retry-{item['id']}"):
            attachment_queue.retry_attachment(item["id"])
            _rerun()
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
    inject_evidence_review_styles()
    inject_judgment_styles()
    st.markdown("## Ranked evidence preview")
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
    st.markdown("**Focus claim**")
    selected_claim = st.selectbox(
        "Focus claim",
        claim_ids,
        index=claim_ids.index(active_claim) if active_claim in claim_ids else 0,
        format_func=lambda cid: label_map.get(cid, cid),
        key="evidence-claim-select",
        label_visibility="collapsed",
    )
    if selected_claim != active_claim:
        claim_queue.set_active_claim(selected_claim)
        store = evidence_store.EvidenceStore()
    store.set_active_claim(selected_claim)
    claim_record = claim_queue.get_claim_record(selected_claim) or {}
    claim_text_override = (claim_record.get("claim") or "").strip()
    initial_claim_text = claim_text_override or None
    existing_lock = store.ensure_claim_state(selected_claim).get("lock_state") or {}
    existing_status = str(existing_lock.get("status") or "").strip().lower()
    polling = existing_status in {"queued", "running"} or bool(
        existing_lock.get("locked")
    )
    reviewer_uid = str(_active_reviewer_uid() or "").strip()
    if not reviewer_uid:
        st.info("Apply scope (user + project) to load evidence activity.")
        return
    state = store.sync_for_claim(
        selected_claim,
        claim_text=initial_claim_text,
        reviewer_uid=reviewer_uid,
        force=bool(polling),
    )
    metadata_claim_text = (state.get("metadata") or {}).get("claim_text") or ""
    active_claim_text = (metadata_claim_text or claim_text_override).strip()
    active_claim_text_payload = active_claim_text or None
    rerun_state = state.get("rerun", {})
    summary = build_progress_summary(state.get("candidates") or [])

    lock_state = state.get("lock_state") or {}
    lock_status = str(lock_state.get("status") or "").strip().lower()
    selection_locked = bool(lock_state.get("locked"))
    if lock_status in {"queued", "running"}:
        st.caption(
            "Evidence run in progress. Use 'Refresh list' to update. "
            "Selection is disabled until complete."
        )

    active_reviewer_uid = reviewer_uid
    j_store = judgment_store.JudgmentStore()
    if active_reviewer_uid:
        j_claim_state = j_store.sync_judgment(
            selected_claim, reviewer_uid=active_reviewer_uid
        )
    else:
        j_claim_state = {"judgment": None, "error": None}
    j_payload = j_claim_state.get("judgment") or {}
    layout_main = st.container()

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
        _ = EvidenceCardRenderer(ui=layout_main, callbacks=callbacks)

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

            claim_id_row = st.columns([3, 1], gap="small")
            with claim_id_row[0]:
                st.caption(f"Claim id: `{selected_claim}`")
            with claim_id_row[1]:
                clipboard.render_copy_to_clipboard(
                    "Copy id",
                    str(selected_claim),
                    key=f"copy-claim-id::{selected_claim}",
                    toast="Claim id copied.",
                    help_text="Copy claim id for API/debug.",
                )

            with st.expander("Span graph debug", expanded=False):
                claim_rec = claim_queue.get_claim_record(selected_claim) or {}
                tgt = normalize_target_id(
                    claim_rec.get("reference_id")
                    or (claim_rec.get("reference_hint") or {}).get("reference_id")
                )
                st.code(str(selected_claim), language="text")
                st.caption(
                    "This uses the new span-first endpoints (span-context/status)."
                )

                show_history_key = f"span-bundle-history::{selected_claim}"
                st.toggle(
                    "Include assertion history in bundle",
                    key=show_history_key,
                    help="Adds history_n_total per claimspan (slower).",
                )

                cache_key = f"span-bundle::{selected_claim}"
                if st.button(
                    "Fetch span bundle",
                    key=f"fetch-span-bundle::{selected_claim}",
                ):
                    try:
                        ctx = get_claim_span_context(
                            get_api_url(),
                            str(selected_claim),
                            target_id=tgt,
                        )
                        claim_status = get_claim_status(
                            get_api_url(),
                            str(selected_claim),
                            target_id=tgt,
                        )
                    except RuntimeError as exc:
                        st.error(str(exc))
                        st.session_state.pop(cache_key, None)
                    else:
                        span_id = (claim_status or {}).get("span_id")
                        reviewer_uid = (claim_status or {}).get("reviewer_uid")
                        if not span_id or not reviewer_uid:
                            st.warning("Missing span_id/reviewer_uid in claim status.")
                            st.session_state.pop(cache_key, None)
                        else:
                            try:
                                bundle = get_span_bundle(
                                    get_api_url(),
                                    str(span_id),
                                    reviewer_uid=str(reviewer_uid),
                                    include_history=bool(
                                        st.session_state.get(show_history_key)
                                    ),
                                )
                            except RuntimeError as exc:
                                st.error(str(exc))
                                st.session_state.pop(cache_key, None)
                            else:
                                st.session_state[cache_key] = {
                                    "context": ctx,
                                    "claim_status": claim_status,
                                    "span_bundle": bundle,
                                }

                cached = st.session_state.get(cache_key)
                if isinstance(cached, dict) and cached.get("span_bundle"):
                    bundle = cached.get("span_bundle") or {}
                    ctx = cached.get("context") or {}
                    claim_status = cached.get("claim_status") or {}
                    _span_status = (bundle.get("span_status") or {}).get("status")
                    st.caption(f"Span status: {_span_status}")
                    st.json({"context": ctx, "claim_status": claim_status})

                    cites = bundle.get("cites") or []
                    reviewer_uid = str(bundle.get("reviewer_uid") or "default")
                    span_id = (bundle.get("span") or {}).get("span_id")
                    if cites and span_id:
                        st.markdown("**Citation roles**")
                        role_options = [
                            "evidentiary",
                            "background",
                            "reputational",
                            "unknown",
                        ]
                        for entry in cites:
                            cited_work_id = entry.get("cited_work_id")
                            current = entry.get("role") or "unknown"
                            if not cited_work_id:
                                continue
                            key = (
                                f"cite-role::{span_id}::{cited_work_id}"
                                f"::{reviewer_uid}"
                            )
                            idx = (
                                role_options.index(current)
                                if current in role_options
                                else 3
                            )
                            st.selectbox(
                                str(cited_work_id),
                                role_options,
                                index=idx,
                                key=key,
                            )

                        if st.button(
                            "Save roles",
                            key=f"save-cite-roles::{span_id}::{reviewer_uid}",
                        ):
                            saved = 0
                            for entry in cites:
                                cited_work_id = entry.get("cited_work_id")
                                if not cited_work_id:
                                    continue
                                key = (
                                    f"cite-role::{span_id}::{cited_work_id}"
                                    f"::{reviewer_uid}"
                                )
                                role = st.session_state.get(key)
                                if role not in role_options:
                                    continue
                                try:
                                    set_span_cite_role(
                                        get_api_url(),
                                        span_id=str(span_id),
                                        cited_work_id=str(cited_work_id),
                                        reviewer_uid=reviewer_uid,
                                        role=str(role),
                                        project_id=get_project_id(),
                                    )
                                except RuntimeError as exc:
                                    st.error(str(exc))
                                    return
                                saved += 1
                            st.success(f"Saved {saved} role(s).")
                            # Refresh bundle.
                            try:
                                refreshed = get_span_bundle(
                                    get_api_url(),
                                    str(span_id),
                                    reviewer_uid=str(reviewer_uid),
                                    include_history=bool(
                                        st.session_state.get(show_history_key)
                                    ),
                                )
                            except RuntimeError:
                                return
                            st.session_state[cache_key]["span_bundle"] = refreshed

                    with st.expander("Raw bundle JSON", expanded=False):
                        st.json(bundle)

        def _render_judgment_controls() -> None:
            judgment = j_payload if isinstance(j_payload, dict) else {}
            reviewer_state = _reviewer_state_suffix(active_reviewer_uid)
            flash_key = f"judgment-flash::{selected_claim}::{reviewer_state}"
            status_default = (judgment.get("status") or "draft").strip().lower()
            if status_default not in {"draft", "final"}:
                status_default = "draft"
            verdict_default = judgment.get("verdict")
            if verdict_default not in {"support", "contradict", "uncertain"}:
                verdict_default = None
            notes_default = (
                judgment.get("notes") if isinstance(judgment.get("notes"), dict) else {}
            )

            status_key = f"judgment-status::{selected_claim}::{reviewer_state}"
            verdict_key = f"judgment-verdict::{selected_claim}::{reviewer_state}"
            notes_open_key = f"judgment-notes-open::{selected_claim}::{reviewer_state}"
            rationale_key = (
                f"judgment-notes-rationale::{selected_claim}::{reviewer_state}"
            )
            caveats_key = f"judgment-notes-caveats::{selected_claim}::{reviewer_state}"
            followups_key = (
                f"judgment-notes-followups::{selected_claim}::{reviewer_state}"
            )

            # Avoid Streamlit "default value + Session State" warnings by letting
            # widgets own their keys; only pass an explicit index when the key
            # does not exist yet.
            st.session_state.setdefault(notes_open_key, False)
            st.session_state.setdefault(
                rationale_key, (notes_default or {}).get("rationale") or ""
            )
            st.session_state.setdefault(
                caveats_key, (notes_default or {}).get("caveats") or ""
            )
            st.session_state.setdefault(
                followups_key, (notes_default or {}).get("followups") or ""
            )

            st.markdown('<div class="judgment-card">', unsafe_allow_html=True)

            if j_claim_state.get("error"):
                st.error(f"Judgment load failed: {j_claim_state['error']}")
            elif st.session_state.get(flash_key):
                st.success("Judgment saved.")
                st.session_state.pop(flash_key, None)

            updated_at = (judgment.get("updated_at") or "").strip()
            if updated_at:
                st.caption(f"Last saved: {updated_at}")

            row = st.columns([2, 4, 2], gap="small")
            with row[0]:
                status_options = ["draft", "final"]
                status_kwargs = {
                    "format_func": lambda v: "Draft" if v == "draft" else "Final",
                    "key": status_key,
                }
                if status_key in st.session_state:
                    status = st.selectbox("Status", status_options, **status_kwargs)
                else:
                    status = st.selectbox(
                        "Status",
                        status_options,
                        index=0 if status_default != "final" else 1,
                        **status_kwargs,
                    )
            with row[1]:
                verdict_options = [None, "support", "contradict", "uncertain"]
                verdict_kwargs = {
                    "horizontal": True,
                    "format_func": lambda v: {
                        None: "No verdict",
                        "support": "Support",
                        "contradict": "Contradict",
                        "uncertain": "Uncertain",
                    }.get(v, "No verdict"),
                    "key": verdict_key,
                }
                if verdict_key in st.session_state:
                    verdict = st.radio("Verdict", verdict_options, **verdict_kwargs)
                else:
                    default_idx = 0
                    if verdict_default in verdict_options:
                        default_idx = verdict_options.index(verdict_default)
                    verdict = st.radio(
                        "Verdict",
                        verdict_options,
                        index=default_idx,
                        **verdict_kwargs,
                    )
            with row[2]:
                reviewer_missing = not bool(active_reviewer_uid)
                must_have_verdict = status == "final"
                disabled = bool(
                    reviewer_missing or (must_have_verdict and verdict is None)
                )
                if st.button(
                    "Save judgment",
                    key=f"judgment-save::{selected_claim}",
                    type="primary",
                    use_container_width=True,
                    disabled=disabled,
                ):
                    if reviewer_missing:
                        st.warning(
                            (
                                "Set Current user in the Project panel before "
                                "saving judgments."
                            )
                        )
                        return
                    notes = {
                        "rationale": (st.session_state.get(rationale_key) or "").strip()
                        or None,
                        "caveats": (st.session_state.get(caveats_key) or "").strip()
                        or None,
                        "followups": (st.session_state.get(followups_key) or "").strip()
                        or None,
                    }
                    notes_payload = (
                        None
                        if not any(notes.values())
                        else {k: v for k, v in notes.items()}
                    )

                    record = claim_queue.get_claim_record(selected_claim) or {}
                    claim_text_snapshot = (
                        record.get("claim") or ""
                    ).strip() or active_claim_text
                    doi_snapshot = record.get("doi") or (
                        record.get("reference_hint") or {}
                    ).get("doi")
                    callout_snapshot = record.get("callout") or (
                        record.get("reference_hint") or {}
                    ).get("callout")

                    selected_tuple = (
                        st.session_state.get("selected_callout_tuple") or {}
                    )
                    tuple_doc_id = selected_tuple.get("doc_id")
                    tuple_cite = selected_tuple.get("citation_index")
                    tuple_target = normalize_target_id(selected_tuple.get("target_id"))
                    tuple_sentence = selected_tuple.get("sentence_id")
                    record_doc_id = record.get("doc_id")

                    provenance = {
                        "doc_id": record_doc_id or tuple_doc_id,
                        "callout": callout_snapshot,
                        "reference_id": record.get("reference_id"),
                        "doi": doi_snapshot,
                        "author": record.get("author"),
                        "year": record.get("year"),
                        "claim_text": claim_text_snapshot,
                        "citation_index": None,
                        "target_id": None,
                        "sentence_id": None,
                    }
                    if (
                        record_doc_id
                        and tuple_doc_id
                        and str(record_doc_id) == str(tuple_doc_id)
                    ):
                        provenance["citation_index"] = tuple_cite
                        provenance["target_id"] = tuple_target
                        provenance["sentence_id"] = tuple_sentence

                    provenance = _maybe_attach_citation_anchor(provenance)

                    stored = j_store.save_judgment(
                        selected_claim,
                        reviewer_uid=active_reviewer_uid,
                        status=str(status),
                        verdict=None if verdict is None else str(verdict),
                        notes=notes_payload,
                        provenance=provenance,
                    )
                    if stored is not None:
                        if str(status) == "final":
                            verdict_value = None if verdict is None else str(verdict)
                            rollup = {
                                "support": "supports",
                                "contradict": "contradicts",
                                "uncertain": "inconsistent",
                                None: "silent",
                            }.get(verdict_value, "silent")
                            try:
                                workflow_api.finalize_assessment(
                                    get_api_url(),
                                    claim_id=str(selected_claim),
                                    reviewer_uid=str(active_reviewer_uid),
                                    citing_doc_id=str(provenance.get("doc_id") or ""),
                                    judgment_snapshot=dict(stored or {}),
                                    rollup_label=rollup,
                                    by_target=None,
                                    assessed_at=datetime.now(timezone.utc)
                                    .isoformat()
                                    .replace("+00:00", "Z"),
                                )
                            except Exception as exc:
                                st.warning(f"Assessment mirror failed: {exc}")
                        st.session_state[flash_key] = True
                        st.session_state[notes_open_key] = False
                        _rerun()

                if disabled:
                    if reviewer_missing:
                        st.caption("Pick a Current user to enable saving.")
                    elif must_have_verdict and verdict is None:
                        st.caption("Final judgments require a verdict.")

            notes_open = bool(st.session_state.get(notes_open_key))
            toggle_label = "Hide notes" if notes_open else "Edit notes"
            if st.button(
                toggle_label,
                key=f"judgment-notes-toggle::{selected_claim}",
                type="secondary",
            ):
                st.session_state[notes_open_key] = not notes_open
                _rerun()

            note_preview_bits = [
                (st.session_state.get(rationale_key) or "").strip(),
                (st.session_state.get(caveats_key) or "").strip(),
                (st.session_state.get(followups_key) or "").strip(),
            ]
            preview = next((val for val in note_preview_bits if val), "")
            if (not notes_open) and preview:
                snippet = (
                    preview if len(preview) <= 110 else preview[:109].rstrip() + "..."
                )
                preview_html = (
                    "<div class='judgment-preview'>Notes: "
                    f"{html.escape(snippet)}"
                    "</div>"
                )
                st.markdown(preview_html, unsafe_allow_html=True)

            with st.expander("Notes (optional)", expanded=notes_open):
                st.text_area(
                    "Rationale",
                    key=rationale_key,
                    height=80,
                    placeholder=(
                        "Why does this claim look " "supported/contradicted/uncertain?"
                    ),
                )
                st.text_area(
                    "Caveats",
                    key=caveats_key,
                    height=80,
                    placeholder=(
                        "Anything unclear, conditional, " "or potentially wrong?"
                    ),
                )
                st.text_area(
                    "Follow-ups",
                    key=followups_key,
                    height=80,
                    placeholder="What should be checked next?",
                )

            st.markdown("**Other reviewers**")
            if not active_reviewer_uid:
                st.caption("Pick a Current user to compare judgments.")
            else:
                try:
                    payload = judgment_api.get_all_judgments(selected_claim)
                except judgment_api.JudgmentApiError as exc:
                    st.caption(f"Other reviewers unavailable: {exc}")
                else:
                    items = (
                        payload.get("judgments") if isinstance(payload, dict) else None
                    )
                    if not isinstance(items, list):
                        items = []

                    others: list[dict] = []
                    for entry in items:
                        if not isinstance(entry, dict):
                            continue
                        reviewer = _normalize_reviewer_name(entry.get("reviewer_uid"))
                        if not reviewer or reviewer == active_reviewer_uid:
                            continue
                        others.append(entry)

                    if not others:
                        st.caption("No other reviewer judgments for this claim yet.")
                    else:
                        for entry in sorted(
                            others,
                            key=lambda item: _normalize_reviewer_name(
                                item.get("reviewer_uid")
                            ).casefold(),
                        ):
                            reviewer = _normalize_reviewer_name(
                                entry.get("reviewer_uid")
                            )
                            status = str(entry.get("status") or "draft").strip().lower()
                            verdict = entry.get("verdict")
                            verdict_text = (
                                str(verdict).strip()
                                if verdict in {"support", "contradict", "uncertain"}
                                else "No verdict"
                            )
                            notes = (
                                entry.get("notes")
                                if isinstance(entry.get("notes"), dict)
                                else {}
                            )
                            note_bits = [
                                str(notes.get("rationale") or "").strip(),
                                str(notes.get("caveats") or "").strip(),
                                str(notes.get("followups") or "").strip(),
                            ]
                            preview = next((val for val in note_bits if val), "")
                            preview = re.sub(r"\s+", " ", preview).strip()
                            if preview:
                                preview = (
                                    preview
                                    if len(preview) <= 95
                                    else preview[:94].rstrip() + "..."
                                )
                                st.caption(
                                    f"{reviewer}: {verdict_text} ({status}) — {preview}"
                                )
                            else:
                                st.caption(f"{reviewer}: {verdict_text} ({status})")

            st.markdown("</div>", unsafe_allow_html=True)

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
                        reviewer_uid=reviewer_uid,
                        force=True,
                    )
                    _rerun()

            if state.get("decision_banner"):
                st.warning(str(state.get("decision_banner")), icon="⚠️")
            lock_state = state.get("lock_state") or {}
            lock_status = (lock_state.get("status") or "").strip().lower()
            if lock_status in {"queued", "running"}:
                st.info("Evidence rerun in progress…", icon="🔁")
            history = state.get("history") or []
            if history:
                latest = history[0]
                attachments_state = (latest.get("metadata") or {}).get(
                    "attachments_state"
                ) or []
                if not attachments_state:
                    st.warning(
                        "No placed sources are recorded for this claim. "
                        "Assign a Source bin PDF to this specific claim and rerun.",
                        icon="⚠️",
                    )

                    claim_rec = claim_queue.get_claim_record(selected_claim) or {}
                    target_id = claim_rec.get("reference_id") or (
                        claim_rec.get("reference_hint") or {}
                    ).get("reference_id")
                    doc_id_hint = claim_rec.get("doc_id")
                    cite_idx_hint = None
                    try:
                        parts = str(selected_claim).split(":")
                        if len(parts) >= 3 and parts[0] == "cite":
                            doc_id_hint = doc_id_hint or parts[1]
                            cite_idx_hint = int(parts[2])
                    except Exception:
                        cite_idx_hint = None

                    can_auto_place = bool(
                        str(doc_id_hint or "").strip()
                        and cite_idx_hint is not None
                        and str(target_id or "").strip()
                    )
                    if can_auto_place and st.button(
                        "Auto-place cited source",
                        key=f"auto-place-source::{selected_claim}",
                    ):
                        try:
                            auto_place_claim_source(
                                get_api_url(),
                                claim_id=str(selected_claim),
                                doc_id=str(doc_id_hint),
                                citation_index=int(cite_idx_hint),
                                target_id=str(target_id),
                                project_id=get_project_id(),
                                user_id=_active_reviewer_uid(),
                            )
                        except RuntimeError as exc:
                            st.error(str(exc))
                            return
                        store.queue_rerun(
                            selected_claim,
                            claim_text=active_claim_text_payload,
                            note="auto-place",
                        )
                        st.session_state.pop("ledger_payload", None)
                        _rerun()
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
                        reviewer_uid=reviewer_uid,
                        **chip["payload"],
                    )
                    _rerun()

            clear_cols = st.columns([1, 3], gap="small")
            with clear_cols[0]:
                if st.button(
                    "Clear decisions",
                    key=f"clear-decisions-{selected_claim}",
                    disabled=bool(state.get("decision_actions_disabled")),
                ):
                    store.clear_decisions(selected_claim, reviewer_uid=reviewer_uid)
                    _rerun()
            with clear_cols[1]:
                st.caption(
                    "Clears pins + triage for this claim/reviewer (append-only event)."
                )

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
                    advanced = {
                        "profile": st.session_state.get(
                            "execution_profile", "Fast/Local"
                        )
                    }
                    if st.session_state.get("hf_remote"):
                        advanced["hf_remote"] = True
                    store.queue_rerun(
                        selected_claim,
                        claim_text=active_claim_text_payload,
                        note="manual rerun",
                        advanced_settings=advanced,
                    )
                    _rerun()
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
                        reviewer_uid=reviewer_uid,
                    )
                    _rerun()
            with btn_cols[2]:
                if st.button(
                    "Refresh list",
                    key=f"refresh-{selected_claim}",
                    disabled=state.get("is_loading"),
                ):
                    store.sync_for_claim(
                        selected_claim,
                        claim_text=active_claim_text_payload,
                        reviewer_uid=reviewer_uid,
                        force=True,
                    )
                    _rerun()
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
                        if isinstance(payload, dict):
                            payload.setdefault(
                                "profile",
                                st.session_state.get("execution_profile", "Fast/Local"),
                            )
                            if st.session_state.get("hf_remote"):
                                payload.setdefault("hf_remote", True)
                        store.queue_rerun(
                            selected_claim,
                            claim_text=active_claim_text_payload,
                            note=note or None,
                            advanced_settings=payload,
                        )
                        _rerun()

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
            pinned = state.get("pinned") or []
            candidates = state.get("candidates") or []
            if not pinned and not candidates:
                st.info(
                    "Evidence will appear once attachments finish matching this claim."
                )
                return

            def _bucket(label: str | None) -> str:
                value = (label or "").strip().lower()
                if value in {"entail", "entails", "entailment"}:
                    return "entail"
                if value in {
                    "contradict",
                    "contradicts",
                    "contradiction",
                    "refute",
                    "refutes",
                }:
                    return "contradict"
                return "neutral"

            def _confidence(candidate: Dict[str, Any]) -> Optional[float]:
                raw = candidate.get("confidence")
                scores = candidate.get("scores") or {}
                if raw is None:
                    raw = scores.get("nli")
                if raw is None:
                    raw = scores.get("combined")
                try:
                    return float(raw) if raw is not None else None
                except (TypeError, ValueError):
                    return None

            display = state.setdefault("display", {})
            display.setdefault("top_entail", 3)
            display.setdefault("top_contradict", 3)
            display.setdefault("min_contradict", 0.6)
            display.setdefault("show_neutral", False)
            display.setdefault("troll_enabled", True)
            display.setdefault("troll_probability", 0.25)
            display.setdefault("troll_band", 0.1)

            with st.expander("Display options", expanded=False):
                st.caption(
                    "Icons: ✓ supports, – neutral/unclear, ✗ contradicts. "
                    "Use Open PDF to jump to the cited span."
                )
                cols = st.columns([1, 1, 1])
                with cols[0]:
                    display["top_entail"] = int(
                        st.number_input(
                            "Top entail",
                            min_value=0,
                            max_value=25,
                            value=int(display.get("top_entail") or 3),
                            step=1,
                            key=f"display-top-entail-{selected_claim}",
                        )
                    )
                    display["show_neutral"] = bool(
                        st.checkbox(
                            "Show neutrals",
                            value=bool(display.get("show_neutral", False)),
                            key=f"display-show-neutral-{selected_claim}",
                        )
                    )
                with cols[1]:
                    display["top_contradict"] = int(
                        st.number_input(
                            "Top contradict",
                            min_value=0,
                            max_value=25,
                            value=int(display.get("top_contradict") or 3),
                            step=1,
                            key=f"display-top-contradict-{selected_claim}",
                        )
                    )
                    display["min_contradict"] = float(
                        st.number_input(
                            "Min contradict confidence",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(display.get("min_contradict") or 0.6),
                            step=0.05,
                            key=f"display-min-contradict-{selected_claim}",
                        )
                    )
                with cols[2]:
                    display["troll_enabled"] = bool(
                        st.checkbox(
                            "Enable calibration item",
                            value=bool(display.get("troll_enabled", True)),
                            key=f"display-troll-enabled-{selected_claim}",
                        )
                    )
                    display["troll_probability"] = float(
                        st.number_input(
                            "Calibration probability",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(display.get("troll_probability") or 0.25),
                            step=0.05,
                            key=f"display-troll-prob-{selected_claim}",
                        )
                    )
                    display["troll_band"] = float(
                        st.number_input(
                            "Calibration band (+/-)",
                            min_value=0.0,
                            max_value=0.5,
                            value=float(display.get("troll_band") or 0.1),
                            step=0.01,
                            key=f"display-troll-band-{selected_claim}",
                        )
                    )

            pinned_candidates = list(pinned)
            remaining = [
                c
                for c in candidates
                if not bool((c.get("decision_state") or {}).get("pinned"))
            ]

            entail_candidates = [
                c for c in remaining if _bucket(c.get("label")) == "entail"
            ]
            contradict_candidates = [
                c
                for c in remaining
                if _bucket(c.get("label")) == "contradict"
                and (
                    _confidence(c) is not None
                    and _confidence(c) >= float(display.get("min_contradict") or 0.6)
                )
            ]
            neutral_candidates = [
                c for c in remaining if _bucket(c.get("label")) == "neutral"
            ]

            top_entail = entail_candidates[: int(display.get("top_entail") or 3)]
            top_contradict = contradict_candidates[
                : int(display.get("top_contradict") or 3)
            ]

            # Session-only review labels used for calibration + disagreement capture.
            review_labels = state.setdefault("review_labels", {})

            def _render_candidate_row(candidate: Dict[str, Any], *, label: str) -> None:
                candidate_id = candidate.get("id") or ""

                prior = review_labels.get(candidate_id)

                def _toggle(label_value: str) -> None:
                    if not candidate_id:
                        return
                    if review_labels.get(candidate_id) == label_value:
                        review_labels.pop(candidate_id, None)
                        return
                    review_labels[candidate_id] = label_value

                meta = candidate.get("metadata") or {}
                not_in_current_run = bool(
                    isinstance(meta, dict) and meta.get("not_in_current_run")
                )
                decision_state = candidate.get("decision_state") or {}
                pinned_flag = bool(decision_state.get("pinned"))
                triage_flag = str(decision_state.get("triage") or "none")
                actions_disabled = bool(state.get("decision_actions_disabled"))

                decision_cols = st.columns([1, 1, 1, 3], gap="small")
                with decision_cols[0]:
                    pin_label = "Unpin" if pinned_flag else "Pin"
                    if st.button(
                        pin_label,
                        key=f"decision-pin-{selected_claim}-{candidate_id}",
                        disabled=actions_disabled,
                        use_container_width=True,
                    ):
                        store.toggle_pin_target(
                            selected_claim,
                            candidate,
                            reviewer_uid=reviewer_uid,
                        )
                        _rerun()
                with decision_cols[1]:
                    accept_label = "Unaccept" if triage_flag == "accepted" else "Accept"
                    if st.button(
                        accept_label,
                        key=f"decision-accept-{selected_claim}-{candidate_id}",
                        disabled=actions_disabled,
                        use_container_width=True,
                    ):
                        store.accept_candidate_target(
                            selected_claim,
                            candidate,
                            reviewer_uid=reviewer_uid,
                        )
                        _rerun()
                with decision_cols[2]:
                    reject_label = "Unreject" if triage_flag == "rejected" else "Reject"
                    if st.button(
                        reject_label,
                        key=f"decision-reject-{selected_claim}-{candidate_id}",
                        disabled=actions_disabled,
                        use_container_width=True,
                    ):
                        store.reject_candidate_target(
                            selected_claim,
                            candidate,
                            reviewer_uid=reviewer_uid,
                        )
                        _rerun()
                with decision_cols[3]:
                    suffix = " (not in current run)" if not_in_current_run else ""
                    if pinned_flag or triage_flag != "none" or suffix:
                        decision_line = (
                            f"Decision: pinned={pinned_flag}, triage={triage_flag}"
                            f"{suffix}"
                        )
                        st.caption(decision_line)

                conf = _confidence(candidate)
                conf_text = f"{conf:.2f}" if conf is not None else "—"
                st.caption(f"[{label}] confidence {conf_text}")
                st.write((candidate.get("text") or "").strip() or "(empty)")

                action_cols = st.columns([1, 1, 1, 2], gap="small")
                with action_cols[0]:
                    if st.button(
                        "Support",
                        key=f"review-entail-{selected_claim}-{candidate_id}",
                        type="primary" if prior == "entail" else "secondary",
                        use_container_width=True,
                        disabled=False,
                    ):
                        _toggle("entail")
                        _rerun()
                with action_cols[1]:
                    if st.button(
                        "Neutral",
                        key=f"review-neutral-{selected_claim}-{candidate_id}",
                        type="primary" if prior == "neutral" else "secondary",
                        use_container_width=True,
                        disabled=False,
                    ):
                        _toggle("neutral")
                        _rerun()
                with action_cols[2]:
                    if st.button(
                        "Contradict",
                        key=f"review-contrad-{selected_claim}-{candidate_id}",
                        type="primary" if prior == "contradict" else "secondary",
                        use_container_width=True,
                        disabled=False,
                    ):
                        _toggle("contradict")
                        _rerun()
                with action_cols[3]:
                    if candidate_id and st.button(
                        "Open PDF",
                        key=f"view-pdf-{selected_claim}-{candidate_id}",
                        use_container_width=True,
                    ):
                        snippet = _cmdf_snippet(
                            str(candidate.get("text") or ""), words=4
                        )
                        copied = clipboard.copy_text(snippet) if snippet else False
                        if snippet:
                            if copied:
                                st.caption(f"Cmd-F snippet copied: {snippet}")
                            else:
                                st.info(f"Copy this Cmd-F snippet: {snippet}")
                        meta = candidate.get("metadata") or {}
                        attachment_id = meta.get("attachment_id") or candidate.get(
                            "attachment_id"
                        )
                        local_path = _local_path_for_attachment(
                            str(attachment_id or "")
                        )
                        if local_path:
                            err = clipboard.open_file(local_path)
                            if err:
                                st.info(f"Could not open local PDF: {err}")
                        else:
                            st.info(
                                "No local PDF path found for this evidence. "
                                "Upload sources on this machine to enable opening."
                            )
                        _rerun()

            # Optional calibration ("troll") item:
            # high-score but low-confidence boundary.
            troll_candidate: Optional[Dict[str, Any]] = None
            if display.get("troll_enabled"):
                run_id = ((state.get("run") or {}).get("run_id") or "").strip()
                troll_state = state.setdefault("troll_state", {})
                current = troll_state.get("run_id")
                if current != run_id:
                    troll_state.clear()
                    troll_state["run_id"] = run_id
                    troll_state["candidate_id"] = None
                    troll_state["enabled"] = False
                    troll_state["seed"] = None
                    try:
                        import hashlib
                        import random

                        seed = int(
                            hashlib.sha256(
                                f"{selected_claim}|{run_id}|troll".encode("utf-8")
                            ).hexdigest()[:8],
                            16,
                        )
                        troll_state["seed"] = seed
                        rng = random.Random(seed)
                        troll_state["enabled"] = rng.random() < float(
                            display.get("troll_probability") or 0.25
                        )
                    except Exception:
                        troll_state["enabled"] = False
                if troll_state.get("enabled") and not troll_state.get("candidate_id"):
                    band = float(display.get("troll_band") or 0.1)
                    already = {
                        c.get("id")
                        for c in (pinned_candidates + top_entail + top_contradict)
                    }
                    eligible = []
                    for cand in candidates:
                        cid = cand.get("id")
                        if not cid or cid in already:
                            continue
                        score = _confidence(cand)
                        if score is None:
                            continue
                        if abs(score - 0.5) <= band:
                            eligible.append(cand)
                    if eligible:
                        try:
                            import random

                            rng = random.Random(troll_state.get("seed") or 0)
                            pick = rng.choice(eligible)
                            troll_state["candidate_id"] = pick.get("id")
                        except Exception:
                            troll_state["candidate_id"] = eligible[0].get("id")
                if troll_state.get("candidate_id"):
                    cid = troll_state.get("candidate_id")
                    troll_candidate = next(
                        (c for c in candidates if c.get("id") == cid), None
                    )

            if pinned_candidates:
                st.markdown("#### Pinned")
                for cand in pinned_candidates:
                    _render_candidate_row(cand, label=_bucket(cand.get("label")))

            decisions = state.get("decisions")
            if isinstance(decisions, dict):
                with st.expander("Decision timeline", expanded=False):
                    events = decisions.get("events") or []
                    if not events:
                        st.caption("No decision events yet.")
                    for ev in events[:20]:
                        if not isinstance(ev, dict):
                            continue
                        action = str(ev.get("action") or "").strip()
                        created_at = str(ev.get("created_at") or "").strip()
                        payload = (
                            ev.get("payload")
                            if isinstance(ev.get("payload"), dict)
                            else {}
                        )
                        snippet = str(payload.get("snippet") or "").strip()
                        line = f"{created_at} — {action}".strip(" -")
                        if snippet:
                            line = f"{line}: {snippet}".strip()
                        st.markdown(f"- {line}")

            if not pinned_candidates and not top_entail and not top_contradict:
                st.markdown("#### Top scored")
                st.caption(
                    "Model labels not available (or filtered out). "
                    "Showing highest-scoring candidates."
                )
                fallback = list(remaining)[:6]
                for cand in fallback:
                    _render_candidate_row(cand, label=_bucket(cand.get("label")))

            if top_entail:
                st.markdown("#### Top entail")
                for cand in top_entail:
                    _render_candidate_row(cand, label="entail")

            if top_contradict:
                st.markdown("#### Top contradict")
                for cand in top_contradict:
                    _render_candidate_row(cand, label="contradict")

            if troll_candidate:
                st.markdown("#### Calibration candidate")
                st.caption("Hard-to-classify item for disagreement/correction data.")
                _render_candidate_row(
                    troll_candidate, label=_bucket(troll_candidate.get("label"))
                )

            if display.get("show_neutral") and neutral_candidates:
                with st.expander(
                    f"Neutral candidates ({len(neutral_candidates)})",
                    expanded=False,
                ):
                    for cand in neutral_candidates:
                        _render_candidate_row(cand, label="neutral")

            st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
            st.markdown("#### Overall source assessment")

            selection_payload = store.sync_selection(selected_claim, force=False) or {}
            selection_verdict = (
                str(selection_payload.get("verdict") or "none").strip().lower()
            )
            selection_note = selection_payload.get("note") or ""

            entail_votes = sum(1 for v in review_labels.values() if v == "entail")
            contradict_votes = sum(
                1 for v in review_labels.values() if v == "contradict"
            )
            if entail_votes == 0 and contradict_votes == 0:
                computed = "silent"
            elif entail_votes > 0 and contradict_votes == 0:
                computed = "supports"
            elif contradict_votes > 0 and entail_votes == 0:
                computed = "contradicts"
            else:
                computed = "inconsistent"

            reviewer_state = _reviewer_state_suffix(active_reviewer_uid)
            overall_by_reviewer = state.setdefault("overall_by_reviewer", {})
            overall_state = overall_by_reviewer.setdefault(reviewer_state, {})
            overall_options = ["supports", "contradicts", "inconsistent", "silent"]
            selection_to_overall = {
                "support": "supports",
                "contradict": "contradicts",
                "uncertain": "inconsistent",
                "none": "silent",
            }
            mapped_overall = selection_to_overall.get(selection_verdict)
            current_overall = overall_state.get("verdict") or mapped_overall or computed
            if current_overall not in overall_options:
                current_overall = computed
            overall_verdict = st.radio(
                "Overall",
                overall_options,
                index=overall_options.index(current_overall),
                horizontal=True,
                key=f"overall-verdict-{selected_claim}::{reviewer_state}",
            )
            overall_state["verdict"] = overall_verdict
            st.caption(
                (
                    f"Default: {computed}. Votes: {entail_votes} support, "
                    f"{contradict_votes} contradict."
                )
            )

            note_default = str(overall_state.get("note") or "")
            if not note_default and selection_note:
                note_default = str(selection_note)
            note = st.text_area(
                "Optional note",
                value=note_default,
                height=80,
                key=f"overall-note-{selected_claim}::{reviewer_state}",
            )
            overall_state["note"] = note

            verdict_map = {
                "supports": "support",
                "contradicts": "contradict",
                "inconsistent": "uncertain",
                "silent": "none",
            }

            st.markdown("#### Final judgment")

            # Prefill from stored judgment payload.
            judgment = j_payload if isinstance(j_payload, dict) else {}

            notes_default = (
                judgment.get("notes") if isinstance(judgment.get("notes"), dict) else {}
            )
            rationale_key = (
                f"judgment-notes-rationale::{selected_claim}::{reviewer_state}"
            )
            caveats_key = f"judgment-notes-caveats::{selected_claim}::{reviewer_state}"
            followups_key = (
                f"judgment-notes-followups::{selected_claim}::{reviewer_state}"
            )
            st.session_state.setdefault(
                rationale_key, (notes_default or {}).get("rationale") or ""
            )
            st.session_state.setdefault(
                caveats_key, (notes_default or {}).get("caveats") or ""
            )
            st.session_state.setdefault(
                followups_key, (notes_default or {}).get("followups") or ""
            )

            validation_default = (
                judgment.get("validation")
                if isinstance(judgment.get("validation"), dict)
                else {}
            )
            advanced_key = f"judgment-advanced::{selected_claim}::{reviewer_state}"
            st.session_state.setdefault(advanced_key, False)
            advanced = bool(st.session_state.get(advanced_key))
            st.toggle(
                "Advanced validation",
                key=advanced_key,
                help=(
                    (
                        "Adds separate ratings for source validity and relevance, "
                        "each with a comment."
                    )
                ),
            )

            validation_rating_options = [
                "strongly_agree",
                "agree",
                "neutral",
                "disagree",
                "strongly_disagree",
            ]
            rating_labels = {
                "strongly_agree": "Strongly agree",
                "agree": "Agree",
                "neutral": "Neutral",
                "disagree": "Disagree",
                "strongly_disagree": "Strongly disagree",
            }
            valid_key = f"judgment-valid::{selected_claim}::{reviewer_state}"
            valid_comment_key = (
                f"judgment-valid-comment::{selected_claim}::{reviewer_state}"
            )
            rel_key = f"judgment-relevant::{selected_claim}::{reviewer_state}"
            rel_comment_key = (
                f"judgment-relevant-comment::{selected_claim}::{reviewer_state}"
            )
            st.session_state.setdefault(
                valid_key, (validation_default or {}).get("source_valid")
            )
            st.session_state.setdefault(
                valid_comment_key,
                (validation_default or {}).get("source_valid_comment") or "",
            )
            st.session_state.setdefault(
                rel_key, (validation_default or {}).get("source_relevant")
            )
            st.session_state.setdefault(
                rel_comment_key,
                (validation_default or {}).get("source_relevant_comment") or "",
            )

            if advanced:
                st.markdown("**Validity & relevance**")
                st.radio(
                    "Source is valid",
                    options=[None] + validation_rating_options,
                    format_func=lambda v: "Unrated"
                    if v is None
                    else rating_labels.get(str(v), str(v)),
                    horizontal=True,
                    key=valid_key,
                )
                st.text_area(
                    "Validity comment",
                    key=valid_comment_key,
                    height=70,
                    placeholder="Why is the source valid/invalid?",
                )
                st.radio(
                    "Source is relevant to the citing context",
                    options=[None] + validation_rating_options,
                    format_func=lambda v: "Unrated"
                    if v is None
                    else rating_labels.get(str(v), str(v)),
                    horizontal=True,
                    key=rel_key,
                )
                st.text_area(
                    "Relevance comment",
                    key=rel_comment_key,
                    height=70,
                    placeholder="Why is the source relevant/irrelevant?",
                )

            with st.expander("Notes", expanded=False):
                st.text_area(
                    "Rationale",
                    key=rationale_key,
                    height=70,
                    placeholder="Brief explanation of your overall judgment.",
                )
                st.text_area(
                    "Caveats",
                    key=caveats_key,
                    height=70,
                    placeholder="Anything unclear, conditional, or potentially wrong?",
                )
                st.text_area(
                    "Follow-ups",
                    key=followups_key,
                    height=70,
                    placeholder="What should be checked next?",
                )

            stored_verdict = verdict_map.get(overall_verdict, "none")
            judgment_verdict = None if stored_verdict == "none" else stored_verdict
            status_to_save = "final" if judgment_verdict is not None else "draft"
            disabled = bool(selection_locked)

            primary_candidate_id: Optional[str] = None
            needs_primary = stored_verdict in {"support", "contradict"}
            if needs_primary:
                desired = "entail" if stored_verdict == "support" else "contradict"
                reviewed = [
                    cid
                    for cid, lbl in (review_labels or {}).items()
                    if str(lbl) == desired
                ]
                if not reviewed:
                    fallback_candidates = (
                        top_entail if desired == "entail" else top_contradict
                    )
                    reviewed = [c.get("id") for c in fallback_candidates if c.get("id")]

                id_to_candidate = {
                    c.get("id"): c
                    for c in candidates
                    if c.get("id") and isinstance(c, dict)
                }
                reviewed = [cid for cid in reviewed if cid in id_to_candidate]
                if reviewed:

                    def _format_primary(cid: str) -> str:
                        cand = id_to_candidate.get(cid) or {}
                        label = _bucket(cand.get("label"))
                        conf = _confidence(cand)
                        conf_text = f"{conf:.2f}" if conf is not None else "—"
                        text = str(cand.get("text") or "").strip()
                        text = re.sub(r"\s+", " ", text).strip()
                        if len(text) > 80:
                            text = text[:79].rstrip() + "..."
                        return f"[{label} {conf_text}] {text}".strip()

                    primary_candidate_id = st.selectbox(
                        "Primary evidence",
                        reviewed,
                        index=0,
                        format_func=_format_primary,
                        key=f"overall-primary-{selected_claim}::{reviewer_state}",
                        help=(
                            "Pick the single primary evidence candidate used to "
                            "justify support/contradict."
                        ),
                    )
                else:
                    st.warning(
                        "Pick at least one evidence candidate (Support/Contradict) "
                        "before saving a non-silent verdict.",
                        icon="⚠️",
                    )
                    disabled = True

            if st.button(
                "Save",
                key=f"final-save-{selected_claim}",
                type="primary",
                disabled=disabled,
            ):
                if not active_reviewer_uid:
                    st.warning(
                        "Set Current user in the Project panel before saving judgments."
                    )
                    return
                store.save_selection(
                    selected_claim,
                    verdict=stored_verdict,
                    primary_candidate_id=primary_candidate_id,
                    note=str(note or "").strip() or None,
                )
                notes = {
                    "rationale": (st.session_state.get(rationale_key) or "").strip()
                    or None,
                    "caveats": (st.session_state.get(caveats_key) or "").strip()
                    or None,
                    "followups": (st.session_state.get(followups_key) or "").strip()
                    or None,
                }
                notes_payload = (
                    None
                    if not any(notes.values())
                    else {k: v for k, v in notes.items()}
                )

                validation_payload = None
                if advanced:
                    validation_payload = {
                        "source_valid": st.session_state.get(valid_key),
                        "source_valid_comment": (
                            st.session_state.get(valid_comment_key) or ""
                        ).strip()
                        or None,
                        "source_relevant": st.session_state.get(rel_key),
                        "source_relevant_comment": (
                            st.session_state.get(rel_comment_key) or ""
                        ).strip()
                        or None,
                    }
                    if not any(validation_payload.values()):
                        validation_payload = None

                record = claim_queue.get_claim_record(selected_claim) or {}
                claim_text_snapshot = (
                    record.get("claim") or ""
                ).strip() or active_claim_text
                doi_snapshot = record.get("doi") or (
                    record.get("reference_hint") or {}
                ).get("doi")
                callout_snapshot = record.get("callout") or (
                    record.get("reference_hint") or {}
                ).get("callout")
                selected_tuple = st.session_state.get("selected_callout_tuple") or {}
                tuple_doc_id = selected_tuple.get("doc_id")
                tuple_cite = selected_tuple.get("citation_index")
                tuple_target = normalize_target_id(selected_tuple.get("target_id"))
                tuple_sentence = selected_tuple.get("sentence_id")
                record_doc_id = record.get("doc_id")

                provenance = {
                    "doc_id": record_doc_id or tuple_doc_id,
                    "callout": callout_snapshot,
                    "reference_id": record.get("reference_id"),
                    "doi": doi_snapshot,
                    "author": record.get("author"),
                    "year": record.get("year"),
                    "claim_text": claim_text_snapshot,
                    "citation_index": None,
                    "target_id": None,
                    "sentence_id": None,
                }
                if (
                    record_doc_id
                    and tuple_doc_id
                    and str(record_doc_id) == str(tuple_doc_id)
                ):
                    provenance["citation_index"] = tuple_cite
                    provenance["target_id"] = tuple_target
                    provenance["sentence_id"] = tuple_sentence

                provenance = _maybe_attach_citation_anchor(provenance)

                j_store.save_judgment(
                    selected_claim,
                    reviewer_uid=active_reviewer_uid,
                    status=status_to_save,
                    verdict=judgment_verdict,
                    notes=notes_payload,
                    validation=validation_payload,
                    provenance=provenance,
                )
                _rerun()

            if disabled and selection_locked:
                st.caption("Save disabled while evidence run is locked.")

        def _render_source_review() -> None:
            candidates = state.get("candidates") or []
            if not candidates:
                st.info("None found. Request a rerun or check attachment status.")
                return

            def _candidate_confidence(candidate: Dict[str, Any]) -> Optional[float]:
                raw = candidate.get("confidence")
                if raw is None:
                    scores = candidate.get("scores") or {}
                    raw = scores.get("combined") or scores.get("confidence")
                try:
                    return float(raw) if raw is not None else None
                except (TypeError, ValueError):
                    return None

            def _format_candidate_option(candidate: Dict[str, Any]) -> str:
                cid = candidate.get("id") or "(missing id)"
                label = (candidate.get("label") or "neutral").strip().lower()
                conf = _candidate_confidence(candidate)
                snippet = (candidate.get("text") or "").strip()
                preview = re.sub(r"\s+", " ", snippet)[:120]
                conf_label = f" {conf:.2f}" if conf is not None else ""
                return f"[{label}{conf_label}] {preview} ({cid[:8]})"

            def _is_muted(candidate: Dict[str, Any]) -> bool:
                label = (candidate.get("label") or "neutral").strip().lower()
                conf = _candidate_confidence(candidate)
                if label == "neutral":
                    return True
                if conf is not None and conf < 0.55:
                    return True
                return False

            def _section_path(candidate: Dict[str, Any]) -> str:
                meta = candidate.get("metadata") or {}
                value = meta.get("section_path") or meta.get("section") or ""
                cleaned = str(value or "").strip()
                if not cleaned or cleaned.lower() in {"unknown", "none"}:
                    return "Body"
                return cleaned

            top_n = 5
            top_hits = list(candidates[:top_n])
            rest = list(candidates[top_n:])

            st.markdown("### Evidence Review")
            st.caption(
                "Source view shows TEI paragraph-bounded excerpt windows "
                "with highlights."
            )

            st.markdown("#### Top hits")
            for candidate in top_hits:
                label = _format_candidate_option(candidate)
                with st.expander(label, expanded=False):
                    excerpt = store.preview_excerpt(selected_claim, candidate)
                    if not excerpt:
                        st.info(
                            (
                                "Excerpt unavailable yet (attachment may still be "
                                "extracting)."
                            )
                        )
                        continue
                    sentences = excerpt.get("sentences") or []
                    meta = candidate.get("metadata") or {}
                    badge_bits = []
                    cand_label = (candidate.get("label") or "neutral").strip().lower()
                    tone = (
                        "entail"
                        if cand_label == "entail"
                        else "contrad"
                        if cand_label == "contradict"
                        else "neutral"
                    )
                    badge_bits.append(
                        (
                            (
                                '<span class="evidence-badge evidence-badge--{tone}">'
                                "{label}</span>"
                            ).format(
                                tone=tone,
                                label=html.escape(cand_label.title()),
                            )
                        )
                    )
                    conf = _candidate_confidence(candidate)
                    if conf is not None:
                        badge_bits.append(
                            (
                                '<span class="evidence-badge evidence-badge--rank">'
                                f"{conf:.2f}" + "</span>"
                            )
                        )
                    page = meta.get("page")
                    if page:
                        badge_bits.append(
                            (
                                '<span class="evidence-badge evidence-badge--info">'
                                "Page " + html.escape(str(page)) + "</span>"
                            )
                        )
                    meta_html = (
                        '<div class="evidence-review__candidate-meta">'
                        + "".join(badge_bits)
                        + "</div>"
                    )
                    st.markdown(
                        meta_html,
                        unsafe_allow_html=True,
                    )

                    highlight_class = "evidence-review__sentence--highlight"
                    rendered_sentences = []
                    for sentence in sentences:
                        text = html.escape(str(sentence.get("text") or ""))
                        classes = ["evidence-review__sentence"]
                        if sentence.get("is_highlight"):
                            classes.append(highlight_class)
                        rendered_sentences.append(
                            f'<div class="{" ".join(classes)}">{text}</div>'
                        )
                    excerpt_html = (
                        '<div class="evidence-review__excerpt">'
                        + "".join(rendered_sentences)
                        + "</div>"
                    )
                    st.markdown(
                        excerpt_html,
                        unsafe_allow_html=True,
                    )
                    if sentences:
                        first_sentence = sentences[0]
                        section_label = first_sentence.get("section_path")
                        if not section_label:
                            section_label = _section_path(candidate)
                        st.caption(f"Section: {section_label}")

            from collections import defaultdict

            grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for candidate in rest:
                grouped[_section_path(candidate)].append(candidate)
            if grouped:
                st.markdown("#### By section")
            for section, items in sorted(grouped.items(), key=lambda pair: pair[0]):
                st.markdown(f"**{section}**")
                for candidate in items:
                    muted = _is_muted(candidate)
                    label = _format_candidate_option(candidate)
                    with st.expander(label, expanded=False):
                        classes = ["evidence-review__candidate"]
                        if muted:
                            classes.append("evidence-review__candidate--muted")
                        highlight_class = "evidence-review__sentence--highlight"
                        excerpt = store.preview_excerpt(selected_claim, candidate)
                        if not excerpt:
                            st.info(
                                "Excerpt unavailable yet (attachment may still be "
                                "extracting)."
                            )
                            continue
                        sentences = excerpt.get("sentences") or []
                        rendered = []
                        for sentence in sentences:
                            text = html.escape(str(sentence.get("text") or ""))
                            s_classes = ["evidence-review__sentence"]
                            if sentence.get("is_highlight"):
                                s_classes.append(highlight_class)
                            rendered.append(
                                f'<div class="{" ".join(s_classes)}">{text}</div>'
                            )
                        grouped_excerpt_html = (
                            '<div class="{classes}">'.format(classes=" ".join(classes))
                            + '<div class="evidence-review__excerpt">'
                            + "".join(rendered)
                            + "</div></div>"
                        )
                        st.markdown(
                            grouped_excerpt_html,
                            unsafe_allow_html=True,
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
                _rerun()

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
                _rerun()

        _render_claim_header()
        _render_status_messages()
        _render_rerun_controls()
        _render_progress_glance()
        _render_evidence_lists()
        _render_share_panel()
        _render_pdf_notice()

    # No right sidebar: keep evidence content full-width.


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
        render_settings_controls()


def render_settings_controls() -> None:
    """Render settings shared by the legacy sidebar and Phase 8 drawer."""
    st.markdown("**Execution**")
    profile_options = ["Fast/Local", "Best/Local"]
    current_profile = st.session_state.get("execution_profile")
    st.session_state["execution_profile"] = st.selectbox(
        "Execution profile",
        profile_options,
        index=profile_options.index(current_profile)
        if current_profile in profile_options
        else 0,
        help="Controls backend retrieval/reranking for evidence reruns.",
        key="execution_profile_selectbox",
    )
    if st.session_state["execution_profile"] == "Best/Local":
        st.caption("Best/Local enables the hybrid pipeline with ColBERT reranking.")

    st.toggle(
        "Use HF Inference API (demo)",
        key="hf_remote",
        help=(
            "When enabled and HF_API_TOKEN is set, NLI can use Hugging Face Inference "
            "API. Remote failures fall back to local automatically."
        ),
    )

    st.markdown("**Pipeline**")
    mode = st.selectbox(
        "Choose pipeline",
        ["classic", "hybrid"],
        index=0 if st.session_state["pipeline_mode"] == "classic" else 1,
        help="Classic = fast, Hybrid = exhaustive",
        key="pipeline_mode_selectbox",
    )
    st.session_state["pipeline_mode"] = mode

    st.markdown("**Ingestion Defaults**")
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

    st.markdown("**API**")
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

    st.markdown("**Models**")
    with st.spinner("Fetching models..."):
        if "available_models" not in st.session_state:
            st.session_state.available_models = get_responsive_models()
    model_selector(
        "LLM model",
        "selected_model",
        st.session_state.available_models,
        allow_custom=False,
    )

    embed_choices = get_models("embed") or list_local_models()
    if settings.EMBED_MODEL not in embed_choices:
        embed_choices.insert(0, settings.EMBED_MODEL)
    model_selector("Embedding model", "embed_model", embed_choices)
    add_model("embed", st.session_state.embed_model)

    st.markdown("**Retrieval**")
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

    rerank_choices = get_models("reranker") or [settings.RERANKER_MODEL]
    model_selector("Reranker model", "reranker_model", rerank_choices)
    add_model("reranker", st.session_state.reranker_model)
    st.number_input(
        "Reranker top-K",
        min_value=1,
        key="reranker_top_k",
    )

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

    st.markdown("**Danger Zone**")
    st.caption("Selective wipe for local dev diagnostics.")
    st.checkbox("Wipe spine data (Postgres)", key="wipe_spine_data")
    st.checkbox("Wipe object store (MinIO/S3)", key="wipe_object_store")
    st.checkbox("Wipe graph/cache stores", key="wipe_graph_caches")
    st.checkbox("Wipe local retrieval indexes", key="wipe_local_indexes")
    st.text_input(
        "Type WIPE to confirm",
        key="wipe_confirm_text",
        placeholder="WIPE",
        label_visibility="collapsed",
    )
    if st.button(
        "Run selective wipe",
        key="run-selective-wipe",
        use_container_width=True,
    ):
        confirm = str(st.session_state.get("wipe_confirm_text") or "").strip()
        try:
            result = project_api.wipe_selective(
                confirm=confirm,
                spine_data=bool(st.session_state.get("wipe_spine_data", False)),
                object_store=bool(st.session_state.get("wipe_object_store", False)),
                graph_caches=bool(st.session_state.get("wipe_graph_caches", False)),
                local_indexes=bool(st.session_state.get("wipe_local_indexes", False)),
            )
            st.success(
                "Wipe complete: "
                f"spine={result.get('spine_tables_truncated')} "
                f"s3_deleted={result.get('s3_deleted_objects')} "
                f"paths={result.get('removed_paths')}"
            )
            st.session_state["project_meta"] = None
            st.session_state["ingested_docs"] = None
            st.session_state["_ingested_docs_loaded"] = False
            st.session_state["selected_doc_id"] = ""
            st.session_state["active_document"] = None
            st.session_state.pop("_followed_citations_cache", None)
        except project_api.ProjectApiError as exc:
            st.error(str(exc))


class _BytesUploadFile:
    def __init__(self, filename: str, data: bytes):
        self.name = filename or "document.pdf"
        self.type = "application/pdf"
        self._data = data
        self.size = len(data)

    def getbuffer(self):
        return self._data


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _intake_stage_icon(stage: str) -> str:
    normalized = (stage or "").strip().lower()
    if normalized in {"unknown", "awaiting-intent"}:
        return ":material/running_with_errors:"
    if normalized in {"error", "failed"}:
        return ":material/running_with_errors:"
    if normalized in {"complete", "done"}:
        return ":material/check_circle:"
    if normalized in {"near-duplicate"}:
        return ":material/running_with_errors:"
    return ":material/clock_loader_10:"


def _intake_guess_intent(*, filename: str, data: bytes) -> str:
    # Heuristic 1: if we're actively chasing within an existing citing doc,
    # new drops are more likely supporting sources.
    if (
        st.session_state.get("selected_doc_id")
        and st.session_state.get(WORKSPACE_ACTIVE_TAB) == WORKSPACE_TAB_REVIEW
    ):
        return "source"

    # Heuristic 2: try to spot obvious bibliography/citation signals in the PDF bytes.
    try:
        sample = data[:200000].decode("latin-1", errors="ignore").lower()
    except Exception:
        sample = ""
    signals = [
        "references",
        "bibliography",
        "works cited",
        "doi:",
        "et al",
    ]
    if any(sig in sample for sig in signals):
        return "citing"

    # Otherwise: ask.
    return "unknown"


def _intake_find_item(item_id: str) -> Optional[dict]:
    inbox = st.session_state.get("intake_inbox") or []
    for item in inbox:
        if isinstance(item, dict) and str(item.get("id")) == str(item_id):
            return item
    return None


def _intake_touch(
    item: dict, *, stage: Optional[str] = None, error: Optional[str] = None
):
    if stage is not None:
        item["stage"] = str(stage)
    if error is not None:
        item["error"] = str(error)
    item["last_event_at"] = _now_iso()


def _intake_route_citing(item_id: str) -> None:
    item = _intake_find_item(item_id)
    if not item:
        return
    data = (st.session_state.get("intake_blobs") or {}).get(item_id)
    if not isinstance(data, (bytes, bytearray)):
        _intake_touch(item, stage="error", error="Missing PDF bytes in session")
        return

    api_url = get_api_url()
    project_id = get_project_id()
    auto_extract = bool(st.session_state.get("auto_extract_on_upload"))

    before_ids = {
        str(doc.get("id"))
        for doc in (st.session_state.get("ingested_docs") or [])
        if isinstance(doc, dict) and doc.get("id")
    }

    _intake_touch(item, stage="uploading")
    try:
        uploaded = upload_pdf(
            api_url,
            _BytesUploadFile(str(item.get("filename") or "document.pdf"), bytes(data)),
            project_id=project_id,
            user_id=_active_reviewer_uid(),
            auto_process=auto_extract,
        )
    except RuntimeError as exc:
        _intake_touch(item, stage="error", error=str(exc))
        return

    doc_id = str((uploaded or {}).get("id") or "").strip()
    if doc_id:
        item["doc_id"] = doc_id
    item["extraction_triggered"] = bool(auto_extract)
    item["resolution_triggered"] = False
    item["near_duplicate_checked"] = False
    item["near_duplicate_matches"] = []
    item["near_duplicate_decision"] = None
    if doc_id and doc_id in before_ids:
        item["note"] = (
            "Exact duplicate bytes; backend should record a new version "
            "under the same work."
        )

    refresh_ingested_docs(show_error=False)
    _ledger_fetch(api_url, force=True)
    _intake_touch(item, stage="uploaded")


def _intake_route_source(item_id: str, *, attach_now: bool = False) -> None:
    item = _intake_find_item(item_id)
    if not item:
        return
    data = (st.session_state.get("intake_blobs") or {}).get(item_id)
    if not isinstance(data, (bytes, bytearray)):
        _intake_touch(item, stage="error", error="Missing PDF bytes in session")
        return

    attachment_queue.init_attachment_queue_state()
    doc_id = st.session_state.get("selected_doc_id") if attach_now else None
    selected_target = normalize_target_id(
        st.session_state.get("citation_selected_target")
    )
    reference_hint = (
        {"reference_id": selected_target} if (attach_now and selected_target) else {}
    )

    _intake_touch(item, stage="queued")
    try:
        created = attachment_queue.enqueue(
            [
                _BytesUploadFile(
                    str(item.get("filename") or "attachment.pdf"), bytes(data)
                )
            ],
            doc_id=str(doc_id) if doc_id else None,
            reference_hint=reference_hint,
            source="intake",
        )
    except Exception as exc:
        _intake_touch(item, stage="error", error=str(exc))
        return
    item["source_queue_item_ids"] = list(created or [])
    _intake_touch(item, stage="uploading-source")


def _intake_near_duplicate_candidates(
    extraction_data: dict, ledger_rows: list[dict]
) -> list[dict]:
    if not isinstance(extraction_data, dict):
        return []
    title = (
        extraction_data.get("title")
        or (extraction_data.get("header") or {}).get("title")
        or ""
    )
    year = (
        extraction_data.get("year")
        or (extraction_data.get("header") or {}).get("year")
        or None
    )
    if not title:
        return []

    def tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", str(text or "").lower()) if t}

    title_tokens = tokens(title)
    if len(title_tokens) < 4:
        return []

    matches: list[dict] = []
    for row in ledger_rows or []:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("anchored")):
            continue
        row_title = row.get("title") or row.get("apa") or row.get("short") or ""
        row_tokens = tokens(str(row_title))
        if not row_tokens:
            continue
        inter = len(title_tokens & row_tokens)
        union = max(1, len(title_tokens | row_tokens))
        jacc = inter / union
        score = jacc
        if year and str(year) in str(row_title):
            score += 0.08
        if score >= 0.68:
            matches.append(
                {
                    "num": row.get("num"),
                    "short": row.get("short"),
                    "title": row.get("title"),
                    "ingest_id": row.get("ingest_id"),
                    "score": round(score, 3),
                }
            )
    matches.sort(key=lambda m: float(m.get("score") or 0.0), reverse=True)
    return matches[:3]


def _intake_refresh_item_status(item: dict) -> None:
    if not isinstance(item, dict):
        return

    api_url = get_api_url()
    project_id = get_project_id()
    intent = str(item.get("intent") or "unknown")
    active_intent = str(item.get("routed_intent") or intent or "unknown")

    if active_intent == "citing":
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id:
            return
        try:
            doc = get_document(api_url, doc_id, project_id=project_id)
        except RuntimeError as exc:
            _intake_touch(item, stage="error", error=str(exc))
            return
        extraction = (doc.get("extraction") or {}) if isinstance(doc, dict) else {}
        resolution = (doc.get("resolution") or {}) if isinstance(doc, dict) else {}
        extraction_status = str(extraction.get("status") or "").strip().lower()
        extraction_error = str(extraction.get("error") or "").strip()
        resolution_status = str(resolution.get("status") or "").strip().lower()
        resolution_error = str(resolution.get("error") or "").strip()

        if extraction_status == "error":
            _intake_touch(
                item, stage="error", error=extraction_error or "Extraction failed"
            )
            return
        if resolution_status == "error":
            _intake_touch(
                item, stage="error", error=resolution_error or "Resolution failed"
            )
            return

        if resolution_status == "complete":
            _intake_touch(item, stage="done")
            return
        if resolution_status == "running":
            _intake_touch(item, stage="resolving")
            return

        if extraction_status == "running":
            _intake_touch(item, stage="extracting")
            return

        extraction_data = extraction.get("data") if isinstance(extraction, dict) else {}
        if extraction_status == "complete" and isinstance(extraction_data, dict):
            if not bool(item.get("near_duplicate_checked")):
                payload = _ledger_fetch(api_url, force=False)
                rows = payload.get("rows") or []
                candidates = _intake_near_duplicate_candidates(extraction_data, rows)
                item["near_duplicate_checked"] = True
                item["near_duplicate_matches"] = candidates
                if candidates:
                    _intake_touch(item, stage="near-duplicate")
                    return

            if bool(st.session_state.get("auto_resolve_on_upload")) and not bool(
                item.get("resolution_triggered")
            ):
                _intake_touch(item, stage="resolving")
                try:
                    trigger_resolution(
                        api_url,
                        doc_id,
                        project_id=project_id,
                        user_id=_active_reviewer_uid(),
                    )
                except RuntimeError as exc:
                    _intake_touch(item, stage="error", error=str(exc))
                    return
                item["resolution_triggered"] = True
                return

            _intake_touch(item, stage="uploaded")
            return

        _intake_touch(item, stage="uploaded")
        return

    if active_intent == "source":
        ids = item.get("source_queue_item_ids") or []
        if not ids:
            return
        attachment_queue.init_attachment_queue_state()
        # Use local queue snapshot for best-effort stage.
        snapshot = attachment_queue.get_queue_snapshot()
        by_id = snapshot.get("items") or {}
        stages = []
        errors = []
        for qid in ids:
            q = by_id.get(qid) or {}
            status = str(q.get("status") or "pending").strip().lower()
            if status == "error":
                errors.extend(q.get("errors") or [])
            stages.append(status)
        if errors:
            last = errors[0]
            msg = last.get("message") if isinstance(last, dict) else str(last)
            _intake_touch(item, stage="error", error=msg or "Source upload failed")
            return
        if any(s in {"pending", "converting", "parsing"} for s in stages):
            _intake_touch(item, stage="processing-source")
            return
        if all(s == "matched" for s in stages):
            _intake_touch(item, stage="done")
            return
        _intake_touch(item, stage="queued")


def _ledger_canonical_stage(
    row: dict,
    ingest_snapshot: dict,
    *,
    stage: str,
    field: str = "status",
) -> str:
    canonical_key = f"canonical_{stage}_{field}"
    legacy_key = f"{stage}_{field}"
    value = row.get(canonical_key)
    if value in (None, ""):
        value = ingest_snapshot.get(canonical_key)
    if value in (None, ""):
        value = row.get(legacy_key)
    if value in (None, ""):
        value = ingest_snapshot.get(legacy_key)
    return str(value or "").strip().lower()


def render_intake_panel(*, max_rows: Optional[int] = None) -> None:
    st.markdown("**Drop PDFs**")
    uploader_key = (
        f"intake-dropzone::{int(st.session_state.get('intake_dropzone_nonce') or 0)}"
    )

    def _handle_drop() -> None:
        files = st.session_state.get(uploader_key) or []
        if not files:
            return
        inbox = st.session_state.get("intake_inbox")
        if not isinstance(inbox, list):
            inbox = []
            st.session_state["intake_inbox"] = inbox
        blobs = st.session_state.get("intake_blobs")
        if not isinstance(blobs, dict):
            blobs = {}
            st.session_state["intake_blobs"] = blobs
        for file_obj in files:
            if file_obj is None:
                continue
            try:
                data = bytes(file_obj.getbuffer())
            except Exception:
                continue
            item_id = str(uuid4())
            intent = _intake_guess_intent(
                filename=str(getattr(file_obj, "name", "")), data=data
            )
            item = {
                "id": item_id,
                "filename": str(getattr(file_obj, "name", "document.pdf")),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest().lower(),
                "intent": intent,
                "stage": "queued",
                "last_event_at": _now_iso(),
                "error": "",
                "note": "",
                "doc_id": None,
                "source_queue_item_ids": [],
                "intent_guess": intent,
                "auto_routed": False,
                "override_available": False,
                "routed_intent": intent if intent in {"citing", "source"} else None,
            }
            inbox.insert(0, item)
            blobs[item_id] = data
            if intent == "citing":
                _intake_route_citing(item_id)
            elif intent == "source":
                _intake_route_source(item_id)
            else:
                item["auto_routed"] = True
                item["override_available"] = False
                item["routed_intent"] = "citing"
                item["note"] = (
                    "Intent unclear; processing metadata first, then placement can be refined."
                )
                _intake_route_citing(item_id)
        st.session_state["intake_dropzone_nonce"] = (
            int(st.session_state.get("intake_dropzone_nonce") or 0) + 1
        )
        _rerun()

    st.file_uploader(
        "Drop PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=uploader_key,
        on_change=_handle_drop,
        label_visibility="collapsed",
    )

    inbox = st.session_state.get("intake_inbox") or []
    if not inbox:
        st.caption(
            "Drop one or more PDFs to start. Intent classification is deferred "
            "until metadata processing completes."
        )
        return

    shown = inbox[: int(max_rows)] if max_rows is not None else list(inbox)
    if max_rows is not None and len(inbox) > int(max_rows):
        st.caption(f"Showing {int(max_rows)} of {len(inbox)} intake items")

    for item in shown:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        _intake_refresh_item_status(item)

        item_id = str(item.get("id"))
        stage = str(item.get("stage") or "")
        intent = str(item.get("intent") or "unknown")
        icon = _intake_stage_icon(stage)
        cols = st.columns([0.9, 5.2, 2.2], gap="small")
        details_key = f"intake-details::{item_id}"
        if details_key not in st.session_state:
            st.session_state[details_key] = False

        with cols[0]:
            if st.button(icon, key=f"intake-icon::{item_id}", use_container_width=True):
                st.session_state[details_key] = not bool(
                    st.session_state.get(details_key)
                )

        with cols[1]:
            filename = str(item.get("filename") or "document.pdf")
            size_label = format_filesize(int(item.get("size") or 0))
            routed_intent = str(item.get("routed_intent") or intent).strip() or intent
            st.markdown(f"**{html.escape(filename)}**")
            if intent == "unknown" and bool(item.get("auto_routed")):
                st.caption(
                    "Intent: unknown (auto-routed to "
                    f"{routed_intent}) • Stage: {stage} • Size: {size_label}"
                )
            else:
                st.caption(f"Intent: {intent} • Stage: {stage} • Size: {size_label}")
            note = str(item.get("note") or "").strip()
            if note:
                st.caption(note)
            if str(item.get("error") or "").strip():
                st.error(str(item.get("error")))

            if bool(st.session_state.get(details_key)):
                with st.expander("Details", expanded=True):
                    st.write(f"Stage: {stage}")
                    st.write(f"Last event: {item.get('last_event_at')}")
                    if item.get("doc_id"):
                        st.write(f"Citing doc_id: {item.get('doc_id')}")
                    if item.get("source_queue_item_ids"):
                        st.write(
                            f"Source queue ids: {item.get('source_queue_item_ids')}"
                        )
                    if item.get("error"):
                        st.write(f"Error: {item.get('error')}")

        with cols[2]:
            if stage == "near-duplicate":
                matches = item.get("near_duplicate_matches") or []
                if matches:
                    st.warning("Near-duplicate detected in project")
                    for m in matches:
                        st.caption(
                            f"#{m.get('num')} {m.get('short')} (score {m.get('score')})"
                        )
                keep_key = f"intake-dup-keep::{item_id}"
                repl_key = f"intake-dup-replace::{item_id}"
                if st.button("Keep both", key=keep_key, use_container_width=True):
                    item["near_duplicate_decision"] = "keep"
                    item["stage"] = "uploaded"
                    _rerun()
                if st.button(
                    "Replace existing",
                    key=repl_key,
                    help=(
                        "Warn: prior review work may not transfer; "
                        "default is Keep both."
                    ),
                    use_container_width=True,
                ):
                    item["near_duplicate_decision"] = "replace"
                    item["note"] = (
                        (str(item.get("note") or "").strip() + "\n")
                        + (
                            "Replace requested (best-effort): "
                            "prior review work may not transfer."
                        )
                    ).strip()
                    item["stage"] = "uploaded"
                    _rerun()


def render_sources_panel(*, max_rows: Optional[int] = None) -> None:
    """Project-shared Source Inbox.

    Backed by `GET /attachments?archived=false` (project-scoped via X-Project-Id).
    """
    inject_attachment_panel_styles()
    attachment_queue.init_attachment_queue_state()

    # Ensure we show the backend-backed, project-shared inbox by default.
    attachment_queue.set_show_history(True)
    attachment_queue.set_show_archived(False)

    controls = st.columns([1, 1], gap="small")
    with controls[0]:
        if st.button(
            "Refresh sources",
            key="sources-refresh",
            use_container_width=True,
        ):
            attachment_queue.advance_inflight_items()
    with controls[1]:
        if st.button(
            "Archive all",
            key="sources-archive-all",
            help="Archive all currently-unarchived sources in this project.",
            use_container_width=True,
        ):
            archived = attachment_queue.archive_all_active()
            st.caption(f"Archived {archived} source(s)")

    # Keep the list fresh even without manual refresh.
    attachment_queue.sync_backend_state()

    items = attachment_queue.get_queue_items()

    def _is_global_source(item: dict) -> bool:
        if not isinstance(item, dict):
            return False
        if bool(item.get("archived")):
            return False
        if str(item.get("claim_id") or "").strip():
            return False
        if str(item.get("doc_id") or "").strip():
            return False
        if str(item.get("target_id") or "").strip():
            return False
        return True

    inbox = [it for it in items if _is_global_source(it)]
    if not inbox:
        st.caption("No sources yet. Upload via Drop PDFs.")
        return

    if max_rows is not None and len(inbox) > int(max_rows):
        st.caption(f"Showing {int(max_rows)} of {len(inbox)} source items")
        inbox = inbox[: int(max_rows)]

    for item in inbox:
        _render_source_bin_row(item)


def render_workspace_left_pane() -> None:

    st.markdown(
        '<div class="ws-pane-header"><div class="ws-pane-header__title">Admin</div></div>',
        unsafe_allow_html=True,
    )

    with st.expander("Scope selector", expanded=True):
        render_scope_selector_block()

    with st.expander("⚙ Settings", expanded=False):
        render_settings_controls()
        st.toggle("Dense", key=WORKSPACE_DENSE_MODE)

    if not scope_lock.has_applied_scope():
        _clear_unapplied_scope_workspace_state()
        st.info("Apply a user + project scope to unlock workspace activity.")
        return

    with st.expander("Upload documents", expanded=True):
        render_intake_panel(max_rows=None)

    with st.expander("Placed documents", expanded=True):
        render_documents_panel(max_rows=None)

    with st.expander("Stray documents", expanded=False):
        render_sources_panel(max_rows=None)

    with st.expander("Project", expanded=False):
        render_project_panel()

    with st.expander("Advanced", expanded=False):
        st.file_uploader(
            "CSV & TEI files",
            type=["csv", "xml"],
            accept_multiple_files=True,
            key="uploaded_files",
            on_change=handle_upload,
        )
        st.markdown("---")

def render_activity_console_strip() -> None:
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    lines.append(f"[{now}] Workspace active")

    # Scope / selection summary
    uid = scope_lock.get_applied_uid() or "(none)"
    pid = scope_lock.get_applied_project_id() or "(none)"
    selected_doc = str(st.session_state.get("selected_doc_id") or "").strip() or "(none)"
    lines.append(f"[scope] uid={uid} project={pid} selected_doc={selected_doc}")

    # Rail summary
    followed = st.session_state.get("followed_citations") or []
    if isinstance(followed, list):
        lines.append(f"[rail] followed_citations={len(followed)}")

    # Ledger diagnostics (ephemeral, high-signal)
    try:
        rows = _ledger_rows_cached()
        placeholders = sum(1 for r in rows if not bool((r or {}).get("anchored")))
        anchored = sum(1 for r in rows if bool((r or {}).get("anchored")))
        duplicates_by_ingest = {}
        for r in rows:
            iid = str((r or {}).get("ingest_id") or "").strip()
            if iid:
                duplicates_by_ingest[iid] = duplicates_by_ingest.get(iid, 0) + 1
        dup_count = sum(1 for c in duplicates_by_ingest.values() if int(c) > 1)
        lines.append(
            f"[ledger] rows={len(rows)} anchored={anchored} placeholders={placeholders} dup_ingest={dup_count}"
        )
    except Exception:
        pass

    # Intake items
    inbox = st.session_state.get("intake_inbox") or []
    for item in inbox[-8:]:
        if not isinstance(item, dict):
            continue
        fname = str(item.get("filename") or "document.pdf")
        stage = str(item.get("stage") or "queued")
        intent = str(item.get("routed_intent") or item.get("intent") or "").strip() or "?"
        lines.append(f"[intake] {fname} -> {stage} ({intent})")

    # Attachment queue items
    try:
        snap = attachment_queue.get_queue_snapshot() or {}
        summary = snap.get("summary") or {}
        if isinstance(summary, dict):
            parts = []
            for k in ["pending", "converting", "parsing", "matched", "error"]:
                parts.append(f"{k}={int(summary.get(k, 0) or 0)}")
            lines.append(f"[attachments] {' '.join(parts)}")
        for att in (attachment_queue.get_queue_items() or [])[-6:]:
            if not isinstance(att, dict):
                continue
            name = str(att.get("filename") or att.get("id") or "attachment")
            status = str(att.get("status") or "pending")
            lines.append(f"[attachment] {name} -> {status}")
    except Exception:
        pass

    lines = lines[-30:]
    st.session_state["activity_console_lines"] = lines

    st.markdown("<div class='activity-strip'>", unsafe_allow_html=True)
    preview = "\n".join(lines[-4:]) if lines else "(idle)"
    if st.button(
        ":material/terminal: Activity Console",
        key="activity-console-toggle",
        use_container_width=True,
    ):
        st.session_state["activity_console_open"] = not bool(
            st.session_state.get("activity_console_open")
        )
    st.code(preview, language="bash")
    if bool(st.session_state.get("activity_console_open")):
        st.code("\n".join(lines), language="bash")
    st.markdown("</div>", unsafe_allow_html=True)


def draw_workspace() -> None:
    scope_lock.ensure_seeded()

    # Top banner - outside columns
    st.markdown('<div class="app-banner">os-ERIN</div>', unsafe_allow_html=True)
    _render_scope_runtime_block()
    
    # CSS for all panes
    st.markdown("""
    <style>
    /* Remove top gap */
    .block-container {
        padding-top: 0rem !important;
    }
    header[data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
    }
    div[data-testid="stToolbar"] { display: none !important; }
    [data-testid="stAppViewContainer"] { margin-top: 0 !important; }
    /* Top banner */
    .app-banner {
        font-size: 1.4rem;
        font-weight: bold;
        padding: 0.5rem 1rem;
        background: var(--ws-surface);
        border-bottom: 1px solid var(--ws-border);
        margin-bottom: 0rem;
    }
    .ms {
        font-family: "Material Symbols Outlined";
        font-weight: 400;
        font-style: normal;
        font-size: 1.05rem;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        direction: ltr;
    }
    /* Sticky headers - adjust positions */
    .ws-pane-header {
        position: sticky !important;
        top: 0px !important;
        z-index: 100 !important;
        background: var(--ws-surface) !important;
    }
    .ws-contextbar {
        position: sticky !important;
        top: 0px !important;
        z-index: 99 !important;
        background: var(--ws-app-bg) !important;
    }
    .ws-tabs {
        position: sticky !important;
        top: 60px !important;
        z-index: 98 !important;
        background: var(--ws-app-bg) !important;
    }
    div[data-testid="stExpander"] summary {
        background: var(--ws-app-bg) !important;
        margin: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        overscroll-behavior: contain !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] div[data-testid="stVerticalBlockBorderWrapper"] {
        height: 76vh !important;
    }
    .citation-workflow-rail div[data-testid="stButton"] > button {
        justify-content: flex-start !important;
        text-align: left !important;
    }
    div[data-testid="stButton"] > button {
        justify-content: flex-start !important;
        text-align: left !important;
    }
    .activity-strip {
        margin-top: 0.5rem;
        border-top: 1px solid var(--ws-border);
        padding-top: 0.25rem;
    }
    .ledger-label--link {
        text-decoration: underline;
        text-underline-offset: 2px;
        color: var(--ws-text);
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([3, 6, 4], gap="large")
    with left:
        left_pane = st.container(height=1200, border=False)
        with left_pane:
            render_workspace_left_pane()
    with center:
        center_pane = st.container(height=1200, border=False)
    with right:
        right_pane = st.container(height=1200, border=False)
    if scope_lock.has_applied_scope():
        draw_ingestion_panel(center=center_pane, right=right_pane)
        render_activity_console_strip()
    else:
        _clear_unapplied_scope_workspace_state()
        with center_pane:
            st.info(
                "Workspace is locked until a valid user + project scope is applied. "
                "Use Scope selector and click Apply/Switch."
            )
        with right_pane:
            st.info("Activity remains gated until scope apply succeeds.")


def draw_ingestion_panel(*, center, right) -> None:
    # Ingestion controls live in the left pane; this function hosts the
    # center (Document/Review) and right (collector) panes.

    # Query params are used for chip navigation (?doc=...&cite=...&target=...).
    # We apply them early (even in a fresh session), then clear them to avoid
    # sticky restarts.
    def _read_query_params() -> dict:
        try:
            raw = st.query_params  # type: ignore[attr-defined]
            return {key: raw.get(key) for key in raw.keys()}
        except Exception:
            return st.experimental_get_query_params()

    params = _read_query_params()
    param_doc = params.get("doc")
    param_docnum = params.get("docnum")
    param_cite = params.get("cite")
    param_target = params.get("target")
    param_sid = params.get("sid")
    if isinstance(param_doc, list):
        param_doc = param_doc[0] if param_doc else None
    if isinstance(param_cite, list):
        param_cite = param_cite[0] if param_cite else None
    if isinstance(param_target, list):
        param_target = param_target[0] if param_target else None
    if isinstance(param_sid, list):
        param_sid = param_sid[0] if param_sid else None
    should_clear_params = bool(param_doc or param_docnum or param_cite or param_target or param_sid)

    docs = st.session_state.get("ingested_docs")
    if docs is None or not st.session_state.get("_ingested_docs_loaded"):
        docs = refresh_ingested_docs(show_error=False)
        st.session_state["_ingested_docs_loaded"] = True
    elif should_clear_params:
        docs = refresh_ingested_docs(show_error=False)

    if (not param_doc) and param_docnum:
        try:
            wanted_num = int(str(param_docnum))
        except Exception:
            wanted_num = None
        if wanted_num is not None:
            found_doc_id = None
            for d in (docs or []):
                if not isinstance(d, dict):
                    continue
                try:
                    if int(d.get("num") or -1) == wanted_num and d.get("id"):
                        found_doc_id = str(d.get("id"))
                        break
                except Exception:
                    continue
            if not found_doc_id:
                for r in (_ledger_rows_cached() or []):
                    if not isinstance(r, dict):
                        continue
                    try:
                        if int(r.get("num") or -1) == wanted_num and r.get("ingest_id"):
                            found_doc_id = str(r.get("ingest_id"))
                            break
                    except Exception:
                        continue
            if found_doc_id:
                param_doc = found_doc_id

    if param_doc and str(param_doc) != str(
        st.session_state.get("selected_doc_id") or ""
    ):
        st.session_state["selected_doc_id"] = str(param_doc)
        load_selected_document(show_error=False)

    if should_clear_params:
        try:
            st.query_params.clear()  # type: ignore[attr-defined]
        except Exception:
            try:
                st.experimental_set_query_params()
            except Exception:
                pass

    if not docs:
        with center:
            st.info("Upload a PDF in the left pane to begin.")
        with right:
            st.info("Select a citation to build a citing-span list.")
        return
    doc_id = st.session_state.get("selected_doc_id")
    if not doc_id:
        with center:
            st.info("Select a placed document to view details.")
        with right:
            st.info("Select a placed document to load citing spans.")
        return
    document = st.session_state.get("active_document")
    if not document or document.get("id") != doc_id:
        document = load_selected_document(show_error=False)
    if not document:
        with center:
            st.info("Select a PDF to view details.")
        return

    # Document details are rendered at the bottom of the center pane.
    extraction = document.get("extraction") or {}
    extraction_data = extraction.get("data") or {}
    body_paragraphs = (
        (extraction_data.get("body") or {}).get("paragraphs") or []
        if isinstance(extraction_data, dict)
        else []
    )
    resolution = document.get("resolution") or {}
    resolution_data = resolution.get("data") or []

    if not body_paragraphs:
        with center:
            st.info("No extracted reading text is available for this document yet.")

    # NOTE: Citation context fetching is used to render the center-pane header.
    # Initialize api_url before defining any closures that reference it.
    api_url = get_api_url()

    selected_index = st.session_state.get("citation_selected_index")
    selected_target = normalize_target_id(
        st.session_state.get("citation_selected_target")
    )

    def _get_context_cached(citation_index: int, target_id: str | None) -> dict:
        cache = st.session_state.get("citation_context_cache") or {}
        key = (doc_id, int(citation_index), normalize_target_id(target_id))
        if key in cache:
            return cache[key] or {}
        try:
            response = get_citation_context(
                api_url,
                doc_id,
                int(citation_index),
                target_id=normalize_target_id(target_id),
                project_id=get_project_id(),
                user_id=_active_reviewer_uid(),
            )
            context = response.get("context") or {}
        except RuntimeError as exc:
            context = {"error": str(exc)}
        cache[key] = context
        st.session_state["citation_context_cache"] = cache
        if selected_index is not None and int(selected_index) == int(citation_index):
            if normalize_target_id(target_id) == selected_target:
                st.session_state["citation_context"] = context
        return context

    if selected_index is not None:
        # Warm cache for selected citation without rendering legacy header text.
        _get_context_cached(int(selected_index), selected_target)
    extraction_stage = (
        (document.get("extraction") or {}) if isinstance(document, dict) else {}
    )
    extraction_complete = (
        extraction_stage.get("status") or ""
    ).strip().lower() == "complete"
    body_payload: dict = {}
    body_error: str | None = None
    paragraphs: list[dict] = []
    if extraction_complete:
        try:
            body_payload = get_document_body(
                api_url,
                doc_id,
                project_id=get_project_id(),
            )
        except RuntimeError as exc:
            body_error = str(exc)
            extraction_complete = False
        else:
            paragraphs = body_payload.get("paragraphs") or []
            if not paragraphs:
                extraction_complete = False

    def select_citation(index: int, target_id: str | None) -> None:
        normalized_target = normalize_target_id(target_id)
        st.session_state["citation_selected_index"] = index
        st.session_state["citation_selected_target"] = normalized_target

        sentence_id = None
        pending_tuple = st.session_state.pop("pending_callout_tuple", None)
        if isinstance(pending_tuple, dict) and pending_tuple.get("doc_id") == doc_id:
            try:
                pending_index = int(pending_tuple.get("citation_index"))
            except Exception:
                pending_index = None
            if pending_index is not None and pending_index == int(index):
                if (
                    normalize_target_id(pending_tuple.get("target_id"))
                    == normalized_target
                ):
                    sentence_id = pending_tuple.get("sentence_id")

        if sentence_id is None:
            # Best-effort provenance: map selected callout to the first sentence_id
            # we can find in the loaded document body. (Hyperlink navigation can't
            # set session state like the old button rows did.)
            try:
                desired_idx = int(index)
            except Exception:
                desired_idx = None
            if desired_idx is not None:
                for para in paragraphs:
                    para_sentences = para.get("sentences") or []
                    if not para_sentences:
                        para_sentences = [
                            {
                                "segments": para.get("segments") or [],
                                "citation_indices": para.get("citation_indices") or [],
                            }
                        ]
                    for sent in para_sentences:
                        for seg in sent.get("segments") or []:
                            if seg.get("type") != "citation":
                                continue
                            try:
                                seg_idx = int(seg.get("citation_index"))
                            except Exception:
                                continue
                            if seg_idx != desired_idx:
                                continue
                            seg_target = normalize_target_id(seg.get("target_id"))
                            if seg_target != normalized_target:
                                continue
                            sentence_id = seg.get("sentence_id") or sent.get(
                                "sentence_id"
                            )
                            break
                        if sentence_id:
                            break
                    if sentence_id:
                        break
        st.session_state["citation_selected_sentence_id"] = sentence_id
        st.session_state["selected_callout_tuple"] = {
            "doc_id": doc_id,
            "citation_index": int(index),
            "target_id": normalized_target,
            "sentence_id": sentence_id,
        }
        st.session_state["citation_context_key"] = None
        st.session_state["citation_context"] = None
        st.session_state["citation_context_error"] = None
        st.session_state["citation_last_context_request"] = None
        st.session_state["citation_follow_open"] = False
        st.session_state["citation_graph_key"] = None
        st.session_state["citation_graph"] = None
        st.session_state["citation_graph_error"] = None
        st.session_state["citation_last_graph_request"] = None

        # Persist the selection in the workflow rail.
        st.session_state["workflow_active_citation"] = index

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
                project_id=get_project_id(),
                user_id=_active_reviewer_uid(),
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
                project_id=get_project_id(),
                user_id=_active_reviewer_uid(),
            )
        except RuntimeError as exc:
            st.session_state["citation_graph_error"] = str(exc)
            show_api_error(f"Failed to load citation graph: {exc}")
            placeholder.empty()
            return
        placeholder.empty()
        st.session_state["citation_graph"] = response

    inject_citation_styles()

    def _set_query_params(
        *, doc: str, cite: Optional[int], target: Optional[str]
    ) -> None:
        payload: dict = {"doc": doc}
        scope_uid = scope_lock.get_applied_uid()
        scope_project = scope_lock.get_applied_project_id()
        if scope_uid:
            payload["uid"] = scope_uid
        if scope_project:
            payload["project"] = scope_project
        if cite is not None:
            payload["cite"] = str(int(cite))
        if target:
            payload["target"] = str(target)
        try:
            st.query_params.clear()  # type: ignore[attr-defined]
            for key, value in payload.items():
                st.query_params[key] = value  # type: ignore[attr-defined]
        except Exception:
            try:
                st.experimental_set_query_params(**payload)
            except Exception:
                return

    def _has_unsaved_claim_edits(citation_index: Optional[int]) -> bool:
        if citation_index is None:
            return False
        try:
            cite_idx = int(citation_index)
        except Exception:
            return False
        reviewer_state = _reviewer_state_suffix(_active_reviewer_uid())
        ta_key = f"{canonical_segments_key(citation_index=cite_idx)}::{reviewer_state}"
        current_text = str(st.session_state.get(ta_key) or "")
        tgt = normalize_target_id(st.session_state.get("citation_selected_target"))
        cite_key = f"{doc_id}::{int(cite_idx)}::{tgt or ''}"
        stored = (
            (st.session_state.get("citation_segments_by_reviewer") or {})
            .get(reviewer_state, {})
            .get(cite_key, [])
        )
        stored_text = "\n".join(
            [str(line).strip() for line in (stored or []) if str(line).strip()]
        )
        return current_text.strip() != stored_text.strip()

    def _save_claim_lines_for_citation(
        citation_index: int, target_id: str | None
    ) -> int:
        cite_idx = int(citation_index)
        context = _get_context_cached(cite_idx, target_id)
        reviewer_state = _reviewer_state_suffix(_active_reviewer_uid())
        ta_key = f"{canonical_segments_key(citation_index=cite_idx)}::{reviewer_state}"
        lines = [
            ln.strip()
            for ln in str(st.session_state.get(ta_key) or "").splitlines()
            if ln.strip()
        ]
        cite_key = f"{doc_id}::{int(cite_idx)}::{normalize_target_id(target_id) or ''}"
        st.session_state.setdefault("citation_segments_by_reviewer", {}).setdefault(
            reviewer_state, {}
        )[cite_key] = lines

        # Best-effort: persist confirmed claims so backend indexes claim nodes
        # for the Graph tab (claim:{doc}:{sentence_id}:{claim_index}).
        sentence_id = None
        selected_tuple = st.session_state.get("selected_callout_tuple")
        if isinstance(selected_tuple, dict):
            try:
                tuple_idx = int(selected_tuple.get("citation_index"))
            except Exception:
                tuple_idx = None
            if (
                str(selected_tuple.get("doc_id") or "").strip() == str(doc_id)
                and tuple_idx is not None
                and tuple_idx == int(cite_idx)
                and normalize_target_id(selected_tuple.get("target_id"))
                == normalize_target_id(target_id)
            ):
                sentence_id = selected_tuple.get("sentence_id")
        if not sentence_id:
            sentence_id = st.session_state.get("citation_selected_sentence_id")
        if not sentence_id:
            sentence_id = (context or {}).get("sentence_id")

        sentence_text = (
            context.get("citing_sentence")
            or context.get("sentence")
            or context.get("citing_prefix")
            or ""
        )
        sentence_text = str(sentence_text or "").strip()

        def _segment_to_claim_index(segment_id: str, fallback: int) -> int:
            text = str(segment_id or "").strip()
            m = re.match(r"^\d+([a-z])$", text, flags=re.IGNORECASE)
            if not m:
                return int(fallback)
            letter = m.group(1).lower()
            return 1 + (ord(letter) - ord("a"))

        if sentence_id and sentence_text and lines:
            try:
                confirmed_claims: list[dict] = []
                for idx, line in enumerate(lines):
                    parsed = to_segment_dict(line)
                    segment_id = parsed.get("segment_id") or ""
                    claim_text = (parsed.get("claim") or line or "").strip()
                    if not claim_text:
                        continue
                    confirmed_claims.append(
                        {
                            "claim_index": int(
                                _segment_to_claim_index(segment_id, idx + 1)
                            ),
                            "parsed_text": claim_text,
                        }
                    )
                if confirmed_claims:
                    prov = {
                        "doc_id": str(doc_id),
                        "citation_index": int(cite_idx),
                        "target_id": normalize_target_id(target_id),
                    }
                    prov = _maybe_attach_citation_anchor(prov)
                    confirm_claims(
                        api_url,
                        document_id=str(doc_id),
                        sentence_id=str(sentence_id),
                        sentence_text=sentence_text,
                        citation_index=int(cite_idx),
                        target_id=normalize_target_id(target_id),
                        reviewer_uid=str(_active_reviewer_uid() or "").strip(),
                        confirmed_claims=confirmed_claims,
                        segmentation_model=str(
                            st.session_state.get("selected_model") or "local"
                        ),
                        cited_work_id=prov.get("cited_work_id"),
                        citation_anchor=prov.get("citation_anchor"),
                        project_id=get_project_id(),
                    )
            except Exception:
                pass
        primary_callout = context.get("callout") or "citation"
        reference_hint = {
            "callout": primary_callout,
            "reference_id": normalize_target_id(target_id),
        }
        saved = 0
        placed = 0
        for idx, line in enumerate(lines):
            parsed = to_segment_dict(line)
            segment_id = parsed.get("segment_id") or f"seg-{idx+1}"
            claim_text = parsed.get("claim") or line
            claim_id = f"cite:{doc_id}:{cite_idx}:{reviewer_state}:{segment_id}"
            claim_queue.register_claim(
                claim_id,
                claim=claim_text,
                callout=primary_callout,
                doc_id=doc_id,
                reference_id=normalize_target_id(target_id),
                reference_hint=reference_hint,
            )
            saved += 1

            if normalize_target_id(target_id):
                try:
                    resp = auto_place_claim_source(
                        api_url,
                        claim_id=str(claim_id),
                        doc_id=str(doc_id),
                        citation_index=int(cite_idx),
                        target_id=str(normalize_target_id(target_id)),
                        project_id=get_project_id(),
                        user_id=_active_reviewer_uid(),
                    )
                except RuntimeError:
                    resp = None
                if (resp or {}).get("attachment"):
                    placed += 1

            if claim_text and str(claim_text).strip():
                evidence_store.queue_rerun(
                    claim_id,
                    claim_text=str(claim_text).strip(),
                    note="auto-claim-save",
                    quiet=True,
                )
        return saved

    pending = st.session_state.get("pending_citation_selection")
    if pending:
        with center:
            st.warning("You have unsaved claim edits. Save before switching citations?")
            action_cols = st.columns([1, 1, 2])
            with action_cols[0]:
                if st.button("Save + switch", key="pending-cite-save"):
                    current_idx = st.session_state.get("citation_selected_index")
                    current_tgt = normalize_target_id(
                        st.session_state.get("citation_selected_target")
                    )
                    if current_idx is not None:
                        _save_claim_lines_for_citation(int(current_idx), current_tgt)
                    select_citation(
                        int(pending["citation_index"]), pending.get("target_id")
                    )
                    _set_query_params(
                        doc=doc_id,
                        cite=int(pending["citation_index"]),
                        target=pending.get("target_id"),
                    )
                    st.session_state.pop("pending_citation_selection", None)
                    _rerun()
            with action_cols[1]:
                if st.button("Discard + switch", key="pending-cite-discard"):
                    current_idx = st.session_state.get("citation_selected_index")
                    if current_idx is not None:
                        cite_idx = int(current_idx)
                        reviewer_state = _reviewer_state_suffix(_active_reviewer_uid())
                        current_tgt = normalize_target_id(
                            st.session_state.get("citation_selected_target")
                        )
                        cite_key = f"{doc_id}::{int(cite_idx)}::{current_tgt or ''}"
                        stored = (
                            (
                                st.session_state.get("citation_segments_by_reviewer")
                                or {}
                            )
                            .get(reviewer_state, {})
                            .get(cite_key, [])
                        )
                        segments_key = (
                            f"{canonical_segments_key(citation_index=cite_idx)}"
                            f"::{reviewer_state}"
                        )
                        st.session_state[segments_key] = "\n".join(stored or [])
                    select_citation(
                        int(pending["citation_index"]), pending.get("target_id")
                    )
                    _set_query_params(
                        doc=doc_id,
                        cite=int(pending["citation_index"]),
                        target=pending.get("target_id"),
                    )
                    st.session_state.pop("pending_citation_selection", None)
                    _rerun()
            with action_cols[2]:
                if st.button("Cancel", key="pending-cite-cancel"):
                    current_idx = st.session_state.get("citation_selected_index")
                    current_tgt = normalize_target_id(
                        st.session_state.get("citation_selected_target")
                    )
                    _set_query_params(
                        doc=doc_id,
                        cite=int(current_idx) if current_idx is not None else None,
                        target=current_tgt,
                    )
                    st.session_state.pop("pending_citation_selection", None)

    if pending is None and param_cite is not None:
        try:
            desired_idx = int(str(param_cite))
        except ValueError:
            desired_idx = None
        current_idx = st.session_state.get("citation_selected_index")
        desired_tgt = str(param_target) if param_target else None
        if desired_idx is not None and (
            current_idx != desired_idx
            or normalize_target_id(st.session_state.get("citation_selected_target"))
            != normalize_target_id(desired_tgt)
        ):
            if _has_unsaved_claim_edits(current_idx):
                st.session_state["pending_citation_selection"] = {
                    "doc_id": doc_id,
                    "citation_index": desired_idx,
                    "target_id": normalize_target_id(desired_tgt),
                }
            else:
                select_citation(desired_idx, desired_tgt)

    def _follow_citation(
        citation_index: int,
        target_id: str | None,
        *,
        span_key: str | None = None,
    ) -> None:
        """Add citation to chase queue, preserving discovery order.
        
        Writes to server-backed opinion layer for durability.
        Falls back to session_state on error.
        """
        normalized_target = normalize_target_id(target_id)
        _rail_debug_log(
            "follow_start",
            doc_id=str(doc_id),
            citation_index=int(citation_index),
            target_id=str(normalized_target or ""),
            span_key=str(span_key or ""),
        )
        
        # Try server-backed approach first
        try:
            reviewer_uid = _active_reviewer_uid()
            if not reviewer_uid:
                return
            current_project_id = get_project_id()
            if api_url and current_project_id:
                # Prefer per-occurrence sid when available so multi-citation
                # sentences accumulate as distinct rail items.
                span_id = resolve_span_id(
                    api_url,
                    current_project_id,
                    doc_id,
                    citation_index,
                    normalized_target,
                )
                event_span_id = str(
                    (str(span_key or "").strip() and f"sid:{str(span_key).strip()}")
                    or span_id
                    or f"sid:auto:{int(citation_index)}:{normalize_target_id(normalized_target) or ''}"
                )
                append_follow(
                    api_url=api_url,
                    project_id=current_project_id,
                    reviewer_uid=reviewer_uid,
                    doc_id=doc_id,
                    citation_index=int(citation_index),
                    target_id=normalized_target,
                    span_id=event_span_id,
                    status="follow",
                    idempotency_key=(
                        f"follow:{event_span_id}:{int(citation_index)}:"
                        f"{normalize_target_id(normalized_target) or ''}:follow"
                    ),
                )
                _rail_debug_log(
                    "follow_append_server_ok",
                    event_span_id=str(event_span_id),
                    citation_index=int(citation_index),
                    target_id=str(normalized_target or ""),
                )
                # Invalidate cache to force refresh
                st.session_state.pop("_followed_citations_cache", None)
        except Exception as exc:
            _rail_debug_log(
                "follow_append_server_error",
                citation_index=int(citation_index),
                target_id=str(normalized_target or ""),
                error=str(exc),
            )
            # Fall back to session_state on any error
            pass
        
        # Always maintain a local ordered list for robust UI behavior.
        followed = st.session_state.get("followed_citations") or []
        resolved_span_key = str(span_key or "").strip() or None
        if not resolved_span_key:
            resolved_span_key = (
                f"auto-{int(citation_index)}-{len(followed)+1}-{int(time.time()*1000)}"
            )

        entry = {
            "doc_id": doc_id,
            "citation_index": int(citation_index),
            "target_id": normalized_target,
            "span_key": resolved_span_key,
        }
        anchor_map = st.session_state.get("citation_anchor_map") or {}
        if resolved_span_key and isinstance(anchor_map, dict):
            info = anchor_map.get(str(resolved_span_key)) or {}
            if isinstance(info, dict):
                if str(info.get("doc_id") or "") == str(doc_id):
                    entry["snippet"] = str(info.get("snippet") or "").strip() or None
                    entry["order"] = int(info.get("order") or 0)
        existing_same_sid = None
        if resolved_span_key:
            for existing in followed:
                if str(existing.get("span_key") or "") == str(resolved_span_key):
                    existing_same_sid = existing
                    break

        if existing_same_sid is not None:
            _rail_debug_log(
                "follow_local_same_sid_skip",
                count=len(followed),
                span_key=str(resolved_span_key or ""),
                citation_index=int(citation_index),
            )
        elif entry not in followed:
            followed.append(entry)
            followed.sort(
                key=lambda item: (
                    int(item.get("citation_index") or 0),
                    int(item.get("order") or 0),
                    str(item.get("target_id") or ""),
                )
            )
            st.session_state["followed_citations"] = followed
            _rail_debug_log(
                "follow_local_append",
                count=len(followed),
                added_span_key=str(resolved_span_key or ""),
                citation_index=int(citation_index),
            )
        else:
            _rail_debug_log(
                "follow_local_duplicate_skip",
                count=len(followed),
                span_key=str(resolved_span_key or ""),
                citation_index=int(citation_index),
            )

    if param_cite is not None:
        try:
            event_key = "|".join(
                [
                    str(doc_id or ""),
                    str(param_cite or ""),
                    str(param_target or ""),
                    str(param_sid or ""),
                ]
            )
            if str(st.session_state.get("_last_follow_event") or "") != event_key:
                st.session_state["_last_follow_event"] = event_key
                _rail_debug_log("follow_event_new", event_key=event_key)
                if param_sid:
                    st.session_state["active_queue_span_key"] = str(param_sid)
                _follow_citation(
                    int(str(param_cite)),
                    str(param_target) if param_target else None,
                    span_key=str(param_sid) if param_sid else None,
                )
                # If this click belongs to a clustered citespan sentence,
                # enqueue all sibling citations in that same sentence.
                sid_val = str(param_sid or "").strip()
                anchor_map = st.session_state.get("citation_anchor_map") or {}
                if sid_val and isinstance(anchor_map, dict):
                    info = anchor_map.get(sid_val) or {}
                    siblings = info.get("related_citations") if isinstance(info, dict) else []
                    if isinstance(siblings, list):
                        for sibling in siblings:
                            try:
                                s_idx = int(sibling.get("citation_index"))
                            except Exception:
                                continue
                            s_tgt = normalize_target_id(sibling.get("target_id"))
                            if s_idx == int(str(param_cite)) and s_tgt == normalize_target_id(param_target):
                                continue
                            _follow_citation(s_idx, s_tgt, span_key=sid_val)
            else:
                _rail_debug_log("follow_event_duplicate_skip", event_key=event_key)
            
        except ValueError:
            pass

    # One-shot UI intent triggered by subpanels.
    chase_intent = st.session_state.pop("chase_intent", None)
    if chase_intent and chase_intent.get("doc_id") == doc_id:
        try:
            cite_idx = int(chase_intent.get("citation_index"))
        except Exception:
            cite_idx = None
        tgt = normalize_target_id(chase_intent.get("target_id"))
        if cite_idx is not None:
            st.session_state[WORKSPACE_ACTIVE_TAB] = WORKSPACE_TAB_REVIEW
            _set_query_params(doc=doc_id, cite=cite_idx, target=tgt)
            select_citation(cite_idx, tgt)

    def _claims_for_citation(cite_idx: int) -> list[str]:
        reviewer_state = _reviewer_state_suffix(_active_reviewer_uid())
        out: list[str] = []
        for record in claim_queue.get_claim_records():
            rid = record.get("id")
            if not rid:
                continue
            if str(rid).startswith(f"cite:{doc_id}:{int(cite_idx)}:"):
                parts = str(rid).split(":")
                # cite:{doc}:{cite_idx}:{reviewer}:{segment}
                if len(parts) >= 5:
                    if parts[3] != reviewer_state:
                        continue
                # Legacy: cite:{doc}:{cite_idx}:{segment} treated as default reviewer.
                elif reviewer_state != "default":
                    continue
                out.append(str(rid))
        return out

    def _is_processed(cite_idx: int) -> bool:
        for cid in _claims_for_citation(cite_idx):
            attached = attachment_queue.get_claim_attachment(cid)
            if attached and str(attached.get("status")) == "matched":
                return True
        return False

    def _drop_followed_entry(cite_idx: int, tgt: Optional[str]) -> None:
        normalized_tgt = normalize_target_id(tgt)

        try:
            reviewer_uid = _active_reviewer_uid()
            current_project_id = get_project_id()
            if reviewer_uid and api_url and current_project_id:
                span_id = resolve_span_id(
                    api_url,
                    current_project_id,
                    doc_id,
                    int(cite_idx),
                    normalized_tgt,
                )
                if span_id:
                    append_follow(
                        api_url=api_url,
                        project_id=current_project_id,
                        reviewer_uid=reviewer_uid,
                        doc_id=doc_id,
                        citation_index=int(cite_idx),
                        target_id=normalized_tgt,
                        span_id=span_id,
                        status="ignore",
                        idempotency_key=f"follow:{span_id}:ignore",
                    )
                    st.session_state.pop("_followed_citations_cache", None)
        except Exception:
            pass

        st.session_state["followed_citations"] = [
            item
            for item in (st.session_state.get("followed_citations") or [])
            if not (
                item.get("doc_id") == doc_id
                and int(item.get("citation_index") or -1) == int(cite_idx)
                and normalize_target_id(item.get("target_id")) == normalized_tgt
            )
        ]

    def _render_chasing_panel(cite_idx: int, tgt: str | None, *, scope: str) -> None:
        meta = st.session_state.get("project_meta")
        reviewers = _project_reviewers(meta) if isinstance(meta, dict) else []

        # Prefer the widget-controlled value if present; it updates immediately
        # when the user switches Current user, while project_meta may lag behind
        # a network round-trip.
        active_from_widget = _normalize_reviewer_name(
            st.session_state.get("project-active-reviewer")
        )
        active_uid = active_from_widget or (_active_reviewer_uid() or None)
        chasing_panel.render(
            doc_id=doc_id,
            citation_index=int(cite_idx),
            target_id=normalize_target_id(tgt),
            scope=scope,
            get_context_cached=_get_context_cached,
            seg_via_llm=seg_via_llm,
            to_segment_dict=to_segment_dict,
            claim_queue_register=claim_queue.register_claim,
            format_reference_summary=format_reference_summary,
            render_retrieval_instructions=claim_queue.render_retrieval_instructions,
            api_url=api_url,
            project_id=get_project_id(),
            selected_model=st.session_state.get("selected_model"),
            active_reviewer_uid=active_uid,
            reviewers=reviewers,
            rerun=_rerun,
        )

    with right:
        st.markdown(
            '<div class="ws-pane-header">'
            '<div class="ws-pane-header__title">Citing spans</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        doc_header = ""
        ledger = _ledger_rows_cached()
        if isinstance(ledger, list):
            for r in ledger:
                if str(r.get("ingest_id") or "") == str(doc_id):
                    doc_header = str(r.get("short") or r.get("title") or "").strip()
                    break
        if not doc_header:
            doc_header = str(document.get("filename") or doc_id)
        st.markdown(
            f"<div class='ws-pane-header__meta'>From: {html.escape(doc_header)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ws-pane-body">', unsafe_allow_html=True)
        st.markdown('<div class="citation-workflow-rail">', unsafe_allow_html=True)

        # Build rail from durable server follows + local UI follows.
        followed: List[Dict[str, Any]] = []
        try:
            reviewer_uid = _active_reviewer_uid()
            if not reviewer_uid:
                return
            current_project_id = get_project_id()
            if api_url and current_project_id:
                server_follows = list_follows_by_doc(
                    api_url=api_url,
                    project_id=current_project_id,
                    reviewer_uid=reviewer_uid,
                    doc_id=doc_id,
                )
                for sf in server_follows:
                    sf_span_id = str(sf.get("span_id") or "").strip()
                    sf_span_key = sf_span_id.replace("sid:", "") if sf_span_id.startswith("sid:") else None
                    sf_cite_idx = int(sf.get("citation_index") or sf.get("sort_key") or 0)
                    sf_order = int(sf.get("sort_key") or sf_cite_idx or 0)
                    sf_snippet = None
                    anchor_map = st.session_state.get("citation_anchor_map") or {}
                    if sf_span_key and isinstance(anchor_map, dict):
                        info = anchor_map.get(sf_span_key) or {}
                        if isinstance(info, dict):
                            sf_order = int(info.get("order") or sf_order)
                            sf_snippet = str(info.get("snippet") or "").strip() or None
                    followed.append(
                        {
                            "doc_id": doc_id,
                            "citation_index": sf_cite_idx,
                            "target_id": sf.get("target_id"),
                            "span_id": sf_span_id,
                            "span_key": sf_span_key,
                            "order": sf_order,
                            "snippet": sf_snippet,
                        }
                    )
                st.session_state["_followed_citations_cache"] = followed
                _rail_debug_log(
                    "rail_server_follows",
                    count=len(server_follows),
                    doc_id=str(doc_id),
                )
        except Exception:
            cached = st.session_state.get("_followed_citations_cache") or []
            followed = [
                entry for entry in cached if entry.get("doc_id") == doc_id
            ]
            _rail_debug_log(
                "rail_server_follows_error",
                cached_count=len(followed),
                doc_id=str(doc_id),
            )

        local_followed = [
            entry
            for entry in (st.session_state.get("followed_citations") or [])
            if entry.get("doc_id") == doc_id
        ]
        for entry in local_followed:
            if entry not in followed:
                followed.append(entry)

        deduped_followed: list[dict[str, Any]] = []
        dedupe_index: dict[str, int] = {}
        for entry in followed:
            sid = str(entry.get("span_key") or entry.get("span_id") or "").strip()
            if sid.startswith("sid:"):
                sid = sid.replace("sid:", "", 1)
            if not sid:
                sid = (
                    f"cite:{int(entry.get('citation_index') or 0)}:"
                    f"{normalize_target_id(entry.get('target_id')) or ''}"
                )
            if sid in dedupe_index:
                pos = dedupe_index[sid]
                existing = deduped_followed[pos]
                merged = dict(existing)
                for key in (
                    "snippet",
                    "span_key",
                    "span_id",
                    "target_id",
                    "citation_index",
                    "order",
                ):
                    v_new = entry.get(key)
                    if v_new not in (None, "", 0):
                        merged[key] = v_new
                if int(merged.get("order") or 0) == 0:
                    merged["order"] = int(entry.get("citation_index") or 0)
                deduped_followed[pos] = merged
            else:
                candidate = dict(entry)
                if int(candidate.get("order") or 0) == 0:
                    candidate["order"] = int(candidate.get("citation_index") or 0)
                dedupe_index[sid] = len(deduped_followed)
                deduped_followed.append(candidate)
        followed = deduped_followed

        _rail_debug_log(
            "rail_merge",
            server_plus_local_count=len(followed),
            local_count=len(local_followed),
            doc_id=str(doc_id),
        )

        # Display in primary-text order.
        followed.sort(
            key=lambda item: (
                int(item.get("citation_index") or 0),
                int(item.get("order") or 0),
                str(item.get("target_id") or ""),
            )
        )
        _rail_debug_log(
            "rail_sorted",
            count=len(followed),
            keys=[
                {
                    "cite": int(item.get("citation_index") or 0),
                    "order": int(item.get("order") or 0),
                    "sid": str(item.get("span_key") or item.get("span_id") or ""),
                }
                for item in followed[:20]
            ],
        )

        def _queue_label(entry: dict) -> str:
            snippet = str(entry.get("snippet") or "").strip()
            if snippet:
                return _sentence_label(snippet)
            try:
                cite_idx = int(entry.get("citation_index") or 0)
            except Exception:
                cite_idx = 0
            tgt = normalize_target_id(entry.get("target_id"))
            context = _get_context_cached(cite_idx, tgt)
            cite_text = (
                context.get("citing_sentence")
                or context.get("sentence")
                or context.get("citing_prefix")
                or ""
            )
            label = _sentence_label(cite_text) if cite_text else f"Citation {cite_idx + 1}"
            if re.fullmatch(r"\(?\d{4}\)?\s*(and|;|,)?\s*", label.strip(), re.IGNORECASE):
                return f"Citation {cite_idx + 1}"
            if len(label.strip()) <= 8:
                return f"Citation {cite_idx + 1}"
            return label

        def _queue_status(cite_idx: int, tgt: Optional[str]) -> Dict[str, bool]:
            reviewer_state = _reviewer_state_suffix(_active_reviewer_uid())
            cite_key = f"{doc_id}::{int(cite_idx)}::{normalize_target_id(tgt) or ''}"
            has_saved = bool(
                (
                    (st.session_state.get("citation_segments_by_reviewer") or {})
                    .get(reviewer_state, {})
                    .get(cite_key)
                )
            )
            return {
                "has_saved": has_saved,
                "processed": _is_processed(int(cite_idx)),
            }

        def _queue_open(cite_idx: int, tgt: Optional[str]) -> None:
            select_citation(int(cite_idx), tgt)

        def _queue_drop(cite_idx: int, tgt: Optional[str]) -> None:
            _drop_followed_entry(int(cite_idx), normalize_target_id(tgt))

        drop_request = st.session_state.pop("queue_drop_request", None)
        if isinstance(drop_request, dict) and str(drop_request.get("doc_id") or "") == str(doc_id):
            try:
                rq_idx = int(drop_request.get("citation_index"))
            except Exception:
                rq_idx = None
            rq_tgt = normalize_target_id(drop_request.get("target_id"))
            if rq_idx is not None:
                _drop_followed_entry(rq_idx, rq_tgt)
                if selected_index is not None and int(selected_index) == int(rq_idx):
                    st.session_state["citation_selected_index"] = None
                    st.session_state["citation_selected_target"] = None

        def _queue_panel(cite_idx: int, tgt: Optional[str], scope: str) -> None:
            _render_chasing_panel(int(cite_idx), normalize_target_id(tgt), scope=scope)

        chase_queue_component.render(
            title="Citing spans",
            caption="Queue up citing spans to segment + chase while sources process.",
            followed=followed,
            selected_index=int(selected_index) if selected_index is not None else None,
            selected_target=normalize_target_id(selected_target),
            get_label=_queue_label,
            get_status=_queue_status,
            on_open=_queue_open,
            on_drop=_queue_drop,
            render_panel=_queue_panel,
            rerun=_rerun,
            scope="rail",
        )

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with center:
        doc_label = str(document.get("filename") or doc_id)
        if extraction_data.get("metadata", {}).get("title"):
            doc_label = str(extraction_data.get("metadata", {}).get("title"))

        doc_html = html.escape(doc_label)
        st.markdown(
            '<div class="ws-pane-header">'
            '<div class="ws-pane-header__title">Work area</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='ws-contextbar'><div class='ws-contextbar__doc'>{doc_html}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ws-pane-body">', unsafe_allow_html=True)
        st.session_state.setdefault(WORKSPACE_ACTIVE_TAB, WORKSPACE_TAB_DOCUMENT)
        st.markdown('<div class="ws-tabs">', unsafe_allow_html=True)
        workspace_tab = st.radio(
            "Workspace tab",
            [WORKSPACE_TAB_DOCUMENT, WORKSPACE_TAB_REVIEW, WORKSPACE_TAB_GRAPH],
            key=WORKSPACE_ACTIVE_TAB,
            horizontal=True,
            format_func=lambda opt: WORKSPACE_TAB_LABELS.get(opt, opt),
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if workspace_tab == WORKSPACE_TAB_DOCUMENT:
            if not extraction_complete:
                st.info("Run extraction to populate the document text.")
                if body_error:
                    st.caption(body_error)
            else:
                st.caption("Click an in-text citation chip to inspect context.")

            active_reviewer_uid = _active_reviewer_uid()
            j_store = judgment_store.JudgmentStore()
            if active_reviewer_uid:
                j_doc_state = j_store.sync_doc(
                    str(doc_id), reviewer_uid=active_reviewer_uid, include_drafts=True
                )
                if j_doc_state.get("error"):
                    st.caption(f"Judgments unavailable: {j_doc_state['error']}")
            else:
                j_doc_state = {}

            anchor_map: dict[str, dict[str, Any]] = {}
            order_counter = 0
            for para_idx, para in enumerate(paragraphs or []):
                para_sentences = para.get("sentences") or []
                if not para_sentences:
                    # Backward compatibility: older backend may return flat segments.
                    para_sentences = [
                        {
                            "segments": para.get("segments") or [],
                            "citation_indices": para.get("citation_indices") or [],
                        }
                    ]

                sentence_html: list[str] = []
                for sent_idx, sent in enumerate(para_sentences):
                    segments = sent.get("segments") or []
                    sentence_snippet = " ".join(
                        str(s.get("text") or "")
                        for s in segments
                        if s.get("type") == "text"
                    ).strip()
                    sentence_citations: list[dict[str, Any]] = []
                    for seg in segments:
                        if seg.get("type") != "citation":
                            continue
                        ci = seg.get("citation_index")
                        if ci is None:
                            continue
                        rel_label = str(
                            seg.get("label") or seg.get("callout") or ""
                        ).strip()
                        sentence_citations.append(
                            {
                                "citation_index": int(ci),
                                "target_id": normalize_target_id(seg.get("target_id")),
                                "label": rel_label,
                            }
                        )
                    sent_citations = set(sent.get("citation_indices") or [])
                    sent_selected = (
                        selected_index is not None and selected_index in sent_citations
                    )
                    sent_text_len = sum(
                        len(seg.get("text") or "")
                        for seg in segments
                        if seg.get("type") == "text"
                    )
                    # If TEI sentence nodes are very long, highlight chips only.
                    use_sentence_highlight = sent_selected and sent_text_len <= 320
                    sent_class = (
                        "citation-sentence-row citation-sentence-selected"
                        if use_sentence_highlight
                        else "citation-sentence-row"
                    )
                    parts: list[str] = []
                    for seg_idx, seg in enumerate(segments):
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
                        # fmt: off
                        raw_label = (
                            seg.get("label")
                            or seg.get("callout")
                            or "citation"
                        )
                        # fmt: on
                        label_text = str(raw_label)
                        if selected_index == cite_index:
                            stripped = label_text.strip()
                            if stripped and not stripped.startswith(("(", "[")):
                                label_text = f"({stripped})"
                            else:
                                label_text = stripped
                        chip_class = "citation-chip"
                        if selected_index == cite_index:
                            chip_class += " citation-chip-selected"
                        normalized_target = normalize_target_id(target_id)
                        span_key = hashlib.sha1(
                            f"{doc_id}:{para_idx}:{sent_idx}:{sentence_snippet}".encode("utf-8")
                        ).hexdigest()[:12]
                        order_counter += 1
                        snippet_text = sentence_snippet
                        anchor_map[span_key] = {
                            "doc_id": str(doc_id),
                            "citation_index": int(cite_index),
                            "target_id": normalized_target,
                            "order": int(order_counter),
                            "snippet": snippet_text,
                            "related_citations": sentence_citations,
                        }

                        if active_reviewer_uid:
                            status_snapshot = j_store.callout_status(
                                str(doc_id),
                                cite_index,
                                normalized_target,
                                reviewer_uid=active_reviewer_uid,
                            )
                        else:
                            status_snapshot = {
                                "validated": False,
                                "outcome": None,
                                "claim_ids": [],
                            }
                        validated = bool(status_snapshot.get("validated"))
                        outcome = status_snapshot.get("outcome") if validated else None
                        if validated:
                            chip_class += " citation-chip--validated"
                            if outcome in {"support", "contradict", "uncertain"}:
                                chip_class += f" citation-chip--{outcome}"
                        else:
                            chip_class += " citation-chip--unvalidated"

                        parts.append(f'<a id="{anchor}"></a>')
                        parts.append(f'<a id="cite-span-{span_key}"></a>')
                        href = _citation_href(
                            doc_id,
                            cite_index,
                            normalized_target,
                            anchor,
                            span_key=span_key,
                        )

                        # Render chip inline as a hyperlink so click drives existing
                        # ?doc=...&cite=...&target=... routing and chase collection.
                        parts.append(
                            (
                                f'<a class="{chip_class} citation-chip-link" '
                                f'href="{html.escape(href)}" target="_self">'
                                f"{html.escape(label_text)}"
                                "</a>"
                            )
                        )
                    rendered_sentence = " ".join(part for part in parts if part)
                    sentence_html.append(
                        f'<span class="{sent_class}">{rendered_sentence}</span>'
                    )

                rendered_para = " ".join(sentence_html)
                st.markdown(
                    f'<div class="citation-paragraph">{rendered_para}</div>',
                    unsafe_allow_html=True,
                )
            st.session_state["citation_anchor_map"] = anchor_map
            pending_span = str(st.session_state.pop("pending_scroll_span_key", "") or "").strip()
            if pending_span:
                st.markdown(
                    (
                        "<script>"
                        f"(function(){{var el=document.getElementById('cite-span-{pending_span}');"
                        "if(el){el.scrollIntoView({block:'center'});}}})();"
                        "</script>"
                    ),
                    unsafe_allow_html=True,
                )

        elif workspace_tab == WORKSPACE_TAB_REVIEW:
            if selected_index is None:
                st.info("Select a citation in Document to edit context + review.")
            else:
                _render_chasing_panel(
                    int(selected_index),
                    selected_target,
                    scope="tab",
                )
            st.divider()
            render_evidence_panel()

        else:
            st.markdown("### Surfing")

            live_surfing_panel.render(
                api_url=get_api_url(),
                seed_doc_id=str(st.session_state.get("selected_doc_id") or ""),
            )

        # Document details at bottom.
        with st.expander("Document details", expanded=False):
            st.write(
                {
                    "Filename": document.get("filename"),
                    "Uploaded": document.get("uploaded_at"),
                    "Status": document.get("status"),
                    "Size (bytes)": document.get("size_bytes"),
                }
            )
            metadata = extraction_data.get("metadata") or {}
            if metadata:
                rows = [
                    {"Field": key, "Value": stringify_value(value)}
                    for key, value in metadata.items()
                ]
                st.dataframe(pd.DataFrame(rows), width="stretch")
            citations = extraction_data.get("citations") or []
            if citations:
                st.markdown("**Citations**")
                st.dataframe(
                    pd.DataFrame(normalize_records(citations)),
                    width="stretch",
                )
            references = extraction_data.get("references") or []
            if references:
                st.markdown("**Bibliography**")
                st.dataframe(
                    pd.DataFrame(normalize_records(references)),
                    width="stretch",
                )
            if resolution_data:
                st.markdown("**Resolution results**")
                st.dataframe(
                    pd.DataFrame(normalize_records(resolution_data)),
                    width="stretch",
                )

        st.markdown("</div>", unsafe_allow_html=True)


def draw_main():
    st.title("Citation-Support Checker")
    draw_ingestion_panel()
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
            api_url = get_api_url()
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
            api_url = get_api_url()
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
    api_url = get_api_url()
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
        "base_url": st.session_state.get("api_base", ""),
    }
    requests.post(f"{api_url}/prebuild", json=payload)


def main():
    init_session_state()
    inject_workspace_styles(dense=bool(st.session_state.get(WORKSPACE_DENSE_MODE)))
    inject_judgment_styles()
    inject_evidence_review_styles()
    draw_workspace()


if __name__ == "__main__":
    main()
