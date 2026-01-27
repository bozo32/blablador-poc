# frontend/ui.py

# === Imports ===
import json
import os
import pathlib
import re
import sys
import tempfile

import graphviz
import pandas as pd
import requests
import streamlit as st

from frontend import claim_queue

st.set_page_config(page_title="Citation-Support Checker", layout="wide")

# Add project root to sys.path so `backend` is importable
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import utils
from backend.bl_client import BlabladorClient
from backend.model_cache import add_model, get_models
from backend.settings import AppSettings
from backend.utils import list_local_models
from frontend.ingestion_api import (
    get_citation_context,
    get_citation_graph,
    get_document,
    list_documents,
    submit_resolution_choice,
    trigger_extraction,
    trigger_resolution,
    upload_pdf,
)
from typing import List, Optional


# Third-party


# Local application

# === Settings & State Initialization ===
settings = AppSettings()


def init_session_state():
    """Initialize Streamlit session state keys from settings."""
    defaults = {
        "api_url": settings.BACKEND_URL,
        "api_key": settings.API_KEY,
        "api_base": settings.API_BASE,
        "embed_model": settings.EMBED_MODEL,
        "max_sentences": settings.MAX_SENTENCES,
        "faiss_min_score": settings.FAISS_MIN_SCORE,
        "reranker_model": settings.RERANKER_MODEL,
        "reranker_top_k": settings.RERANKER_TOP_K,
        "nli_model": settings.NLI_MODEL,
        "nli_threshold": settings.NLI_THRESHOLD,
        "selected_model": None,
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
        "auto_extract_on_upload": True,
        "auto_resolve_on_upload": True,
        "citation_debug": False,
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
                    st.error(f"Resolution failed: {exc}")
                    return


def stringify_value(value: object) -> str | int | float | bool | None:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return value


def normalize_records(records: list[dict]) -> list[dict]:
    return [
        {key: stringify_value(val) for key, val in record.items()} for record in records
    ]


def show_api_error(message: str) -> None:
    if hasattr(st, "toast"):
        st.toast(message)
    else:
        st.error(message)


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
            color: #1f2933;
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
        .citation-divider {
            height: 1px;
            background: #e6e6e6;
            margin: 12px 0 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["citation_styles_loaded"] = True


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
    st.subheader("PDF Ingestion")
    docs = st.session_state.get("ingested_docs")
    if docs is None:
        docs = refresh_ingested_docs(show_error=False)
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
                    st.error(f"Resolution failed: {exc}")
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info(
                "No metadata available yet. Run extraction to populate this section."
            )

    with st.expander("Citations"):
        citations = extraction_data.get("citations") or []
        if citations:
            st.dataframe(
                pd.DataFrame(normalize_records(citations)),
                use_container_width=True,
            )
        else:
            st.info("No citations extracted yet.")

    with st.expander("Bibliography"):
        references = extraction_data.get("references") or []
        if references:
            st.dataframe(
                pd.DataFrame(normalize_records(references)),
                use_container_width=True,
            )
        else:
            st.info("No bibliography entries extracted yet.")

    with st.expander("Resolution Results"):
        if resolution_data:
            st.dataframe(
                pd.DataFrame(normalize_records(resolution_data)),
                use_container_width=True,
            )
        else:
            st.info("No resolved references yet. Run resolution after extraction.")

    st.divider()
    st.subheader("Citation Context")

    citations = extraction_data.get("citations") or []
    if not citations:
        st.info("No citations extracted yet.")
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
    left_col, right_col = st.columns([5, 7])

    with left_col:
        st.markdown("#### Callouts")
        st.caption("Click the badges below each sentence to open context.")
        grouped: dict[str, list[tuple[int, dict]]] = {}
        for idx, citation in enumerate(citations):
            sentence = (citation.get("sentence") or "Sentence unavailable").strip()
            grouped.setdefault(sentence, []).append((idx, citation))

        for sentence, items in grouped.items():
            callouts = [
                format_callout(item[1].get("callout") or "citation") for item in items
            ]
            rendered_sentence, unmatched = render_sentence_with_callouts(
                sentence, callouts
            )
            st.markdown(
                f'<div class="citation-sentence">{rendered_sentence}</div>',
                unsafe_allow_html=True,
            )
            if st.session_state.get("citation_debug"):
                debug_rows = [
                    {
                        "callout": item[1].get("callout"),
                        "target_id": item[1].get("target_id"),
                    }
                    for item in items
                ]
                st.caption(f"Raw sentence: {sentence}")
                st.json(debug_rows)
            if unmatched:
                overflow = " ".join(
                    f'<span class="citation-chip">{item.get("display")}</span>'
                    for item in unmatched
                )
                st.markdown(
                    f'<div style="margin-top:6px;">{overflow}</div>',
                    unsafe_allow_html=True,
                )
            max_per_row = 4
            for offset in range(0, len(items), max_per_row):
                row_items = items[offset : offset + max_per_row]
                cols = st.columns(len(row_items))
                for col, (citation_index, citation) in zip(cols, row_items):
                    callout = citation.get("callout") or "citation"
                    display_callout = format_callout(callout)["display"]
                    target_id = normalize_target_id(citation.get("target_id"))
                    selected = (
                        st.session_state.get("citation_selected_index")
                        == citation_index
                        and st.session_state.get("citation_selected_target")
                        == target_id
                    )
                    label = f"{display_callout} ✓" if selected else display_callout
                    if col.button(
                        label,
                        key=f"citation-callout-{citation_index}",
                        type="secondary",
                    ):
                        select_citation(citation_index, target_id)
            st.markdown('<div class="citation-divider"></div>', unsafe_allow_html=True)

    with right_col:
        st.markdown("#### Context")
        selected_index = st.session_state.get("citation_selected_index")
        selected_target = normalize_target_id(
            st.session_state.get("citation_selected_target")
        )
        if selected_index is None:
            st.info("Select a citation callout to view its context.")
            st.button(
                "Retrieval instructions",
                key="retrieval-disabled",
                disabled=True,
                help="Select a citation to load retrieval guidance",
            )
        else:
            context_request = {
                "api_url": st.session_state.get("api_url", "http://localhost:8000"),
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

            context = st.session_state.get("citation_context")
            if context:
                reference = context.get("reference") or {}
                reference_id = reference.get("id") or selected_target
                doc_identifier = context.get("document_id") or doc_id
                show_retrieval = st.button(
                    "Retrieval instructions",
                    key="retrieval-instructions-button",
                    help="Open canonical citation, DOI links, and fallback steps",
                )
                st.markdown(
                    highlight_callout(context.get("previous_sentence"), None),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    highlight_callout(context.get("sentence"), context.get("callout")),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    highlight_callout(context.get("next_sentence"), None),
                    unsafe_allow_html=True,
                )
                follow_label = (
                    "Hide citation details"
                    if st.session_state.get("citation_follow_open")
                    else "Follow citation"
                )
                if st.button(follow_label, key="citation-follow-toggle"):
                    st.session_state["citation_follow_open"] = not st.session_state.get(
                        "citation_follow_open", False
                    )
                if st.session_state.get("citation_follow_open"):
                    reference = context.get("reference")
                    resolution = context.get("resolution")
                    if not reference and not resolution:
                        st.info("Citation metadata unavailable.")
                    summary = format_reference_summary(
                        reference or {}, resolution or {}
                    )
                    if summary:
                        st.markdown("**Reference summary**")
                        st.markdown(summary)
                    else:
                        st.caption("Reference summary unavailable.")
                    if not resolution:
                        st.caption("Resolution missing — run Resolve References.")
                    else:
                        status = resolution.get("status")
                        if status in ("mismatch", "needs_review"):
                            reference_id = resolution.get("reference_id")
                            st.caption("Resolution needs review.")
                            if resolution.get("mismatch_reason"):
                                st.caption(
                                    f"Reason: {resolution.get('mismatch_reason')}"
                                )
                            (
                                citing_summary,
                                has_consolidated,
                            ) = build_citing_bibliography_summary(
                                reference or {}, resolution or {}
                            )
                            if citing_summary:
                                st.markdown("**Citing bibliography**")
                                st.markdown(citing_summary)
                            else:
                                st.caption("Citing bibliography unavailable.")
                            if not has_consolidated:
                                st.warning(
                                    "Consolidated metadata missing for this reference."
                                )
                            if resolution.get("openalex"):
                                st.caption(
                                    "OpenAlex may resolve to the citing article; "
                                    "review before selecting."
                                )
                            candidate_options = build_resolution_candidates(resolution)
                            if reference_id and candidate_options:
                                option_keys = list(candidate_options.keys())
                                default_key = resolution.get("selected_source")
                                if default_key not in option_keys:
                                    default_key = option_keys[0]
                                selected_source = st.selectbox(
                                    "Select source",
                                    option_keys,
                                    index=option_keys.index(default_key),
                                    format_func=lambda key: candidate_options.get(
                                        key, key
                                    ),
                                    key=f"resolution-select-{reference_id}",
                                )
                                if st.button(
                                    "Apply selection",
                                    key=f"resolution-apply-{reference_id}",
                                ):
                                    api_url = st.session_state.get(
                                        "api_url", "http://localhost:8000"
                                    )
                                    try:
                                        submit_resolution_choice(
                                            api_url,
                                            doc_id,
                                            reference_id,
                                            selected_source,
                                        )
                                        load_selected_document(show_error=False)
                                        load_citation_context(context_request)
                                        st.success("Resolution updated.")
                                    except RuntimeError as exc:
                                        st.error(f"Failed to update resolution: {exc}")
                            else:
                                st.caption("No resolution candidates available.")
                    with st.expander("Show raw metadata"):
                        if reference:
                            st.markdown("**Bibliography entry**")
                            st.json(reference)
                        else:
                            st.caption("Bibliography entry unavailable.")
                        if resolution:
                            st.markdown("**Resolved metadata**")
                            st.json(resolution)
                        else:
                            st.caption("Resolved metadata unavailable.")
                if show_retrieval:
                    with st.expander("Retrieval instructions", expanded=True):
                        if reference_id:
                            claim_queue.render_retrieval_instructions(
                                api_url=context_request["api_url"],
                                doc_id=doc_identifier,
                                reference_id=reference_id,
                            )
                        else:
                            st.warning(
                                "Reference metadata missing — run resolution to fetch"
                                " dossiers."
                            )
            else:
                st.info("Context unavailable yet. Try running extraction/resolution.")
                fallback = None
                if selected_index is not None and 0 <= selected_index < len(citations):
                    fallback = citations[selected_index]
                if fallback:
                    fallback_sentence = fallback.get("sentence") or ""
                    callout = fallback.get("callout")
                    callouts = [format_callout(callout)] if callout else []
                    rendered, _ = render_sentence_with_callouts(
                        fallback_sentence, callouts
                    )
                    st.markdown(rendered, unsafe_allow_html=True)
                if st.session_state.get("citation_context_error"):
                    if st.button("Retry context", key="citation-context-retry"):
                        last_request = st.session_state.get(
                            "citation_last_context_request"
                        )
                        if last_request:
                            load_citation_context(last_request)

        st.divider()
        st.markdown("#### Citation Graph")
        if selected_target is None:
            st.info("Select a citation with a target ID to view the graph.")
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
            if selected_target:
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
                "api_url": st.session_state.get("api_url", "http://localhost:8000"),
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
                        "Check that reference resolution populated a DOI "
                        "or OpenAlex ID."
                    )
                if st.button("Retry graph", key="citation-graph-retry"):
                    last_request = st.session_state.get("citation_last_graph_request")
                    if last_request:
                        load_citation_graph(last_request)


def draw_main():
    st.title("Citation-Support Checker")
    draw_ingestion_panel()
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
