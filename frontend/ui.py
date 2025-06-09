# frontend/ui.py

# === Imports ===
import os
import pathlib
import re
import sys
import tempfile

import pandas as pd
import requests
import streamlit as st

# Add project root to sys.path so `backend` is importable
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import utils
from backend.bl_client import BlabladorClient
from backend.model_cache import add_model, get_models
from backend.settings import Settings
from backend.utils import list_local_models


# Third-party


# Local application

# === Settings & State Initialization ===
settings = Settings()


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
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


# === Helpers ===


def model_selector(
    label: str, session_key: str, choices: list[str], allow_custom: bool = True
):
    """Generic dropdown + Custom... text input helper."""
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
SEGMENT_PROMPT_TEMPLATE = """You are an expert at breaking sentences into standalone proposition segments.
For example, given a sentence A and B cause 1 and 2 you generate 4 segments:
    A causes 1
    A causes 2
    B causes 1
    B causes 2
You only work with the words provided in the prompting sentence
For example, given the sentence A causes B, you generate:
    A causes B
You stop segmenting when you run out of words in the original sentence.
You always keep modifiers (adjectives or adverbs) with what they modify (nouns or verbs). For example
immersion in cold water or snow causes hypothermia
becomes
    cold water causes hypothermia
    snow causes hypothermia

Some citing sentences directly mention the source.
They may take variants on the form '(author name) found that A causes 1.'.
In such cases drop the direct mention of the source so that the sentence becomes:
    A causes 1

 **Do not output any explanation, commentary, or extra text. Only list the segments generated from the original sentence. Your output must end after the last segment.**

List each segment on its own line, numbered {row_idx}a, {row_idx}b, etc., continuing alphabetically.

Sentence:
{sentence}
Segments:
"""

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


# === UI Drawing ===


def draw_sidebar():
    init_session_state()
    with st.sidebar:
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


def draw_main():
    if not st.session_state.started:
        st.info("Configure settings then click Start segmentation.")
        return

    st.title("Citation-Support Checker")
    import glob

    folder = st.session_state.get("data_dir", "")
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        st.error("No CSV file found in the folder.")
        return
    df = utils.read_csv(csv_files[0])
    st.write("Loaded rows:", len(df))
    st.write("Folder path:", folder)

    # ==== Begin Streamlit form for segment evaluation UI ====
    if st.session_state.seg_requested:
        # 1) Populate seg_cache if empty
        if not st.session_state.get("seg_cache"):
            st.session_state["seg_cache"] = {}
            with st.spinner("Segmenting sentences…"):
                for idx, row in df.iterrows():
                    st.session_state["seg_cache"][idx] = seg_via_llm(
                        row["tei_sentence"],
                        idx,
                        st.session_state["selected_model"],
                    )

        # Now, use a Streamlit form with one expander per sentence/segment.
        with st.form("sentence_segment_evaluation_form"):
            for idx, row in df.iterrows():
                expanded = True if idx == df.index[0] else False
                with st.expander(
                    f"Row {idx}: {row['tei_sentence']}",
                    expanded=expanded,
                ):
                    seg_text = st.text_area(
                        f"Candidate segments (edit if needed) - Row {idx}",
                        "\n".join(st.session_state.get("seg_cache", {}).get(idx, [])),
                        key=f"ta-{idx}",
                        height=120,
                    )
                    st.session_state[f"edited-{idx}"] = [
                        ln.strip() for ln in seg_text.splitlines() if ln.strip()
                    ]
            submitted = st.form_submit_button("Submit all choices")

        # If form submitted, process all user inputs together
        if submitted:
            api_url = st.session_state.get("api_url", "http://localhost:8000")
            with st.spinner("Running citation-support for all rows…"):
                for idx, row in df.iterrows():
                    segments_list = []
                    for i, seg_text in enumerate(
                        st.session_state.get(f"edited-{idx}", [])
                    ):
                        segments_list.append(
                            {
                                "segment_id": f"{idx}{chr(97 + i)}",
                                "claim": seg_text,
                            }
                        )

                    # Coerce possible NaN → "" for the 'tei_target' field
                    tid = row.get("tei_target", "")
                    if pd.isna(tid):
                        tid = ""

                    payload = {
                        "folder": st.session_state["data_dir"],
                        "row_id": idx,
                        "citing_title": row.get("Cited Author", ""),
                        "citing_id": tid,
                        "original_sentence": row["tei_sentence"],
                        "segments": segments_list,
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
                        },
                    }
                    # Remove unused variable resp; just assign to results directly
                    try:
                        response = requests.post(f"{api_url}/segment", json=payload)
                        st.session_state["results"][idx] = response.json()
                    except BaseException:
                        st.session_state["results"][idx] = response.text

            # --- Results/Assessment UI ---
            # After the segmentation form
            if st.session_state.get("results"):
                # Show results/assessment forms for rows with results
                for idx in sorted(st.session_state["results"].keys()):
                    result = st.session_state["results"].get(idx)
                    if not (result and isinstance(result, dict)):
                        continue

                    st.markdown(f"### Results for Row {idx}")
                    original_sentence = result.get("original_sentence", "")
                    st.markdown(f"**Original sentence:** {original_sentence}")

                    # Per-row form for all segments in this row
                    with st.form(f"assessment_form_row_{idx}"):
                        segs = result.get("segments", [])
                        all_troll_pay_items = []
                        for seg_idx, seg in enumerate(segs):
                            seg_id = seg.get("segment_id", "")
                            claim = seg.get("claim", "")
                            exp_key = f"exp_{idx}_{seg_id}"

                            with st.expander(
                                f"Row {idx} Segment {seg_id}: {claim}",
                                expanded=(
                                    seg_idx == 0
                                    and not st.session_state.get(f"done_row_{idx}", False)
                                ),
                            ):
                                evidence = seg.get("evidence", [])
                                N = settings.NLI_CANDIDATES_SHOWN
                                support = sorted(
                                    [ev for ev in evidence if ev.get("label") == "entailment"],
                                    key=lambda ev: ev.get("score", 0),
                                    reverse=True
                                )[:N]
                                contradiction = sorted(
                                    [ev for ev in evidence if ev.get("label") == "contradiction"],
                                    key=lambda ev: ev.get("score", 0),
                                    reverse=True
                                )[:N]

                                st.write("**Supporting Evidence:**")
                                support_checked = []
                                for i, ev in enumerate(support):
                                    eid = ev["id"]
                                    text = ev.get("text", "")
                                    section_path = (
                                        ev.get("section_path")
                                        or ev.get("section_head")
                                        or ""
                                    )
                                    label = (
                                        f"**Section:** {section_path}\n{text}"
                                        if section_path
                                        else text
                                    )
                                    key = f"support_cb_{idx}_{seg_id}_{eid}"
                                    checked = st.session_state.get(key, False)
                                    cb = st.checkbox(label, value=checked, key=key)
                                    if cb:
                                        support_checked.append(eid)
                                seg["user_selected_support"] = support_checked

                                st.write("**Contradicting Evidence:**")
                                contra_checked = []
                                for i, ev in enumerate(contradiction):
                                    eid = ev["id"]
                                    text = ev.get("text", "")
                                    section_path = (
                                        ev.get("section_path")
                                        or ev.get("section_head")
                                        or ""
                                    )
                                    label = (
                                        f"**Section:** {section_path}\n{text}"
                                        if section_path
                                        else text
                                    )
                                    key = f"contradict_cb_{idx}_{seg_id}_{eid}"
                                    checked = st.session_state.get(key, False)
                                    cb = st.checkbox(label, value=checked, key=key)
                                    if cb:
                                        contra_checked.append(eid)
                                seg["user_selected_contradiction"] = contra_checked

                                # --- Collect troll pay ambiguous evidence (per segment) ---
                                troll_pay_items = []
                                for ev in evidence:
                                    all_scores = ev.get("all_scores", {})
                                    label_scores = sorted(all_scores.items(), key=lambda x: -x[1])
                                    for i, (label1, score1) in enumerate(label_scores):
                                        if label1 not in {"entailment", "contradiction"}:
                                            continue
                                        for j, (label2, score2) in enumerate(label_scores):
                                            if j == i:
                                                continue
                                            if label2 not in {"entailment", "contradiction"}:
                                                continue
                                            if abs(score1 - score2) <= settings.TROLL_PAY_MARGIN:
                                                troll_pay_items.append((seg_id, ev))
                                                break
                                        else:
                                            continue
                                        break
                                all_troll_pay_items.extend(troll_pay_items)

                        # After all segments, show one troll-pay expander per row (if any)
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
                            # Find the most ambiguous item (lowest margin between top 2 NLI scores)
                            def ambiguity(ev):
                                scores = sorted(ev.get("all_scores", {}).values(), reverse=True)
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
                                with st.expander("🧌 tax! Help us on this ambiguous case", expanded=True):
                                    st.write(
                                        "This candidate was difficult for the model to classify. Please help by labeling it:"
                                    )
                                    eid = ev["id"]
                                    text = ev.get("text", "")
                                    section_path = ev.get("section_path") or ev.get("section_head") or ""
                                    label = (
                                        f"**Row {idx} Segment {seg_id}**<br>**Section:** {section_path}<br>{text}"
                                        if section_path
                                        else text
                                    )
                                    radio_key = f"troll_radio_{idx}_{seg_id}_{eid}"
                                    prev = st.session_state.get(radio_key, "I dunno")
                                    choice = st.radio(
                                        label,
                                        options=["Entails", "Neutral", "Contradicts", "I dunno"],
                                        index=["Entails", "Neutral", "Contradicts", "I dunno"].index(prev)
                                            if prev in ["Entails", "Neutral", "Contradicts", "I dunno"] else 3,
                                        key=radio_key,
                                        horizontal=True,
                                    )
                                    # Assign to proper segment (optional)
                                    for seg in segs:
                                        if seg.get("segment_id", "") == seg_id:
                                            troll_selected = seg.get("user_selected_trollpay", {})
                                            troll_selected[eid] = choice
                                            seg["user_selected_trollpay"] = troll_selected

                        submitted = st.form_submit_button("Submit Assessment")
                        if submitted:
                            st.session_state[f"done_row_{idx}"] = True
                            st.success("Assessment saved!")

            # After the form, show raw JSON per row (unchanged)
            for idx in sorted(st.session_state["results"].keys()):
                result = st.session_state["results"].get(idx)
                if result:
                    st.markdown("**Raw Result JSON (includes your selections):**")
                    st.json(result)
        else:
            # Only show *one* instructional message if no results yet
            st.info(
                "Click 'Submit all choices' above to search for candidate cited sentences."
            )
    # ==== End Streamlit form for segment evaluation UI ====

    if st.session_state.seg_requested and "faiss_started" not in st.session_state:
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
                    f"Failed to start backend indexing (HTTP {e.response.status_code}): {e}\nResponse body:\n{err_body}"
                )
            except Exception as e:
                st.error(f"Failed to start backend indexing: {e}")
        st.session_state["faiss_started"] = True

    # ------------------------
    # offer download once all rows are done
    if all(st.session_state.get(f"done-{i}", False) for i in df.index):
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
