import sys, pathlib, os
import tempfile
import streamlit as st
import requests
import re
from pathlib import Path


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent   # …/blablador_python
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import utils
from backend.bl_client import BlabladorClient   # new central wrapper

def get_responsive_models():
    """Fetch available Blablador models and return those that respond successfully."""
    api_key = st.session_state.get("bl_api_key", "")
    base_url = st.session_state.get("bl_base_url", "")
    if not api_key or not base_url:
        return []
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/v1/models",
                            headers={"Authorization": f"Bearer {api_key}"},
                            timeout=10)
        resp.raise_for_status()
        model_ids = [m["id"] for m in resp.json().get("data", [])]
    except Exception as e:
        st.error(f"Error fetching model list: {e}")
        return []
    responsive = []
    for m in model_ids:
        try:
            test = requests.post(
                f"{base_url.rstrip('/')}/v1/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": m, "prompt": "Test", "max_tokens": 1},
                timeout=5,
            )
            if test.status_code == 200:
                responsive.append(m)
        except:
            continue
    return responsive

  # ---------- default Session-State values ----------
_defaults = {
      "embed_model":    "alias-embeddings",
      "max_chunks":     256,
      "faiss_min_score":0.20,
      "max_sentences": 250,
}
for _k, _v in _defaults.items():
      st.session_state.setdefault(_k, _v)

# ensure our model keys exist in session_state
st.session_state.setdefault("selected_model", None)
st.session_state.setdefault("nli_model", None)

def reset_segmentation():
    # wipe any previous “done-…” flags
    for k in list(st.session_state.keys()):
        if k.startswith("done-"):
            del st.session_state[k]
    st.session_state.started = False
    st.session_state.seg_requested = False
    st.session_state.pop("seg_cache", None)
    # force a fresh FAISS build next time
    st.session_state.pop("faiss_started", None)


SEGMENT_PROMPT_TEMPLATE = """You are an expert at breaking sentences into standalone proposition segments.
For example, given a sentence A and B cause 1 and 2 generate 4 segments:
    A causes 1
    A causes 2
    B causes 1
    B causes 2

Keep modifiers with nouns. For example
immersion in cold water or snow causes hypothermia 
becomes
    cold water causes hypothermia
    snow causes hypothermia

List each segment on its own line, numbered {row_idx}a, {row_idx}b, etc., continuing alphabetically. 

Sentence:
{sentence}

Segments:
"""

# Map friendly aliases to actual Blablador model names
ALIAS_TO_MODEL = {
    "alias-large":     "alias-large",
    "alias-llama3-huge": "alias-llama3-huge",
    "alias-embeddings":  "alias-embeddings",
}
DEFAULT_ALIAS = "alias-large"


# --------------------------------------------------------------------
if "started" not in st.session_state:
    st.session_state.started = False
if "seg_requested" not in st.session_state:
    st.session_state.seg_requested = False

if 'max_sentences' not in st.session_state:
    st.session_state['max_sentences'] = 5

SEG_RE = re.compile(r"^\s*\d+[a-z]\.", re.I)          # e.g. 2a.

def seg_via_llm(sentence: str, row_idx: int, model: str) -> list[str]:
    prompt = SEGMENT_PROMPT_TEMPLATE.format(row_idx=row_idx, sentence=sentence)
    actual_model = model  # use the model string as chosen in the UI

    client = BlabladorClient(
        api_key=st.session_state.get("bl_api_key", ""),
        base_url=st.session_state.get("bl_base_url", "")
    )
    try:
        text = client.completion(
            prompt,
            model=actual_model,
            temperature=0,
            max_tokens=256
        )
    except Exception as e:
        st.error(f"Blablador API error: {e}")
        return []

    if actual_model == DEFAULT_ALIAS and "error" in text.lower():
        return []
    lines = [ln.strip() for ln in text.splitlines()]
    # segment-extraction logic unchanged below...
    segments = [ln for ln in lines if SEG_RE.match(ln)]
    if not segments:
        relaxed = [ln for ln in lines if f"{row_idx}a" in ln.lower()]
        segments = [ln.lstrip("-• ").strip() for ln in relaxed]
    if not segments:
        bullets = [ln for ln in lines if ln.lstrip().startswith(("-", "•"))]
        cleaned = [re.sub(r"^[-•\s]+", "", ln) for ln in bullets]
        segments = [f"{row_idx}{chr(97+i)} {txt}" for i, txt in enumerate(cleaned)]
    if not segments:
        segments = [f"{row_idx}a <MODEL RETURNED NO SEGMENTS>"]
    # Stop on duplicate segment IDs to avoid echoing example bullets
    unique_segments = []
    seen = set()
    for seg in segments:
        label = seg.split(maxsplit=1)[0]  # e.g., "0a."
        if label in seen:
            break
        seen.add(label)
        unique_segments.append(seg)
    segments = unique_segments
    return segments

def handle_upload():
    """Load uploaded files into a temp folder and reset segmentation on new upload."""
    uploaded = st.session_state.get("uploaded_files", [])
    if uploaded:
        tmpdir = tempfile.mkdtemp()
        for f in uploaded:
            with open(os.path.join(tmpdir, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.session_state["data_dir"] = tmpdir
        st.session_state["results"] = {}  # clear any previous results
        # clear done flags
        for key in list(st.session_state.keys()):
            if key.startswith("done-"):
                del st.session_state[key]
        reset_segmentation()
        st.success(f"Loaded {len(uploaded)} files")

# ---------- Streamlit app ----------
def main():
    st.session_state.setdefault("seg_cache", {})
    if "results" not in st.session_state:
        st.session_state["results"] = {}
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = chat_aliases[0]


    folder_path = st.session_state.get("data_dir", "")

    with st.sidebar:
        st.header("Upload your data")
        uploaded = st.file_uploader(
            "Upload your CSV and TEI XML files",
            type=["csv", "xml"],
            accept_multiple_files=True,
            key="uploaded_files",
            on_change=handle_upload
        )
        # In case the app reloads and uploaded_files persists, but data_dir was lost:
        if uploaded and "data_dir" not in st.session_state:
            handle_upload()

        st.header("Settings")
        # Always ensure API key and base URL fields are present and session_state is set
        api_key = st.text_input("Blablador API Key", value=os.getenv("BLABLADOR_API_KEY", ""), help="Your Blablador API key")
        st.session_state["api_key"] = api_key
        st.text_input("API Key", key="api_key")
        st.session_state["bl_api_key"] = api_key
        base_url = st.text_input("Blablador Base URL", value=os.getenv("BLABLADOR_BASE", ""), help="e.g. https://api.helmholtz-blablador.fz-juelich.de")
        st.session_state["base_url"] = base_url.rstrip("/")
        st.text_input("Base URL", key="base_url")
        st.session_state["bl_base_url"] = base_url.rstrip("/")
        # FastAPI backend URL
        backend_url = st.text_input(
            "Citation API URL",
            value=os.getenv("BACKEND_URL", "http://localhost:8000"),
            help="Your FastAPI backend URL"
        )
        st.session_state["api_url"] = backend_url.rstrip("/")
        # Dynamic LLM model selection
        if "available_models" not in st.session_state:
            with st.spinner("Fetching responsive Blablador models..."):
                st.session_state["available_models"] = get_responsive_models()
        chat_models = st.session_state.get("available_models") or ["alias-llama3-huge"]
        # Safely pick an index
        selected_lm = st.session_state.get("selected_model")
        if selected_lm not in chat_models:
            selected_lm = chat_models[0]
        default_index = chat_models.index(selected_lm)
        st.selectbox(
            "Select LLM model",
            chat_models,
            index=default_index,
            key="selected_model",
            on_change=reset_segmentation,
            help="Supported Blablador chat models (fetched dynamically)"
        )

        # Embedding model (Blablador only)
        st.selectbox(
            "Select embedding model",
            ["alias-embeddings"],
            index=0,
            key="embed_model"
        )

        st.number_input(
            "Max initial sentences (FAISS cap)",
            min_value=1, max_value=1000, value=250, step=1,
            key="max_sentences",
            on_change=reset_segmentation
        )

        st.slider(
            "FAISS min similarity",
            0.0, 1.0, 0.2, 0.01,
            key="faiss_min_score",
            on_change=reset_segmentation
        )

        # NLI model selection (pick one of the built-in checkpoints or enter your own)
        nli_choices = [
            "cross-encoder/nli-deberta-v3-base",
            "BlackBeenie/nli-deberta-v3-large",
            "amoux/scibert_nli_squad",
            "Custom..."
        ]
        selected = st.selectbox(
            "Select local NLI model",
            nli_choices,
            index=0,
            key="nli_model_choice",
            help="Pick a built-in model or choose Custom to paste any HF repo path"
        )
        if selected == "Custom...":
            custom = st.text_input(
                "Custom NLI model path (HF format)",
                value=st.session_state.get("nli_model", ""),
                key="nli_model_custom",
                help="e.g. my-org/my-fine-tuned-nli-model"
            ).strip()
            st.session_state["nli_model"] = custom or None
        else:
            st.session_state["nli_model"] = selected

        # Now show the NLI confidence threshold slider
        if "nli_threshold" not in st.session_state:
            st.session_state["nli_threshold"] = 0.5
        st.slider(
            "NLI confidence threshold",
            0.0, 1.0, 0.5, 0.01,
            key="nli_threshold",
            on_change=reset_segmentation
        )

        if st.button("Start segmentation"):
            if "data_dir" in st.session_state:
                st.session_state.started = True
                st.session_state.seg_requested = True

    if not st.session_state.started:
        st.info("Configure settings in the sidebar, then click Start segmentation to begin.")
        return

    st.title("Citation-Support Checker")
    import glob
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        st.error("No CSV file found in the folder.")
        return
    df = utils.read_csv(csv_files[0])
    st.write("Loaded rows:", len(df))
    st.write("Folder path:", folder_path)

    # ---- drop blank or NaN sentences ---------------------------------
    # ensure everything is a string first, then strip out empty lines
    mask = (
        df["tei_sentence"]
        .fillna("")      # turn NaN → ""
        .astype(str)     # ensure dtype=object with strings
        .str.strip()     # safe to call .str now
        .ne("")          # keep non-empty strings
    )
    df   = df[mask]                       # keep only rows that have text
    # ------------------------------------------------------------------

    # Perform segmentation if requested
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
        # 2) Display all rows with editable segments
        for idx, row in df.iterrows():
            st.markdown("---")
            st.write(f"**Row {idx}**  \n{row['tei_sentence']}")
            seg_text = st.text_area(
                f"Candidate segments (edit if needed) - Row {idx}",
                "\n".join(st.session_state.get("seg_cache", {}).get(idx, [])),
                key=f"ta-{idx}",
                height=120,
            )
            st.session_state[f"edited-{idx}"] = [ln.strip() for ln in seg_text.splitlines() if ln.strip()]

        # Single submit button at bottom
        if st.button("Submit All Rows for Processing"):
            api_url = st.session_state.get("api_url", "http://localhost:8000")
            with st.spinner("Running citation-support for all rows…"):
                for idx, row in df.iterrows():
                    segments_list = []
                    for i, seg_text in enumerate(st.session_state.get(f"edited-{idx}", [])):
                        segments_list.append({
                            "segment_id": f"{idx}{chr(97 + i)}",
                            "claim": seg_text
                        })
                    payload = {
                        "folder": st.session_state["data_dir"],
                        "row_id": idx,
                        "citing_title": row.get("Cited Author", ""),
                        "citing_id": row.get("tei_target", ""),
                        "original_sentence": row["tei_sentence"],
                        "segments": segments_list,
                        "settings": {
                            "embed_model":    st.session_state["embed_model"],
                            "max_sentences":  st.session_state["max_sentences"],
                            "faiss_min_score": st.session_state["faiss_min_score"],
                            "nli_model":      st.session_state["nli_model"],
                            "llm_model":      st.session_state["selected_model"],
                            "nli_threshold":  st.session_state["nli_threshold"],
                            "api_key":        st.session_state["api_key"],
                            "base_url":       st.session_state["base_url"],
                        },
                    }
                    resp = requests.post(f"{api_url}/segment", json=payload)
                    try:
                        st.session_state["results"][idx] = resp.json()
                    except:
                        st.session_state["results"][idx] = resp.text

            # Display results for all rows
            for idx in df.index:
                st.markdown(f"### Results for Row {idx}")
                result = st.session_state["results"].get(idx)
                if isinstance(result, dict):
                    st.json(result)
                else:
                    st.write(result)

    if st.session_state.seg_requested and "faiss_started" not in st.session_state:
        # now kick off backend index build in background
        with st.spinner("Building FAISS index in background…"):
            api_url = st.session_state.get("api_url", "http://localhost:8000")
            try:
                body = {
                    "folder": str(st.session_state["data_dir"]),
                    "embed_model":   st.session_state["embed_model"],
                    "max_chunks":    int(st.session_state["max_sentences"]),
                    "faiss_min_score": float(st.session_state["faiss_min_score"]),
                    "api_key":  st.session_state["bl_api_key"],
                    "base_url": st.session_state["bl_base_url"],
                }
                prebuild_resp = requests.post(
                    f"{api_url}/prebuild",
                    json=body
                )
                prebuild_resp.raise_for_status()
                st.success("Backend indexing completed.")
            except requests.HTTPError as e:
                # Show detailed server error response
                try:
                    err_body = e.response.json()
                except Exception:
                    err_body = e.response.text
                st.error(f"Failed to start backend indexing (HTTP {e.response.status_code}): {e}\nResponse body:\n{err_body}")
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
            mime="application/json"
        )

main()