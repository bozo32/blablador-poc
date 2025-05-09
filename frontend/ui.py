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
    """Clear any existing segmentation so we can rerun with new settings."""
    st.session_state.started = False
    st.session_state.seg_requested = False
    if "seg_cache" in st.session_state:
        del st.session_state.seg_cache

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
    actual_model = ALIAS_TO_MODEL.get(model, model)
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
        st.warning(
            f"Blablador API error ({e}), falling back to '{DEFAULT_ALIAS}'."
        )
        actual_model = ALIAS_TO_MODEL[DEFAULT_ALIAS]
        try:
            text = client.completion(
                prompt,
                model=actual_model,
                temperature=0,
                max_tokens=256

            )
        except Exception as e2:
            st.error(f"Blablador API error after fallback: {e2}")
            text = ""
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
    if "results" not in st.session_state:
        st.session_state["results"] = {}

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

        st.header("Settings")
        api_key = st.text_input("Blablador API Key", value=os.getenv("BLABLADOR_API_KEY", ""), help="Your Blablador API key")
        base_url = st.text_input("Blablador Base URL", value=os.getenv("BLABLADOR_BASE", ""), help="e.g. https://api.helmholtz-blablador.fz-juelich.de")
        st.session_state["bl_api_key"] = api_key
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
            key="max_sentences"
        )

        st.slider(
            "FAISS min similarity",
            0.0, 1.0, 0.2, 0.01,
            key="faiss_min_score"
        )

        nli_choices = ["cross-encoder/nli-deberta-v3-base","BlackBeenie/nli-deberta-v3-large","amoux/scibert_nli_squad"]
        nli_model = st.selectbox("Select local NLI model (runs via HuggingFace transformers)", nli_choices, index=0)

        nli_threshold = st.slider("NLI confidence threshold", 0.0, 1.0, 0.5, 0.01)

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
    mask = df["tei_sentence"].notna() & df["tei_sentence"].str.strip().ne("")
    df   = df[mask]                       # keep only rows that have text
    # ------------------------------------------------------------------

    if st.session_state.seg_requested:
        if "seg_cache" not in st.session_state:
            st.session_state.seg_cache = {}
            with st.spinner("Segmenting sentences..."):
                for idx, row in df.iterrows():
                    if idx not in st.session_state.seg_cache:
                        st.session_state.seg_cache[idx] = seg_via_llm(
                            row["tei_sentence"], idx, model=st.session_state["selected_model"]
                        )

        for idx, row in df.iterrows():
            st.markdown("---")
            st.write(f"**Row {idx}**  \n{row['tei_sentence']}")

            seg_text = st.text_area(
                "Candidate segments (edit if needed)",
                "\n".join(st.session_state.seg_cache.get(idx, [])),
                key=f"ta-{idx}",
                height=120,
            )

            done_flag = f"done-{idx}"
            # placeholder for spinner/result for this row
            result_placeholder = st.empty()

            if st.button("Submit", key=f"submit-{idx}", disabled=st.session_state.get(done_flag, False)):
                # use a global spinner to show progress during the POST
                api_url = st.session_state.get("api_url", "http://localhost:8000")
                with st.spinner("Running segmentation & checks…"):
                    # collect the edited segments from the text area
                    segments_list = [
                        ln.strip() for ln in seg_text.splitlines() if ln.strip()
                    ]
                    # Build the payload to match the backend's SegmentRequest model
                    payload = {
                        "folder": folder_path,
                        "row": idx,
                        "citing_title": row.get("Cited Author", ""),
                        "citing_id":    row.get("tei_target", ""),
                        "original_sentence": row["tei_sentence"],
                        "segments": segments_list,
                        "settings": {
                            "llm_model":  st.session_state["selected_model"],
                            "nli_model":  st.session_state["nli_model"],
                            "data_dir":   st.session_state.get("data_dir")
                        }
                    }
                    # POST to the /segment endpoint
                    resp = requests.post(f"{api_url}/segment", json=payload)
                st.session_state[done_flag] = True
                # parse and store result
                try:
                    result = resp.json()
                except Exception:
                    result = resp.text
                st.session_state["results"][idx] = result
                # display result
                result_placeholder.success("Done")
                if isinstance(result, dict):
                    result_placeholder.json(result)
                else:
                    result_placeholder.write(result)
            elif st.session_state.get(done_flag, False):
                # show previously stored result
                stored = st.session_state["results"].get(idx)
                if stored is not None:
                    if isinstance(stored, dict):
                        result_placeholder.json(stored)
                    else:
                        result_placeholder.write(stored)

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
                    st.success("Backend indexing started.")
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