from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
import os
from pathlib import Path
import subprocess
import sys
import requests

_LOCAL_NLI_PATHS = {
    "roberta-large-mnli": "roberta-large-mnli",
    "facebook/bart-large-mnli": "facebook/bart-large-mnli",
    # add other model names and paths as needed
}

_LOCAL_NLI_MODELS: Dict[str, Any] = {}


def get_local_nli_pipeline(model_name: str, batch_size: int = 32):
    tokenizer = AutoTokenizer.from_pretrained(_LOCAL_NLI_PATHS[model_name], use_fast=False)
    model     = AutoModelForSequenceClassification.from_pretrained(_LOCAL_NLI_PATHS[model_name])
    return hf_pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,              # CPU
        batch_size=batch_size,  # batch many inferences at once
    )

def assess(model_name: str, premise: str, hypothesis: str):
    if model_name in _LOCAL_NLI_PATHS:
        classifier = get_local_nli_pipeline(model_name)
        result = classifier(f"{premise} </s></s> {hypothesis}")
        return result
    # existing logic for other models
    # ...

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Citation checking CLI")
    parser.add_argument("--folder", help="Path to folder containing CSV and TEI.XML files", default=os.getenv("DATA_FOLDER", "."))
    parser.add_argument("--api_key", help="Blablador API key", default=os.getenv("BLABLADOR_API_KEY", ""))
    parser.add_argument("--api_base", help="Blablador API base URL", default=os.getenv("BLABLADOR_BASE", None))
    args = parser.parse_args()

    args.csv = str(Path(args.folder) / "short.csv")  # or whichever default filename you're expecting
    os.environ["SOURCE_DIR"] = str(Path(args.folder))

    if args.api_key:
        os.environ["BLABLADOR_API_KEY"] = args.api_key
    if args.api_base:
        os.environ["BLABLADOR_BASE"] = args.api_base

    # Prebuild is now triggered later from UI after user uploads folder
    # Preprocess citations

    # Launch backend and frontend
    backend_cmd = [
        "uvicorn", "backend.main:app",
        "--host", "localhost",
        "--port", "8000",
        "--reload",
        "--log-level", "debug",
    ]
    # pass arguments to ui.py _after_ the `--` separator so Streamlit doesn't
    # try to parse them.
    frontend_cmd = ["streamlit", "run", "frontend/ui.py"]

    def backend_ready(proc: subprocess.Popen) -> bool:
        """
        Poll the backend health‑check while also detecting early crashes.
        Returns True when http://localhost:8000/docs is reachable.
        """
        import requests, time
        for _ in range(30):
            if proc.poll() is not None:
                print("❌ Backend process exited prematurely.")
                return False
            try:
                if requests.get("http://localhost:8000/docs", timeout=1).status_code == 200:
                    return True
            except requests.exceptions.ConnectionError:
                print("⏳ Waiting for backend to start...")
            time.sleep(1)
        return False

    backend_proc = None
    frontend_proc = None

    try:
        # show backend logs live so the user can see errors
        backend_proc = subprocess.Popen(backend_cmd)
        if not backend_ready(backend_proc):
            print("❌ Backend failed to start. See error above.")
            backend_proc.terminate()
            backend_proc.wait()
            sys.exit(1)

        frontend_proc = subprocess.Popen(frontend_cmd)
        print("🚀 Backend and frontend started. Press Ctrl+C to stop.")
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        if backend_proc and backend_proc.poll() is None:
            backend_proc.terminate()
        if frontend_proc and frontend_proc.poll() is None:
            frontend_proc.terminate()
        if backend_proc:
            backend_proc.wait()
        if frontend_proc:
            frontend_proc.wait()