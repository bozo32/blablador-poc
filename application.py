import os
os.environ["TRANSFORMERS_CACHE"] = str(os.path.expanduser("~/.cache/huggingface"))

from backend import utils
utils.set_sane_threads()

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict
from backend import utils

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline as hf_pipeline,
)

try:
    from transformers import AdamW
except ImportError:
    import torch
    import transformers
    transformers.AdamW = torch.optim.AdamW


def assess(model_name: str, premise: str, hypothesis: str):
    from backend.nli import get_nli_pipeline

    pipeline = get_nli_pipeline(model_name)
    result = pipeline(f"{premise} </s></s> {hypothesis}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Citation checking CLI")
    parser.add_argument(
        "--folder",
        help="Path to folder containing CSV and TEI.XML files",
        default=os.getenv("DATA_FOLDER", "."),
    )
    parser.add_argument(
        "--api_key", help="Blablador API key", default=os.getenv("API_KEY", "")
    )
    parser.add_argument(
        "--api_base", help="Blablador API base URL", default=os.getenv("API_BASE", None)
    )
    args = parser.parse_args()

    args.csv = str(
        Path(args.folder) / "short.csv"
    )  # or whichever default filename you're expecting
    os.environ["SOURCE_DIR"] = str(Path(args.folder))

    # Export unified env vars for frontend and backend
    os.environ["API_KEY"] = args.api_key
    if args.api_base:
        os.environ["API_BASE"] = args.api_base
    # Set backend URL for the UI to call
    os.environ.setdefault("BACKEND_URL", f"http://localhost:8000")

    # Prebuild is now triggered later from UI after user uploads folder
    # Preprocess citations

    # Launch backend, frontend, and ColBERT server
    backend_cmd = [
        "uvicorn",
        "backend.main:app",
        "--host",
        "localhost",
        "--port",
        "8000",
        "--reload",
        "--log-level",
        "debug",
    ]
    # --- Launch ColBERT in a *separate* conda env so we don't fight over
    #     torch / transformers versions.  Let users pick the env via
    #     COLBERT_CONDA_ENV; default = "colbert-server" (see environment.yml).
    colbert_env = os.environ.get("COLBERT_CONDA_ENV", "colbert-server")
    colbert_cmd = [
        "conda", "run", "-n", colbert_env,
        "uvicorn", "colbert_server.colbert:app",
        "--host", "localhost", "--port", "7001",
    ]
    # pass arguments to ui.py _after_ the `--` separator so Streamlit doesn't
    # try to parse them.
    frontend_cmd = [
        "streamlit",
        "run",
        "frontend/ui.py",
        "--server.fileWatcherType",
        "watchdog",
    ]

    def backend_ready(proc: subprocess.Popen) -> bool:
        """
        Poll the backend health‑check while also detecting early crashes.
        Returns True when http://localhost:8000/docs is reachable.
        """
        import time

        import requests

        for _ in range(30):
            if proc.poll() is not None:
                print("❌ Backend process exited prematurely.")
                return False
            try:
                if (
                    requests.get("http://localhost:8000/docs", timeout=1).status_code
                    == 200
                ):
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
