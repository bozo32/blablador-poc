from __future__ import annotations

from pathlib import Path
from typing import Union

import requests

from backend.settings import settings


class GrobidError(RuntimeError):
    pass


def extract_tei(pdf_path: Union[str, Path]) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    url = f"{settings.GROBID_URL.rstrip('/')}/api/processFulltextDocument"
    with path.open("rb") as pdf_file:
        files = {"input": (path.name, pdf_file, "application/pdf")}
        response = requests.post(
            url,
            files=files,
            data={
                "consolidateCitations": "1",
                "consolidateHeader": "1",
            },
            timeout=settings.GROBID_TIMEOUT,
        )

    if response.status_code != 200:
        message = response.text.strip()
        raise GrobidError(
            "GROBID extraction failed " f"({response.status_code}): {message}"
        )

    return response.text
