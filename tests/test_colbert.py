"""Skip ColBERT integration checks unless optional deps are installed."""

import pytest


try:
    __import__("colbert")
except Exception:
    pytest.skip("colbert not usable in this environment", allow_module_level=True)
pytest.skip(
    "ColBERT integration requires local index + heavy deps", allow_module_level=True
)
