"""Skip ColBERT integration checks unless optional deps are installed."""

import pytest


pytest.importorskip("colbert")
pytest.skip(
    "ColBERT integration requires local index + heavy deps", allow_module_level=True
)
