"""Skip coreference integration checks unless optional deps are installed."""

import pytest


try:
    __import__("fastcoref")
except Exception:
    pytest.skip("fastcoref not usable in this environment", allow_module_level=True)
pytest.skip("fastcoref integration requires model downloads", allow_module_level=True)
