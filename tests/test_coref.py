"""Skip coreference integration checks unless optional deps are installed."""

import pytest


pytest.importorskip("fastcoref")
pytest.skip("fastcoref integration requires model downloads", allow_module_level=True)
