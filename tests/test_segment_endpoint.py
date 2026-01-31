"""Legacy debug script placeholder.

This file was previously a one-off local experiment and isn't a deterministic
unit test (it downloads models and references undefined variables).

We keep the filename for historical context but skip it in automated runs.
"""

import pytest


pytest.skip(
    "Non-deterministic local debug script (not a unit test)", allow_module_level=True
)
