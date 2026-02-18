from __future__ import annotations

import os

import pytest


def main() -> int:
    # Running pytest in-process can trigger native teardown crashes in some
    # environments (observed as: "terminate called without an active exception"
    # after the suite completes). Use os._exit to skip interpreter teardown.
    code = int(pytest.main(["-q"]))
    os._exit(code)


if __name__ == "__main__":
    main()
