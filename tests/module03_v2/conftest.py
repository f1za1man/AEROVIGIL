"""Shared helper for the Module 03 v2 tests: missing outputs FAIL, they do not skip.

Set AEROVIGIL_ALLOW_MISSING_DATA=1 to skip deliberately (for example on a machine
that does not hold the dataset).
"""

import os

import pytest


@pytest.fixture
def require():
    def _require(path):
        if path.exists():
            return path
        if os.environ.get("AEROVIGIL_ALLOW_MISSING_DATA") == "1":
            pytest.skip(f"missing {path} (AEROVIGIL_ALLOW_MISSING_DATA=1)")
        pytest.fail(f"Missing output {path}. Run the Module 03 v2 pipeline first.")

    return _require
