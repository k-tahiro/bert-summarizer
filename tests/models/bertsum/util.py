import os

import pytest

skip_on_ga = pytest.mark.skipif(
    os.getenv("TEST_ENVIRONMENT") == "GitHub Actions",
    reason="Skip unittest to save memory",
)
