import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def isolated_environment():
    """ConnectionSettings.save() exports its values, which must not leak between tests."""
    with patch.dict(os.environ):
        yield
