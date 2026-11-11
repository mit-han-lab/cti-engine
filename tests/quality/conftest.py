"""Fixtures the quality checks share."""

import pytest

from tests.support.architecture import Package


@pytest.fixture(scope="session")
def package() -> Package:
    """The engine's source tree, parsed once."""

    return Package.engine()
