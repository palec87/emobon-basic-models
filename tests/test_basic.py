"""Basic tests for emobon-models package."""

import emobon_models


def test_package_has_version() -> None:
    """Test that the package has a version attribute."""
    assert hasattr(emobon_models, "__version__")
