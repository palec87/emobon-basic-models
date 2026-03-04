"""Basic tests for emobon-models package."""

import emobon_models


def test_hello() -> None:
    """Test the hello function."""
    result = emobon_models.hello()
    assert isinstance(result, str)
    assert "Hello from emobon-models!" == result


def test_package_has_version() -> None:
    """Test that the package has a version attribute."""
    assert hasattr(emobon_models, "__version__")
