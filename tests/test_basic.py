"""Basic tests for mgnify-methods package."""

import mgnify_methods


def test_hello() -> None:
    """Test the hello function."""
    result = mgnify_methods.hello()
    assert isinstance(result, str)
    assert "Hello from mgnify-methods!" == result


def test_package_has_version() -> None:
    """Test that the package has a version attribute."""
    assert hasattr(mgnify_methods, "__version__")
