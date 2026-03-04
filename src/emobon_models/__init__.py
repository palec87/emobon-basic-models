"""Utilities for EMOBON abundance modeling workflows."""

from emobon_models.modeling_config import ModelingConfig
from emobon_models.modeling_config import modeling_config_from_analysis
from emobon_models.modeling_runner import run_group_loocv_with_mlflow

__version__ = "0.1.0"


def hello() -> str:
    """Return a greeting message."""
    return "Hello from emobon-models!"


__all__ = [
    "ModelingConfig",
    "modeling_config_from_analysis",
    "run_group_loocv_with_mlflow",
    "hello",
]
