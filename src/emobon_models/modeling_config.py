"""Configuration models for EMOBON metadata-to-abundance modeling."""

from dataclasses import dataclass
from typing import Any
from typing import Mapping


@dataclass(slots=True)
class ModelingConfig:
    """Typed configuration for random-forest abundance modeling."""

    feature_column: str
    sample_id_column: str | None = None
    missing_column_threshold: float = 0.4
    n_estimators: int = 500
    random_state: int = 42
    n_jobs: int = -1
    max_depth: int | None = None
    min_samples_leaf: int = 1
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "emobon-rf-loocv"
    mlflow_run_name: str | None = None


def _validate_threshold(threshold: float) -> float:
    """Validate and return the missingness threshold."""
    if not 0.0 <= threshold <= 1.0:
        msg = "missing_column_threshold must be between 0.0 and 1.0"
        raise ValueError(msg)
    return threshold


def modeling_config_from_analysis(config: Mapping[str, Any]) -> ModelingConfig:
    """Build modeling config from the notebook analysis configuration."""
    feature_column = str(config["feature"])
    modeling_section = dict(config.get("modeling", {}))

    threshold = float(modeling_section.get("missing_column_threshold", 0.4))
    threshold = _validate_threshold(threshold)

    rf_params = dict(modeling_section.get("random_forest", {}))
    mlflow_section = dict(modeling_section.get("mlflow", {}))

    return ModelingConfig(
        feature_column=feature_column,
        sample_id_column=modeling_section.get("sample_id_column"),
        missing_column_threshold=threshold,
        n_estimators=int(rf_params.get("n_estimators", 500)),
        random_state=int(rf_params.get("random_state", 42)),
        n_jobs=int(rf_params.get("n_jobs", -1)),
        max_depth=rf_params.get("max_depth"),
        min_samples_leaf=int(rf_params.get("min_samples_leaf", 1)),
        mlflow_tracking_uri=mlflow_section.get("tracking_uri"),
        mlflow_experiment_name=str(
            mlflow_section.get("experiment_name", "emobon-rf-loocv")
        ),
        mlflow_run_name=mlflow_section.get("run_name"),
    )
