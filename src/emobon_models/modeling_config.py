"""Configuration models for EMOBON metadata-to-abundance modeling."""

from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import Mapping


ModelType = Literal[
    "random_forest",
    "ridge",
    "pls",
    "elasticnet",
]


@dataclass(slots=True)
class ModelingConfig:
    """Typed configuration for metadata-to-abundance modeling."""

    feature_column: str
    model_type: ModelType = "random_forest"
    sample_id_column: str | None = None
    missing_column_threshold: float = 0.4

    # Legacy random-forest fields kept for backward compatibility.
    n_estimators: int = 500
    random_state: int = 42
    n_jobs: int = -1
    max_depth: int | None = None
    min_samples_leaf: int = 1

    random_forest_params: dict[str, Any] | None = None
    ridge_params: dict[str, Any] | None = None
    pls_params: dict[str, Any] | None = None
    elasticnet_params: dict[str, Any] | None = None

    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "emobon-rf-loocv"
    mlflow_run_name: str | None = None


def _validate_threshold(threshold: float) -> float:
    """Validate and return the missingness threshold."""
    if not 0.0 <= threshold <= 1.0:
        msg = "missing_column_threshold must be between 0.0 and 1.0"
        raise ValueError(msg)
    return threshold


def _validate_model_type(model_type: str) -> ModelType:
    """Validate and return supported model type name."""
    allowed = {
        "random_forest",
        "ridge",
        "pls",
        "elasticnet",
    }
    if model_type not in allowed:
        msg = f"Unsupported model_type '{model_type}'"
        raise ValueError(msg)
    return model_type  # type: ignore[return-value]


def _mapping_to_dict(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert mapping or missing section to a mutable dictionary."""
    if mapping is None:
        return {}
    return dict(mapping)


def modeling_config_from_analysis(config: Mapping[str, Any]) -> ModelingConfig:
    """Build modeling config from the notebook analysis configuration."""
    feature_column = str(config["feature"])
    modeling_section = dict(config.get("modeling", {}))

    threshold = float(modeling_section.get("missing_column_threshold", 0.4))
    threshold = _validate_threshold(threshold)

    model_type = _validate_model_type(
        str(modeling_section.get("model_type", "random_forest"))
    )

    rf_params = _mapping_to_dict(modeling_section.get("random_forest"))
    ridge_params = _mapping_to_dict(modeling_section.get("ridge"))
    pls_params = _mapping_to_dict(modeling_section.get("pls"))
    elasticnet_params = _mapping_to_dict(modeling_section.get("elasticnet"))
    mlflow_section = _mapping_to_dict(modeling_section.get("mlflow"))

    default_rf_params: dict[str, Any] = {
        "n_estimators": 500,
        "random_state": 42,
        "n_jobs": -1,
        "max_depth": None,
        "min_samples_leaf": 1,
    }
    default_ridge_params: dict[str, Any] = {
        "alpha": 1.0,
        "fit_intercept": True,
        "copy_X": True,
        "max_iter": None,
        "tol": 1e-4,
        "solver": "auto",
        "positive": False,
        "random_state": None,
    }
    default_pls_params: dict[str, Any] = {
        "n_components": 2,
        "scale": False,
        "max_iter": 500,
        "tol": 1e-6,
        "copy": True,
    }
    default_elasticnet_params: dict[str, Any] = {
        "alpha": 1.0,
        "l1_ratio": 0.5,
        "fit_intercept": True,
        "copy_X": True,
        "max_iter": 1000,
        "tol": 1e-4,
        "warm_start": False,
        "random_state": None,
        "selection": "cyclic",
    }

    rf_full = {**default_rf_params, **rf_params}
    ridge_full = {**default_ridge_params, **ridge_params}
    pls_full = {**default_pls_params, **pls_params}
    elasticnet_full = {
        **default_elasticnet_params,
        **elasticnet_params,
    }

    return ModelingConfig(
        feature_column=feature_column,
        model_type=model_type,
        sample_id_column=modeling_section.get("sample_id_column"),
        missing_column_threshold=threshold,
        n_estimators=int(rf_full["n_estimators"]),
        random_state=int(rf_full["random_state"]),
        n_jobs=int(rf_full["n_jobs"]),
        max_depth=rf_full["max_depth"],
        min_samples_leaf=int(rf_full["min_samples_leaf"]),
        random_forest_params=rf_full,
        ridge_params=ridge_full,
        pls_params=pls_full,
        elasticnet_params=elasticnet_full,
        mlflow_tracking_uri=mlflow_section.get("tracking_uri"),
        mlflow_experiment_name=str(
            mlflow_section.get("experiment_name", "emobon-rf-loocv")
        ),
        mlflow_run_name=mlflow_section.get("run_name"),
    )
