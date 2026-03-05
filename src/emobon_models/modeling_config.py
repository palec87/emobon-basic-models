"""Configuration models for EMOBON metadata-to-abundance modeling."""

from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import Mapping

from mgnify_methods.utils.logging import get_logger


logger = get_logger(__name__, level="INFO")


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

    tuning_enabled: bool = False
    tuning_method: str = "grid_search"
    tuning_scoring: str = "neg_mean_squared_error"
    tuning_n_jobs: int = -1
    tuning_refit: bool = True
    tuning_error_score: str | float = "raise"
    tuning_verbose: int = 0
    tuning_inner_cv_folds: int = 3
    tuning_max_candidates: int = 10
    tuning_grids: dict[str, dict[str, list[Any]]] | None = None

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


def _normalize_tuning_grids(
    tuning_grids: Mapping[str, Any] | None,
) -> dict[str, dict[str, list[Any]]]:
    """Normalize tuning grids to {model: {param: [values]}} format."""
    if tuning_grids is None:
        return {}

    normalized: dict[str, dict[str, list[Any]]] = {}
    for model_name, raw_grid in dict(tuning_grids).items():
        if not isinstance(raw_grid, Mapping):
            msg = f"Tuning grid for '{model_name}' must be a mapping"
            raise ValueError(msg)

        model_grid: dict[str, list[Any]] = {}
        for param_name, param_values in dict(raw_grid).items():
            if isinstance(param_values, list):
                values = param_values
            else:
                values = [param_values]
            if not values:
                msg = (
                    f"Tuning grid parameter '{param_name}' for "
                    f"'{model_name}' cannot be empty"
                )
                raise ValueError(msg)
            model_grid[str(param_name)] = values

        normalized[str(model_name)] = model_grid

    return normalized


def _validate_tuning_method(method: str) -> str:
    """Validate requested tuning method."""
    if method != "grid_search":
        msg = f"Unsupported tuning method '{method}'"
        raise ValueError(msg)
    return method


def modeling_config_from_analysis(config: Mapping[str, Any]) -> ModelingConfig:
    """Build modeling config from the notebook analysis configuration."""
    logger.info("Building ModelingConfig from analysis config")
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
    tuning_section = _mapping_to_dict(modeling_section.get("tuning"))
    mlflow_section = _mapping_to_dict(modeling_section.get("mlflow"))

    tuning_enabled = bool(tuning_section.get("enabled", False))
    tuning_method = _validate_tuning_method(
        str(tuning_section.get("method", "grid_search"))
    )
    tuning_scoring = str(
        tuning_section.get("scoring", "neg_mean_squared_error")
    )
    tuning_n_jobs = int(tuning_section.get("n_jobs", -1))
    tuning_refit = bool(tuning_section.get("refit", True))
    tuning_error_score = tuning_section.get("error_score", "raise")
    tuning_verbose = int(tuning_section.get("verbose", 0))
    tuning_inner_cv_folds = int(tuning_section.get("inner_cv_folds", 3))
    tuning_max_candidates = int(tuning_section.get("max_candidates", 10))
    tuning_grids = _normalize_tuning_grids(
        tuning_section.get("grids")
    )

    if tuning_inner_cv_folds < 2:
        msg = "modeling.tuning.inner_cv_folds must be >= 2"
        raise ValueError(msg)

    if tuning_enabled:
        model_grid = tuning_grids.get(model_type, {})
        if not model_grid:
            msg = (
                "Tuning is enabled but no grid is configured for "
                f"model_type '{model_type}'"
            )
            raise ValueError(msg)
        logger.info(
            "Tuning enabled for model '%s' with %d grid params",
            model_type,
            len(model_grid),
        )

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

    model_config = ModelingConfig(
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
        tuning_enabled=tuning_enabled,
        tuning_method=tuning_method,
        tuning_scoring=tuning_scoring,
        tuning_n_jobs=tuning_n_jobs,
        tuning_refit=tuning_refit,
        tuning_error_score=tuning_error_score,
        tuning_verbose=tuning_verbose,
        tuning_inner_cv_folds=tuning_inner_cv_folds,
        tuning_max_candidates=tuning_max_candidates,
        tuning_grids=tuning_grids,
        mlflow_tracking_uri=mlflow_section.get("tracking_uri"),
        mlflow_experiment_name=str(
            mlflow_section.get("experiment_name", "emobon-rf-loocv")
        ),
        mlflow_run_name=mlflow_section.get("run_name"),
    )
    logger.info(
        "Model config ready: model_type=%s, tuning_enabled=%s",
        model_config.model_type,
        model_config.tuning_enabled,
    )
    return model_config
