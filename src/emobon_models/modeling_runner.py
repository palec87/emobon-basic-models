"""Training and evaluation utilities for EMOBON regression models."""

from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mgnify_methods.utils.logging import get_logger

from emobon_models.modeling_config import ModelingConfig
from emobon_models.modeling_data import ModelingDataset
from emobon_models.modeling_data import prepare_modeling_dataset


logger = get_logger(__name__, level="INFO")


def _standardize_numeric_features(config: ModelingConfig) -> bool:
    """Return whether numeric metadata features should be standardized."""
    return config.model_type in {"ridge", "pls", "elasticnet"}


def _build_preprocessor(
    metadata: pd.DataFrame,
    config: ModelingConfig,
) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical data."""
    numeric_columns = metadata.select_dtypes(
        include=["number", "bool"]
    ).columns
    categorical_columns = metadata.select_dtypes(
        exclude=["number", "bool"]
    ).columns
    logger.info(
        "Building preprocessor: numeric=%d categorical=%d",
        len(numeric_columns),
        len(categorical_columns),
    )

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if _standardize_numeric_features(config):
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ]
    )


def _build_pipeline(
    metadata: pd.DataFrame,
    config: ModelingConfig,
) -> Pipeline:
    """Build full model pipeline for multi-output abundance prediction."""
    preprocessor = _build_preprocessor(metadata, config)
    model = _build_model(config)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _build_model(config: ModelingConfig) -> Any:
    """Build selected estimator based on configured model type."""
    logger.info("Building estimator for model_type=%s", config.model_type)
    if config.model_type == "random_forest":
        rf_params = dict(config.random_forest_params or {})
        return RandomForestRegressor(**rf_params)

    if config.model_type == "ridge":
        ridge_params = dict(config.ridge_params or {})
        return Ridge(**ridge_params)

    if config.model_type == "pls":
        pls_params = dict(config.pls_params or {})
        return PLSRegression(**pls_params)

    if config.model_type == "elasticnet":
        elasticnet_params = dict(config.elasticnet_params or {})
        return MultiTaskElasticNet(**elasticnet_params)

    msg = f"Unsupported model_type: {config.model_type}"
    raise ValueError(msg)


def _active_model_params(config: ModelingConfig) -> dict[str, Any]:
    """Return MLflow params for the active model only."""
    if config.model_type == "random_forest":
        model_params = dict(config.random_forest_params or {})
    elif config.model_type == "ridge":
        model_params = dict(config.ridge_params or {})
    elif config.model_type == "pls":
        model_params = dict(config.pls_params or {})
    elif config.model_type == "elasticnet":
        model_params = dict(config.elasticnet_params or {})
    else:
        msg = f"Unsupported model_type: {config.model_type}"
        raise ValueError(msg)

    prefixed_params: dict[str, Any] = {}
    for key, value in model_params.items():
        prefixed_params[f"{config.model_type}__{key}"] = value
    return prefixed_params


def _normalize_tuning_grid(
    model_grid: dict[str, list[Any]],
) -> dict[str, list[Any]]:
    """Prefix model hyperparameter grid keys for pipeline search."""
    normalized: dict[str, list[Any]] = {}
    for key, values in model_grid.items():
        search_key = key if "__" in key else f"model__{key}"
        normalized[search_key] = values
    return normalized


def _count_grid_candidates(model_grid: dict[str, list[Any]]) -> int:
    """Count number of combinations in a model parameter grid."""
    total = 1
    for values in model_grid.values():
        total *= len(values)
    return total


def _inner_group_splitter(
    groups: pd.Series,
    n_splits: int,
) -> GroupKFold | None:
    """Build inner group splitter with safe split count from groups."""
    unique_groups = int(groups.dropna().astype("string").nunique())
    safe_splits = min(n_splits, unique_groups)
    if safe_splits < 2:
        return None
    return GroupKFold(n_splits=safe_splits)


def _fit_pipeline_with_optional_tuning(
    metadata: pd.DataFrame,
    config: ModelingConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    train_groups: pd.Series,
) -> tuple[Pipeline, dict[str, Any], float | None, pd.DataFrame | None]:
    """Fit pipeline directly or through nested GridSearchCV."""
    pipeline = _build_pipeline(metadata, config)
    logger.info("FITTING Training X and y")
    logger.info(f"{X_train.head()}")
    logger.info(f"{y_train.head()} with groups")
    logger.info(f"{train_groups.head()}")
    if not config.tuning_enabled:
        logger.info("Tuning disabled; fitting pipeline directly")
        pipeline.fit(X_train, y_train)
        return pipeline, {}, None, None

    if config.tuning_method != "grid_search":
        msg = f"Unsupported tuning method: {config.tuning_method}"
        raise ValueError(msg)

    tuning_grids = dict(config.tuning_grids or {})
    model_grid = dict(tuning_grids.get(config.model_type, {}))
    if not model_grid:
        msg = (
            "No tuning grid found for model type "
            f"'{config.model_type}'"
        )
        raise ValueError(msg)

    n_candidates = _count_grid_candidates(model_grid)
    logger.info(
        "Tuning enabled for %s with %d candidates",
        config.model_type,
        n_candidates,
    )
    if config.tuning_max_candidates > 0:
        if n_candidates > config.tuning_max_candidates:
            msg = (
                "Grid has more candidates than tuning_max_candidates: "
                f"{n_candidates} > {config.tuning_max_candidates}"
            )
            raise ValueError(msg)

    splitter = _inner_group_splitter(
        train_groups,
        config.tuning_inner_cv_folds,
    )
    if splitter is None:
        logger.info("Skipping inner CV tuning due to insufficient groups")
        pipeline.fit(X_train, y_train)
        return pipeline, {}, None, None

    search_grid = _normalize_tuning_grid(model_grid)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=search_grid,
        scoring=config.tuning_scoring,
        n_jobs=config.tuning_n_jobs,
        cv=splitter,
        refit=config.tuning_refit,
        error_score=config.tuning_error_score,
        verbose=config.tuning_verbose,
    )
    search.fit(X_train, y_train, groups=train_groups)
    logger.info(
        "Grid search done for %s; best_score=%.6f",
        config.model_type,
        float(search.best_score_),
    )

    best_params = {
        key.replace("model__", "", 1): value
        for key, value in search.best_params_.items()
    }

    cv_results = pd.DataFrame(search.cv_results_)
    keep_columns = [
        column
        for column in cv_results.columns
        if column.startswith("param_")
        or column in {"mean_test_score", "rank_test_score"}
    ]
    cv_summary = cv_results[keep_columns].copy()

    return (
        search.best_estimator_,
        best_params,
        float(search.best_score_),
        cv_summary,
    )


def _group_loocv_masks(
    groups: pd.Series,
) -> list[tuple[str, pd.Series, pd.Series]]:
    """Create leave-one-group-out masks from a grouping feature series."""
    masks: list[tuple[str, pd.Series, pd.Series]] = []
    unique_groups = groups.dropna().astype("string").unique().tolist()
    for group_value in unique_groups:
        test_mask = groups.astype("string") == group_value
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        masks.append((str(group_value), train_mask, test_mask))
    return masks


def _fold_metrics(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute fold metrics for multi-output abundance regression."""
    mae = float(mean_absolute_error(y_true.to_numpy(), y_pred))
    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true.to_numpy(),
                y_pred,
            )
        )
    )
    return {"mae": mae, "rmse": rmse}


def _feature_importance_table(
    fitted_pipeline: Pipeline,
) -> pd.DataFrame:
    """Build feature-stat table using importances or model coefficients."""
    preprocessor: ColumnTransformer = fitted_pipeline.named_steps[
        "preprocessor"
    ]
    model = fitted_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        stat_type = "importance"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            importances = np.abs(coef)
        elif coef.ndim == 2:
            feature_count = len(feature_names)
            if coef.shape[0] == feature_count:
                importances = np.abs(coef).mean(axis=1)
            elif coef.shape[1] == feature_count:
                importances = np.abs(coef).mean(axis=0)
            else:
                msg = "Unable to align coefficient shape with feature names"
                raise ValueError(msg)
        else:
            msg = "Unsupported coefficient array shape"
            raise ValueError(msg)
        stat_type = "coef_abs_mean"
    else:
        msg = "Model does not expose feature_importances_ or coef_"
        raise ValueError(msg)

    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
            "stat_type": stat_type,
        }
    ).sort_values("importance", ascending=False)


def _get_mlflow_module() -> Any:
    """Import mlflow lazily and raise a clear error if unavailable."""
    try:
        import mlflow
    except ImportError as exc:
        msg = "mlflow is required for run_group_loocv_with_mlflow"
        raise ImportError(msg) from exc
    return mlflow


def _resolve_mlflow_tracking_uri(config_uri: str | None) -> str:
    """Return a safe tracking URI for MLflow on local environments."""
    if not config_uri:
        default_dir = Path.cwd() / "mlruns"
        return default_dir.resolve().as_uri()

    if len(config_uri) > 1 and config_uri[1:3] == ":\\":
        return Path(config_uri).resolve().as_uri()

    parsed = urlparse(config_uri)
    if parsed.scheme in {
        "http",
        "https",
        "sqlite",
        "postgresql",
        "mysql",
        "mssql",
    }:
        return config_uri

    if parsed.scheme and parsed.scheme != "file":
        return config_uri

    if parsed.scheme == "":
        return Path(config_uri).resolve().as_uri()

    decoded_path = unquote(parsed.path)
    if not decoded_path:
        return (Path.cwd() / "mlruns").resolve().as_uri()

    if decoded_path.startswith("/") and len(decoded_path) > 2:
        if decoded_path[2:3] == ":":
            decoded_path = decoded_path[1:]

    return Path(decoded_path).resolve().as_uri()


def _tracking_uri_to_local_dir(tracking_uri: str) -> Path | None:
    """Convert file-based tracking URI to local directory path."""
    parsed = urlparse(tracking_uri)
    if parsed.scheme != "file":
        return None

    decoded_path = unquote(parsed.path)
    if decoded_path.startswith("/") and len(decoded_path) > 2:
        if decoded_path[2:3] == ":":
            decoded_path = decoded_path[1:]
    return Path(decoded_path)


def _write_artifacts(
    temp_dir: Path,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    importances: pd.DataFrame,
    tuning_best_params: pd.DataFrame | None = None,
    tuning_cv_scores: pd.DataFrame | None = None,
) -> None:
    """Write dataframes to CSV files before MLflow artifact logging."""
    logger.info("Writing run artifacts to temporary directory: %s", temp_dir)
    fold_metrics.to_csv(temp_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(temp_dir / "fold_predictions.csv", index=False)
    importances.to_csv(temp_dir / "feature_importances.csv", index=False)
    if tuning_best_params is not None:
        if not tuning_best_params.empty:
            tuning_best_params.to_csv(
                temp_dir / "fold_best_params.csv",
                index=False,
            )
    if tuning_cv_scores is not None:
        if not tuning_cv_scores.empty:
            tuning_cv_scores.to_csv(
                temp_dir / "fold_inner_cv_scores.csv",
                index=False,
            )


def run_group_loocv_with_mlflow(
    metadata_df: pd.DataFrame,
    abundance_df: pd.DataFrame,
    config: ModelingConfig,
) -> dict[str, pd.DataFrame | list[str]]:
    """Run group-based LOOCV and log training outputs to MLflow."""
    logger.info("Starting group-based LOOCV modeling")
    dataset: ModelingDataset = prepare_modeling_dataset(
        metadata_df=metadata_df,
        abundance_df=abundance_df,
        config=config,
    )

    splitter = _group_loocv_masks(dataset.groups)
    if not splitter:
        msg = "Unable to create group-based LOOCV splits from feature column"
        raise ValueError(msg)

    mlflow = _get_mlflow_module()
    tracking_uri = _resolve_mlflow_tracking_uri(config.mlflow_tracking_uri)
    tracking_dir = _tracking_uri_to_local_dir(tracking_uri)
    if tracking_dir is not None:
        tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    logger.info(
        "MLflow configured: experiment=%s model_type=%s",
        config.mlflow_experiment_name,
        config.model_type,
    )

    fold_metrics_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []
    tuning_best_rows: list[dict[str, Any]] = []
    tuning_cv_rows: list[pd.DataFrame] = []

    with mlflow.start_run(run_name=config.mlflow_run_name):
        mlflow.log_params(
            {
                "model_type": config.model_type,
                "feature_column": config.feature_column,
                "missing_column_threshold": config.missing_column_threshold,
                "tuning_enabled": config.tuning_enabled,
                "tuning_method": config.tuning_method,
                "tuning_inner_cv_folds": config.tuning_inner_cv_folds,
                "tuning_scoring": config.tuning_scoring,
                **_active_model_params(config),
            }
        )

        for fold_idx, masks in enumerate(splitter):
            group_value, train_mask, test_mask = masks
            X_train = dataset.metadata.loc[train_mask]
            X_test = dataset.metadata.loc[test_mask]
            y_train = dataset.abundance.loc[train_mask]
            y_test = dataset.abundance.loc[test_mask]
            inner_groups = dataset.groups.loc[train_mask]
            logger.info(
                "Fold %d/%d group=%s train=%d test=%d",
                fold_idx + 1,
                len(splitter),
                group_value,
                int(train_mask.sum()),
                int(test_mask.sum()),
            )

            (
                pipeline,
                best_params,
                best_score,
                cv_summary,
            ) = _fit_pipeline_with_optional_tuning(
                metadata=dataset.metadata,
                config=config,
                X_train=X_train,
                y_train=y_train,
                train_groups=inner_groups,
            )
            y_pred = pipeline.predict(X_test)

            if best_params:
                best_row: dict[str, Any] = {
                    "fold": fold_idx,
                    "group": group_value,
                    "best_score": best_score,
                }
                best_row.update(best_params)
                tuning_best_rows.append(best_row)

                for param_key, param_value in best_params.items():
                    mlflow.log_param(
                        f"fold_{fold_idx}__best__{param_key}",
                        param_value,
                    )

            if best_score is not None:
                mlflow.log_metric(
                    "fold_inner_best_score",
                    best_score,
                    step=fold_idx,
                )
                logger.info(
                    "Fold %d inner best score: %.6f",
                    fold_idx,
                    best_score,
                )

            if cv_summary is not None:
                cv_row = cv_summary.copy()
                cv_row.insert(0, "fold", fold_idx)
                cv_row.insert(1, "group", group_value)
                tuning_cv_rows.append(cv_row)

            metrics = _fold_metrics(y_test, y_pred)
            logger.info(
                "Fold %d metrics: mae=%.6f rmse=%.6f",
                fold_idx,
                metrics["mae"],
                metrics["rmse"],
            )
            mlflow.log_metrics(
                {
                    "fold_mae": metrics["mae"],
                    "fold_rmse": metrics["rmse"],
                },
                step=fold_idx,
            )

            fold_metrics_rows.append(
                {
                    "fold": fold_idx,
                    "group": group_value,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    **metrics,
                }
            )

            fold_pred_df = pd.DataFrame(
                y_pred,
                index=y_test.index,
                columns=y_test.columns,
            )
            fold_pred_df = fold_pred_df.add_prefix("pred__")

            fold_true_df = y_test.add_prefix("true__")
            fold_full_df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "sample_id": y_test.index,
                            "fold": fold_idx,
                            "group": group_value,
                        }
                    ).set_index("sample_id"),
                    fold_true_df,
                    fold_pred_df,
                ],
                axis=1,
            ).reset_index()
            prediction_rows.append(fold_full_df)

        fold_metrics_df = pd.DataFrame(fold_metrics_rows)
        predictions_df = pd.concat(prediction_rows, axis=0, ignore_index=True)

        summary_metrics = pd.DataFrame(
            [
                {
                    "mae_mean": float(fold_metrics_df["mae"].mean()),
                    "mae_std": float(fold_metrics_df["mae"].std(ddof=0)),
                    "rmse_mean": float(fold_metrics_df["rmse"].mean()),
                    "rmse_std": float(fold_metrics_df["rmse"].std(ddof=0)),
                }
            ]
        )
        mlflow.log_metrics(
            {
                "mae_mean": float(summary_metrics.iloc[0]["mae_mean"]),
                "rmse_mean": float(summary_metrics.iloc[0]["rmse_mean"]),
            }
        )
        logger.info(
            "Summary metrics: mae_mean=%.6f rmse_mean=%.6f",
            float(summary_metrics.iloc[0]["mae_mean"]),
            float(summary_metrics.iloc[0]["rmse_mean"]),
        )

        (
            final_pipeline,
            final_best_params,
            final_best_score,
            _,
        ) = _fit_pipeline_with_optional_tuning(
            metadata=dataset.metadata,
            config=config,
            X_train=dataset.metadata,
            y_train=dataset.abundance,
            train_groups=dataset.groups,
        )

        if final_best_params:
            for param_key, param_value in final_best_params.items():
                mlflow.log_param(
                    f"final_best__{param_key}",
                    param_value,
                )
        if final_best_score is not None:
            mlflow.log_metric("final_inner_best_score", final_best_score)

        importances_df = _feature_importance_table(final_pipeline)
        logger.info(
            "Computed feature statistics table with %d rows",
            len(importances_df),
        )

        tuning_best_params_df = pd.DataFrame(tuning_best_rows)
        if tuning_cv_rows:
            tuning_cv_scores_df = pd.concat(
                tuning_cv_rows,
                axis=0,
                ignore_index=True,
            )
        else:
            tuning_cv_scores_df = pd.DataFrame()

        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
        )

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _write_artifacts(
                temp_dir=temp_dir,
                fold_metrics=fold_metrics_df,
                predictions=predictions_df,
                importances=importances_df,
                tuning_best_params=tuning_best_params_df,
                tuning_cv_scores=tuning_cv_scores_df,
            )
            mlflow.log_artifact(str(temp_dir / "fold_metrics.csv"))
            mlflow.log_artifact(str(temp_dir / "fold_predictions.csv"))
            mlflow.log_artifact(str(temp_dir / "feature_importances.csv"))
            if not tuning_best_params_df.empty:
                mlflow.log_artifact(str(temp_dir / "fold_best_params.csv"))
            if not tuning_cv_scores_df.empty:
                mlflow.log_artifact(
                    str(temp_dir / "fold_inner_cv_scores.csv")
                )

    logger.info("Completed LOOCV run successfully")

    return {
        "fold_metrics": fold_metrics_df,
        "summary_metrics": summary_metrics,
        "predictions": predictions_df,
        "feature_importances": importances_df,
        "dropped_metadata_columns": pd.DataFrame(
            {"column": dataset.dropped_metadata_columns}
        ),
    }
