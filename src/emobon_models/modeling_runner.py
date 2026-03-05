"""Training and evaluation utilities for EMOBON regression models."""

from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import unquote
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from emobon_models.modeling_config import ModelingConfig
from emobon_models.modeling_data import ModelingDataset
from emobon_models.modeling_data import prepare_modeling_dataset


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
) -> None:
    """Write dataframes to CSV files before MLflow artifact logging."""
    fold_metrics.to_csv(temp_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(temp_dir / "fold_predictions.csv", index=False)
    importances.to_csv(temp_dir / "feature_importances.csv", index=False)


def run_group_loocv_with_mlflow(
    metadata_df: pd.DataFrame,
    abundance_df: pd.DataFrame,
    config: ModelingConfig,
) -> dict[str, pd.DataFrame | list[str]]:
    """Run group-based LOOCV and log training outputs to MLflow."""
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

    fold_metrics_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []

    with mlflow.start_run(run_name=config.mlflow_run_name):
        mlflow.log_params(
            {
                "model_type": config.model_type,
                "feature_column": config.feature_column,
                "missing_column_threshold": config.missing_column_threshold,
                **_active_model_params(config),
            }
        )

        for fold_idx, masks in enumerate(splitter):
            group_value, train_mask, test_mask = masks
            X_train = dataset.metadata.loc[train_mask]
            X_test = dataset.metadata.loc[test_mask]
            y_train = dataset.abundance.loc[train_mask]
            y_test = dataset.abundance.loc[test_mask]

            pipeline = _build_pipeline(dataset.metadata, config)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = _fold_metrics(y_test, y_pred)
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

        final_pipeline = _build_pipeline(dataset.metadata, config)
        final_pipeline.fit(dataset.metadata, dataset.abundance)
        importances_df = _feature_importance_table(final_pipeline)

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
            )
            mlflow.log_artifact(str(temp_dir / "fold_metrics.csv"))
            mlflow.log_artifact(str(temp_dir / "fold_predictions.csv"))
            mlflow.log_artifact(str(temp_dir / "feature_importances.csv"))

    return {
        "fold_metrics": fold_metrics_df,
        "summary_metrics": summary_metrics,
        "predictions": predictions_df,
        "feature_importances": importances_df,
        "dropped_metadata_columns": pd.DataFrame(
            {"column": dataset.dropped_metadata_columns}
        ),
    }
