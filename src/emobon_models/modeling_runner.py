"""Training and evaluation utilities for EMOBON random-forest models."""

from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from emobon_models.modeling_config import ModelingConfig
from emobon_models.modeling_data import ModelingDataset
from emobon_models.modeling_data import prepare_modeling_dataset


def _build_preprocessor(metadata: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical data."""
    numeric_columns = metadata.select_dtypes(
        include=["number", "bool"]
    ).columns
    categorical_columns = metadata.select_dtypes(
        exclude=["number", "bool"]
    ).columns

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
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
    preprocessor = _build_preprocessor(metadata)
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


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
    """Build feature-importance table using transformed feature space."""
    preprocessor: ColumnTransformer = fitted_pipeline.named_steps[
        "preprocessor"
    ]
    model: RandomForestRegressor = fitted_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
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
    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    fold_metrics_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []

    with mlflow.start_run(run_name=config.mlflow_run_name):
        mlflow.log_params(
            {
                "feature_column": config.feature_column,
                "missing_column_threshold": config.missing_column_threshold,
                "n_estimators": config.n_estimators,
                "random_state": config.random_state,
                "n_jobs": config.n_jobs,
                "max_depth": config.max_depth,
                "min_samples_leaf": config.min_samples_leaf,
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
