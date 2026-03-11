"""Inverse EMOBON model: predicts metadata from taxonomy/abundance.

The standard (forward) model predicts abundance profiles from metadata
features.  This module inverts that direction: abundance/taxonomy data
are used as predictive features (X) to regress or classify metadata
columns (y).

Numeric metadata columns are handled by a sklearn regression pipeline
that reuses the regressor selected via ``ModelingConfig.model_type``.
Categorical metadata columns are handled by a separate classification
pipeline whose estimator is selected via
``ModelingConfig.classification_model_type``.  Predictions from both
pipelines are merged into a single output DataFrame so callers receive
a uniform result regardless of target-column types.

The same ``prepare_modeling_dataset`` alignment logic, group-LOOCV
splitting, and MLflow tracking infrastructure used by the forward
model are also used here so workflows stay consistent.
"""

from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mgnify_methods.utils.logging import get_logger

from emobon_models.modeling_config import ModelingConfig
from emobon_models.modeling_data import ModelingDataset
from emobon_models.modeling_data import prepare_modeling_dataset
from emobon_models.modeling_runner import _active_model_params
from emobon_models.modeling_runner import _build_model
from emobon_models.modeling_runner import _get_mlflow_module
from emobon_models.modeling_runner import _group_loocv_masks
from emobon_models.modeling_runner import _resolve_mlflow_tracking_uri
from emobon_models.modeling_runner import _tracking_uri_to_local_dir


logger = get_logger(__name__, level="INFO")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _split_metadata_by_dtype(
    metadata: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Split metadata column names into numeric and categorical lists.

    Parameters
    ----------
    metadata:
        Metadata DataFrame whose columns are to be classified.

    Returns
    -------
    tuple of (numeric_cols, categorical_cols)
        Column name lists partitioned by dtype.
    """
    numeric_cols = (
        metadata.select_dtypes(include=["number", "bool"])
        .columns.tolist()
    )
    categorical_cols = (
        metadata.select_dtypes(exclude=["number", "bool"])
        .columns.tolist()
    )
    return numeric_cols, categorical_cols


def _impute_numeric_targets(y: pd.DataFrame) -> pd.DataFrame:
    """Return *y* with missing numeric values imputed by column median.

    Parameters
    ----------
    y:
        Numeric target DataFrame that may contain NaN values.
    """
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(
        imputer.fit_transform(y),
        index=y.index,
        columns=y.columns,
    )


def _impute_categorical_targets(y: pd.DataFrame) -> pd.DataFrame:
    """Return *y* with missing categorical values filled by most-frequent.

    Parameters
    ----------
    y:
        Categorical target DataFrame that may contain NaN / None values.
    """
    imputer = SimpleImputer(strategy="most_frequent")
    return pd.DataFrame(
        imputer.fit_transform(y),
        index=y.index,
        columns=y.columns,
    )


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def _build_abundance_preprocessor() -> Pipeline:
    """Build median-impute + standard-scale pipeline for abundance features.

    Abundance features are all numeric, so a single shared preprocessing
    pipeline (median imputation followed by standard scaling) is used for
    both the regression and classification sub-pipelines.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _build_regression_pipeline(config: ModelingConfig) -> Pipeline:
    """Build regression pipeline predicting numeric metadata from abundance.

    Parameters
    ----------
    config:
        Modeling configuration; ``model_type`` selects the regressor.
    """
    preprocessor = _build_abundance_preprocessor()
    regressor = _build_model(config)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    )


def _build_classifier_model(config: ModelingConfig) -> Any:
    """Build a classification estimator for the configured classifier type.

    Parameters
    ----------
    config:
        Modeling configuration; ``classification_model_type`` selects
        the underlying classifer (``"random_forest"`` or
        ``"logistic_regression"``).
    """
    logger.info(
        "Building classifier for classification_model_type=%s",
        config.classification_model_type,
    )
    if config.classification_model_type == "random_forest":
        rfc_params = dict(
            config.random_forest_classifier_params or {}
        )
        return RandomForestClassifier(**rfc_params)
    if config.classification_model_type == "logistic_regression":
        lr_params = dict(config.logistic_regression_params or {})
        return LogisticRegression(**lr_params)
    msg = (
        "Unsupported classification_model_type: "
        f"{config.classification_model_type}"
    )
    raise ValueError(msg)


def _build_classification_pipeline(config: ModelingConfig) -> Pipeline:
    """Build classification pipeline predicting categorical metadata.

    ``RandomForestClassifier`` supports multi-output natively; all
    other classifiers are wrapped with ``MultiOutputClassifier`` so
    the pipeline always produces a prediction for every categorical
    target column in a single ``predict`` call.

    Parameters
    ----------
    config:
        Modeling configuration; ``classification_model_type`` selects
        the underlying classifier.
    """
    preprocessor = _build_abundance_preprocessor()
    base_classifier = _build_classifier_model(config)
    if isinstance(base_classifier, RandomForestClassifier):
        estimator: Any = base_classifier
    else:
        estimator = MultiOutputClassifier(base_classifier)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _inverse_fold_metrics(
    y_true_num: pd.DataFrame | None,
    y_pred_num: np.ndarray | None,
    y_true_cat: pd.DataFrame | None,
    y_pred_cat: np.ndarray | None,
) -> dict[str, float]:
    """Compute per-fold metrics for numeric and categorical target sets.

    Regression targets yield MAE and RMSE (rows with any NaN ground
    truth are excluded from the computation).  Classification targets
    yield mean per-target accuracy and mean per-target macro-F1 (NaN
    ground-truth entries are excluded per column).

    Parameters
    ----------
    y_true_num:
        Ground-truth numeric targets (``None`` if no numeric columns).
    y_pred_num:
        Predicted numeric values (``None`` if no numeric columns).
    y_true_cat:
        Ground-truth categorical targets (``None`` if no categorical
        columns).
    y_pred_cat:
        Predicted categorical values (``None`` if no categorical
        columns).
    """
    metrics: dict[str, float] = {}

    y_pred_num_arr: np.ndarray | None = None
    y_pred_cat_arr: np.ndarray | None = None

    if y_pred_num is not None:
        y_pred_num_arr = np.asarray(y_pred_num)
        if y_pred_num_arr.ndim == 1:
            y_pred_num_arr = y_pred_num_arr.reshape(-1, 1)

    if y_pred_cat is not None:
        y_pred_cat_arr = np.asarray(y_pred_cat)
        if y_pred_cat_arr.ndim == 1:
            y_pred_cat_arr = y_pred_cat_arr.reshape(-1, 1)

    if y_true_num is not None and y_pred_num_arr is not None:
        valid_mask = ~y_true_num.isna().any(axis=1)
        if valid_mask.any():
            metrics["mae"] = float(
                mean_absolute_error(
                    y_true_num.loc[valid_mask].to_numpy(),
                    y_pred_num_arr[valid_mask.values],
                )
            )
            metrics["rmse"] = float(
                np.sqrt(
                    mean_squared_error(
                        y_true_num.loc[valid_mask].to_numpy(),
                        y_pred_num_arr[valid_mask.values],
                    )
                )
            )

    if y_true_cat is not None and y_pred_cat_arr is not None:
        y_true_arr = y_true_cat.to_numpy()
        n_targets = y_true_arr.shape[1]
        per_target_acc: list[float] = []
        per_target_f1: list[float] = []
        for j in range(n_targets):
            col_true = y_true_arr[:, j]
            col_pred = y_pred_cat_arr[:, j]
            valid = pd.notna(col_true)
            if valid.any():
                per_target_acc.append(
                    float(
                        accuracy_score(col_true[valid], col_pred[valid])
                    )
                )
                per_target_f1.append(
                    float(
                        f1_score(
                            col_true[valid],
                            col_pred[valid],
                            average="macro",
                            zero_division=0,
                        )
                    )
                )
        if per_target_acc:
            metrics["accuracy"] = float(np.mean(per_target_acc))
            metrics["f1_macro"] = float(np.mean(per_target_f1))

    return metrics


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------


def _extract_pipeline_importances(
    pipeline: Pipeline,
    target_type: str,
) -> list[dict[str, Any]]:
    """Extract importances or coefficients from a fitted inverse pipeline.

    Handles ``RandomForestClassifier``/``Regressor``
    (``feature_importances_``), linear models with ``coef_``, and
    ``MultiOutputClassifier`` wrappers.  Feature names are read from
    the inner ``preprocessor`` pipeline step (abundance column names
    after impute+scale).

    Parameters
    ----------
    pipeline:
        A fitted ``Pipeline`` with ``preprocessor`` and ``model`` steps.
    target_type:
        Value for the ``target_type`` column in the output rows;
        use ``"regression"`` or ``"classification"``.
    """
    model = pipeline.named_steps["model"]
    inner_preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names: list[str] = list(
            inner_preprocessor.get_feature_names_out()
        )
    except Exception:
        feature_names = []

    importances: np.ndarray
    stat_type: str

    if isinstance(model, MultiOutputClassifier):
        estimators = model.estimators_
        if hasattr(estimators[0], "feature_importances_"):
            importances = np.mean(
                [e.feature_importances_ for e in estimators],
                axis=0,
            )
            stat_type = "importance"
        elif hasattr(estimators[0], "coef_"):
            per_est: list[np.ndarray] = []
            for e in estimators:
                coef = np.asarray(e.coef_)
                if coef.ndim == 1:
                    per_est.append(np.abs(coef))
                else:
                    per_est.append(np.abs(coef).mean(axis=0))
            importances = np.mean(per_est, axis=0)
            stat_type = "coef_abs_mean"
        else:
            return []
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        stat_type = "importance"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.abs(coef).mean(axis=0)
        stat_type = "coef_abs_mean"
    else:
        return []

    if not feature_names:
        feature_names = [
            f"feature_{i}" for i in range(len(importances))
        ]

    return [
        {
            "feature": name,
            "importance": float(imp),
            "stat_type": stat_type,
            "target_type": target_type,
        }
        for name, imp in zip(feature_names, importances)
    ]


def _inverse_feature_importance_table(
    reg_pipeline: Pipeline | None,
    cls_pipeline: Pipeline | None,
) -> pd.DataFrame:
    """Combine feature importances from both inverse sub-pipelines.

    Returns a single DataFrame sorted by descending importance with a
    ``target_type`` column (``"regression"`` or ``"classification"``)
    that indicates which pipeline each row belongs to.

    Parameters
    ----------
    reg_pipeline:
        Fitted regression pipeline (``None`` if no numeric targets).
    cls_pipeline:
        Fitted classification pipeline (``None`` if no categorical
        targets).
    """
    rows: list[dict[str, Any]] = []
    if reg_pipeline is not None:
        rows.extend(
            _extract_pipeline_importances(reg_pipeline, "regression")
        )
    if cls_pipeline is not None:
        rows.extend(
            _extract_pipeline_importances(
                cls_pipeline, "classification"
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "stat_type",
                "target_type",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def _write_inverse_artifacts(
    temp_dir: Path,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    importances: pd.DataFrame,
) -> None:
    """Write inverse model run artifacts to CSV files.

    Parameters
    ----------
    temp_dir:
        Directory to write output files into.
    fold_metrics:
        Per-fold metrics DataFrame.
    predictions:
        Combined prediction DataFrame with ``true__`` and ``pred__``
        prefixed columns for all metadata targets.
    importances:
        Feature importance table combining both sub-pipelines.
    """
    logger.info(
        "Writing inverse run artifacts to: %s", temp_dir
    )
    fold_metrics.to_csv(temp_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(
        temp_dir / "fold_predictions.csv", index=False
    )
    importances.to_csv(
        temp_dir / "feature_importances.csv", index=False
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_inverse_group_loocv_with_mlflow(
    metadata_df: pd.DataFrame,
    abundance_df: pd.DataFrame,
    config: ModelingConfig,
) -> dict[str, pd.DataFrame | list[str]]:
    """Run inverse LOOCV (taxonomy -> metadata) and log outputs to MLflow.

    Inverts the standard modeling direction: abundance/taxonomy features
    are used as inputs (X) to predict metadata targets (y).  Numeric
    metadata columns are handled by a regression pipeline; categorical
    columns are handled by a classification pipeline.  Predictions from
    both are merged into a single output DataFrame so callers receive a
    uniform result regardless of target-column types.

    The same ``prepare_modeling_dataset`` call is used as in the forward
    model so data alignment, missingness filtering, and LOOCV grouping
    logic are identical across both directions.

    Parameters
    ----------
    metadata_df:
        Raw metadata table (samples x columns).
        ``config.feature_column`` defines the LOOCV grouping variable.
    abundance_df:
        Raw abundance/taxonomy table (samples x taxa).  Used as X.
    config:
        Modeling configuration.  Regression uses ``model_type``;
        classification uses ``classification_model_type``.

    Returns
    -------
    dict with the following keys:

    ``fold_metrics``
        Per-fold MAE/RMSE (numeric) and accuracy/F1-macro
        (categorical).
    ``predictions``
        Single DataFrame with ``fold``, ``group``, ``true__*`` and
        ``pred__*`` columns for all predicted metadata targets.
    ``feature_importances``
        Importance table with one row per abundance feature, labelled
        by ``target_type`` (``"regression"`` or ``"classification"``).
    ``dropped_metadata_columns``
        Metadata columns dropped by the missingness threshold filter.
    """
    logger.info("Starting inverse group-based LOOCV modeling")
    dataset: ModelingDataset = prepare_modeling_dataset(
        metadata_df=metadata_df,
        abundance_df=abundance_df,
        config=config,
    )

    # Invert direction: X = abundance, y = aligned metadata.
    X_full = dataset.abundance
    y_full = dataset.metadata
    if config.feature_column in y_full.columns:
        y_full = y_full.drop(columns=[config.feature_column])
        logger.info(
            "Excluded grouping feature '%s' from inverse targets",
            config.feature_column,
        )

    if y_full.shape[1] == 0:
        msg = (
            "No metadata target columns remain after excluding "
            f"feature column '{config.feature_column}'"
        )
        raise ValueError(msg)

    numeric_cols, categorical_cols = _split_metadata_by_dtype(y_full)
    logger.info(
        "Inverse targets: %d numeric columns, %d categorical columns",
        len(numeric_cols),
        len(categorical_cols),
    )

    splitter = _group_loocv_masks(dataset.groups)
    if not splitter:
        msg = (
            "Unable to create group-based LOOCV splits from "
            "feature column"
        )
        raise ValueError(msg)

    mlflow = _get_mlflow_module()
    tracking_uri = _resolve_mlflow_tracking_uri(
        config.mlflow_tracking_uri
    )
    tracking_dir = _tracking_uri_to_local_dir(tracking_uri)
    if tracking_dir is not None:
        tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    logger.info(
        "MLflow configured: experiment=%s model_type=%s "
        "classification_model_type=%s",
        config.mlflow_experiment_name,
        config.model_type,
        config.classification_model_type,
    )

    fold_metrics_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []

    with mlflow.start_run(run_name=config.mlflow_run_name):
        mlflow.log_params(
            {
                "model_direction": "inverse",
                "model_type": config.model_type,
                "classification_model_type": (
                    config.classification_model_type
                ),
                "feature_column": config.feature_column,
                "missing_column_threshold": (
                    config.missing_column_threshold
                ),
                "n_numeric_targets": len(numeric_cols),
                "n_categorical_targets": len(categorical_cols),
                **_active_model_params(config),
            }
        )

        for fold_idx, (
            group_value,
            train_mask,
            test_mask,
        ) in enumerate(splitter):
            X_train = X_full.loc[train_mask]
            X_test = X_full.loc[test_mask]
            logger.info(
                "Fold %d/%d group=%s train=%d test=%d",
                fold_idx + 1,
                len(splitter),
                group_value,
                int(train_mask.sum()),
                int(test_mask.sum()),
            )

            y_pred_num: np.ndarray | None = None
            y_pred_cat: np.ndarray | None = None
            y_true_num: pd.DataFrame | None = None
            y_true_cat: pd.DataFrame | None = None

            if numeric_cols:
                y_train_num = _impute_numeric_targets(
                    y_full.loc[train_mask, numeric_cols]
                )
                y_true_num = y_full.loc[test_mask, numeric_cols]
                reg_pipeline = _build_regression_pipeline(config)
                reg_pipeline.fit(X_train, y_train_num)
                y_pred_num = np.asarray(reg_pipeline.predict(X_test))
                if y_pred_num.ndim == 1:
                    y_pred_num = y_pred_num.reshape(-1, 1)

            if categorical_cols:
                y_train_cat = _impute_categorical_targets(
                    y_full.loc[train_mask, categorical_cols]
                )
                y_true_cat = y_full.loc[test_mask, categorical_cols]
                cls_pipeline = _build_classification_pipeline(config)
                cls_pipeline.fit(X_train, y_train_cat)
                y_pred_cat = np.asarray(cls_pipeline.predict(X_test))
                if y_pred_cat.ndim == 1:
                    y_pred_cat = y_pred_cat.reshape(-1, 1)

            metrics = _inverse_fold_metrics(
                y_true_num, y_pred_num, y_true_cat, y_pred_cat
            )
            logger.info("Fold %d metrics: %s", fold_idx, metrics)
            mlflow.log_metrics(
                {k: v for k, v in metrics.items()},
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

            # Assemble per-fold predictions into a single DataFrame.
            parts: list[pd.DataFrame] = [
                pd.DataFrame(
                    {
                        "sample_id": y_full.loc[test_mask].index,
                        "fold": fold_idx,
                        "group": group_value,
                    }
                ).set_index("sample_id")
            ]
            if y_true_num is not None and y_pred_num is not None:
                parts.append(y_true_num.add_prefix("true__"))
                parts.append(
                    pd.DataFrame(
                        y_pred_num,
                        index=y_true_num.index,
                        columns=[
                            f"pred__{c}" for c in numeric_cols
                        ],
                    )
                )
            if y_true_cat is not None and y_pred_cat is not None:
                parts.append(y_true_cat.add_prefix("true__"))
                parts.append(
                    pd.DataFrame(
                        y_pred_cat,
                        index=y_true_cat.index,
                        columns=[
                            f"pred__{c}" for c in categorical_cols
                        ],
                    )
                )
            prediction_rows.append(
                pd.concat(parts, axis=1).reset_index()
            )

        fold_metrics_df = pd.DataFrame(fold_metrics_rows)
        predictions_df = pd.concat(
            prediction_rows, axis=0, ignore_index=True
        )

        # Summary metrics averaged across folds.
        summary: dict[str, float] = {}
        for metric in ("mae", "rmse", "accuracy", "f1_macro"):
            if metric in fold_metrics_df.columns:
                col = fold_metrics_df[metric]
                summary[f"{metric}_mean"] = float(col.mean())
                summary[f"{metric}_std"] = float(col.std(ddof=0))
        mlflow.log_metrics(
            {
                k: v
                for k, v in summary.items()
                if not k.endswith("_std")
            }
        )
        logger.info("Inverse summary metrics: %s", summary)

        # Fit final pipelines on full dataset for feature importances.
        final_reg_pipeline: Pipeline | None = None
        final_cls_pipeline: Pipeline | None = None
        if numeric_cols:
            y_num_full = _impute_numeric_targets(y_full[numeric_cols])
            final_reg_pipeline = _build_regression_pipeline(config)
            final_reg_pipeline.fit(X_full, y_num_full)
        if categorical_cols:
            y_cat_full = _impute_categorical_targets(
                y_full[categorical_cols]
            )
            final_cls_pipeline = _build_classification_pipeline(
                config
            )
            final_cls_pipeline.fit(X_full, y_cat_full)

        importances_df = _inverse_feature_importance_table(
            final_reg_pipeline, final_cls_pipeline
        )
        logger.info(
            "Inverse feature importance table: %d rows",
            len(importances_df),
        )

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _write_inverse_artifacts(
                temp_dir,
                fold_metrics_df,
                predictions_df,
                importances_df,
            )
            mlflow.log_artifact(str(temp_dir / "fold_metrics.csv"))
            mlflow.log_artifact(
                str(temp_dir / "fold_predictions.csv")
            )
            mlflow.log_artifact(
                str(temp_dir / "feature_importances.csv")
            )

    logger.info("Completed inverse LOOCV run successfully")
    return {
        "fold_metrics": fold_metrics_df,
        "predictions": predictions_df,
        "feature_importances": importances_df,
        "dropped_metadata_columns": dataset.dropped_metadata_columns,
    }
