"""Tests for EMOBON modeling module helpers."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from emobon_models.modeling_config import ModelingConfig
from emobon_models.modeling_config import modeling_config_from_analysis
from emobon_models.modeling_data import prepare_modeling_dataset
from emobon_models.modeling_runner import _count_grid_candidates
from emobon_models.modeling_runner import _build_model
from emobon_models.modeling_runner import _build_preprocessor
from emobon_models.modeling_runner import _fit_pipeline_with_optional_tuning
from emobon_models.modeling_runner import _group_loocv_masks
from emobon_models.modeling_runner import _normalize_tuning_grid


def test_modeling_config_from_analysis_uses_defaults() -> None:
    """Build config from minimal analysis config with expected defaults."""
    config = {
        "feature": "study_tag",
        "modeling": {
            "missing_column_threshold": 0.5,
        },
    }

    model_cfg = modeling_config_from_analysis(config)

    assert model_cfg.feature_column == "study_tag"
    assert model_cfg.missing_column_threshold == 0.5
    assert model_cfg.n_estimators == 500


def test_prepare_modeling_dataset_filters_and_aligns() -> None:
    """Align metadata and abundance tables and drop high-missing columns."""
    metadata = pd.DataFrame(
        {
            "study_tag": ["A", "A", "B"],
            "season": ["spring", None, None],
            "latitude": [1.0, 2.0, 3.0],
        },
        index=["s1", "s2", "s3"],
    )
    abundance = pd.DataFrame(
        {
            "taxon_1": [0.1, 0.2, 0.3],
            "taxon_2": [0.0, 0.1, 0.2],
        },
        index=["s1", "s2", "s3"],
    )

    model_cfg = modeling_config_from_analysis(
        {
            "feature": "study_tag",
            "modeling": {"missing_column_threshold": 0.4},
        }
    )
    dataset = prepare_modeling_dataset(metadata, abundance, model_cfg)

    assert dataset.metadata.index.tolist() == ["s1", "s2", "s3"]
    assert dataset.abundance.shape == (3, 2)
    assert "season" in dataset.dropped_metadata_columns
    assert dataset.groups.tolist() == ["A", "A", "B"]


def test_group_loocv_masks_creates_group_folds() -> None:
    """Create leave-one-group-out folds from group labels."""
    groups = pd.Series(["A", "A", "B", "C"], index=["s1", "s2", "s3", "s4"])
    masks = _group_loocv_masks(groups)

    fold_groups = [fold[0] for fold in masks]
    assert set(fold_groups) == {"A", "B", "C"}
    assert len(masks) == 3


def test_modeling_config_reads_ridge_params() -> None:
    """Parse ridge model type and model-specific configuration."""
    config = {
        "feature": "study_tag",
        "modeling": {
            "model_type": "ridge",
            "ridge": {
                "alpha": 2.5,
                "solver": "svd",
            },
        },
    }

    model_cfg = modeling_config_from_analysis(config)

    assert model_cfg.model_type == "ridge"
    assert model_cfg.ridge_params is not None
    assert model_cfg.ridge_params["alpha"] == 2.5
    assert model_cfg.ridge_params["solver"] == "svd"


def test_build_model_selects_expected_estimator() -> None:
    """Select expected sklearn estimator for each supported model type."""
    rf_config = ModelingConfig(
        feature_column="study_tag",
        model_type="random_forest",
        random_forest_params={
            "n_estimators": 10,
            "random_state": 1,
            "n_jobs": 1,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
    )
    ridge_config = ModelingConfig(
        feature_column="study_tag",
        model_type="ridge",
        ridge_params={"alpha": 1.0},
    )
    pls_config = ModelingConfig(
        feature_column="study_tag",
        model_type="pls",
        pls_params={"n_components": 2},
    )
    elasticnet_config = ModelingConfig(
        feature_column="study_tag",
        model_type="elasticnet",
        elasticnet_params={"alpha": 1.0},
    )

    assert isinstance(_build_model(rf_config), RandomForestRegressor)
    assert isinstance(_build_model(ridge_config), Ridge)
    assert isinstance(_build_model(pls_config), PLSRegression)
    assert isinstance(_build_model(elasticnet_config), MultiTaskElasticNet)


def test_build_preprocessor_scales_numeric_for_linear_models() -> None:
    """Apply numeric scaling for ridge/pls/elasticnet but not random forest."""
    metadata = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["A", "B", "A"],
        }
    )

    rf_config = ModelingConfig(
        feature_column="study_tag",
        model_type="random_forest",
    )
    ridge_config = ModelingConfig(
        feature_column="study_tag",
        model_type="ridge",
    )

    rf_preprocessor = _build_preprocessor(metadata, rf_config)
    ridge_preprocessor = _build_preprocessor(metadata, ridge_config)

    rf_numeric: Pipeline = rf_preprocessor.transformers[0][1]
    ridge_numeric: Pipeline = ridge_preprocessor.transformers[0][1]

    assert "scaler" not in rf_numeric.named_steps
    assert "scaler" in ridge_numeric.named_steps


def test_modeling_config_reads_tuning_settings() -> None:
    """Parse optional tuning settings and model-specific search grid."""
    config = {
        "feature": "study_tag",
        "modeling": {
            "model_type": "ridge",
            "tuning": {
                "enabled": True,
                "method": "grid_search",
                "inner_cv_folds": 3,
                "max_candidates": 10,
                "grids": {
                    "ridge": {
                        "alpha": [0.1, 1.0],
                    }
                },
            },
        },
    }

    model_cfg = modeling_config_from_analysis(config)

    assert model_cfg.tuning_enabled is True
    assert model_cfg.tuning_method == "grid_search"
    assert model_cfg.tuning_inner_cv_folds == 3
    assert model_cfg.tuning_grids is not None
    assert model_cfg.tuning_grids["ridge"]["alpha"] == [0.1, 1.0]


def test_modeling_config_tuning_requires_grid_for_model() -> None:
    """Require tuning grid for selected model when tuning is enabled."""
    config = {
        "feature": "study_tag",
        "modeling": {
            "model_type": "ridge",
            "tuning": {
                "enabled": True,
                "grids": {
                    "random_forest": {"n_estimators": [100, 200]},
                },
            },
        },
    }

    with pytest.raises(ValueError):
        modeling_config_from_analysis(config)


def test_tuning_grid_helpers_prefix_and_count_candidates() -> None:
    """Prefix tuning keys for pipeline search and count combinations."""
    model_grid = {
        "alpha": [0.1, 1.0],
        "l1_ratio": [0.2, 0.8],
    }

    normalized = _normalize_tuning_grid(model_grid)

    assert "model__alpha" in normalized
    assert "model__l1_ratio" in normalized
    assert _count_grid_candidates(model_grid) == 4


def test_fit_pipeline_with_optional_tuning_runs_grid_search() -> None:
    """Run nested tuning path and return best params for ridge model."""
    metadata = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat": ["A", "A", "B", "B", "C", "C"],
        },
        index=[f"s{i}" for i in range(6)],
    )
    abundance = pd.DataFrame(
        {
            "taxon_1": [0.1, 0.2, 0.25, 0.4, 0.45, 0.6],
            "taxon_2": [0.5, 0.45, 0.4, 0.35, 0.25, 0.2],
        },
        index=metadata.index,
    )
    groups = pd.Series(["G1", "G1", "G2", "G2", "G3", "G3"],
                       index=metadata.index)

    config = ModelingConfig(
        feature_column="study_tag",
        model_type="ridge",
        ridge_params={"alpha": 1.0},
        tuning_enabled=True,
        tuning_inner_cv_folds=3,
        tuning_max_candidates=2,
        tuning_grids={
            "ridge": {
                "alpha": [0.1, 1.0],
            }
        },
    )

    pipeline, best_params, best_score, cv_summary = (
        _fit_pipeline_with_optional_tuning(
            metadata=metadata,
            config=config,
            X_train=metadata,
            y_train=abundance,
            train_groups=groups,
        )
    )

    assert isinstance(pipeline, Pipeline)
    assert best_params["alpha"] in {0.1, 1.0}
    assert best_score is not None
    assert cv_summary is not None
    assert "mean_test_score" in cv_summary.columns


# ---------------------------------------------------------------------------
# Inverse runner tests
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402  (already imported above via sklearn)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from emobon_models.modeling_runner_inverse import (
    _build_classifier_model,
    _build_classification_pipeline,
    _build_regression_pipeline,
    _inverse_fold_metrics,
    _split_metadata_by_dtype,
    run_inverse_group_loocv_with_mlflow,
)


def test_split_metadata_by_dtype_separates_columns() -> None:
    """Split numeric and object-dtype metadata columns correctly."""
    metadata = pd.DataFrame(
        {
            "num1": [1.0, 2.0],
            "num2": [3.0, 4.0],
            "cat1": ["A", "B"],
            "cat2": ["X", "Y"],
        }
    )
    numeric_cols, categorical_cols = _split_metadata_by_dtype(metadata)

    assert set(numeric_cols) == {"num1", "num2"}
    assert set(categorical_cols) == {"cat1", "cat2"}


def test_split_metadata_by_dtype_all_numeric() -> None:
    """Return empty categorical list when all columns are numeric."""
    metadata = pd.DataFrame({"a": [1.0], "b": [2.0]})
    numeric_cols, categorical_cols = _split_metadata_by_dtype(metadata)

    assert set(numeric_cols) == {"a", "b"}
    assert categorical_cols == []


def test_build_classifier_model_random_forest() -> None:
    """Build RandomForestClassifier for random_forest classification type."""
    config = ModelingConfig(
        feature_column="study_tag",
        classification_model_type="random_forest",
        random_forest_classifier_params={
            "n_estimators": 10,
            "random_state": 0,
        },
    )
    clf = _build_classifier_model(config)

    assert isinstance(clf, RandomForestClassifier)
    assert clf.n_estimators == 10


def test_build_classifier_model_logistic_regression() -> None:
    """Build LogisticRegression for logistic_regression classification type."""
    config = ModelingConfig(
        feature_column="study_tag",
        classification_model_type="logistic_regression",
        logistic_regression_params={"max_iter": 200},
    )
    clf = _build_classifier_model(config)

    assert isinstance(clf, LogisticRegression)


def test_build_classifier_model_invalid_type_raises() -> None:
    """Raise ValueError for unsupported classification_model_type."""
    config = ModelingConfig(
        feature_column="study_tag",
    )
    # Force an invalid value bypassing the validator.
    object.__setattr__(
        config, "classification_model_type", "svm"
    )
    with pytest.raises(ValueError, match="Unsupported"):
        _build_classifier_model(config)


def test_build_regression_pipeline_returns_pipeline() -> None:
    """Build regression pipeline with preprocessor and model steps."""
    config = ModelingConfig(
        feature_column="study_tag",
        model_type="random_forest",
        random_forest_params={"n_estimators": 5, "random_state": 0},
    )
    pipeline = _build_regression_pipeline(config)

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_build_classification_pipeline_wraps_lr() -> None:
    """Wrap non-RF classifiers with MultiOutputClassifier."""
    from sklearn.multioutput import MultiOutputClassifier

    config = ModelingConfig(
        feature_column="study_tag",
        classification_model_type="logistic_regression",
    )
    pipeline = _build_classification_pipeline(config)
    model = pipeline.named_steps["model"]

    assert isinstance(model, MultiOutputClassifier)
    assert isinstance(model.estimator, LogisticRegression)


def test_build_classification_pipeline_rf_not_wrapped() -> None:
    """RandomForestClassifier is used directly, not wrapped."""
    config = ModelingConfig(
        feature_column="study_tag",
        classification_model_type="random_forest",
        random_forest_classifier_params={
            "n_estimators": 5,
            "random_state": 0,
        },
    )
    pipeline = _build_classification_pipeline(config)
    model = pipeline.named_steps["model"]

    assert isinstance(model, RandomForestClassifier)


def test_inverse_fold_metrics_numeric_only() -> None:
    """Compute MAE/RMSE for numeric targets; no classification metrics."""
    y_true = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
    )
    y_pred = np.array([[1.1, 4.1], [2.1, 5.1], [3.1, 6.1]])

    metrics = _inverse_fold_metrics(y_true, y_pred, None, None)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "accuracy" not in metrics
    assert "f1_macro" not in metrics
    assert metrics["mae"] == pytest.approx(0.1, abs=1e-6)


def test_inverse_fold_metrics_categorical_only() -> None:
    """Compute accuracy/F1 for categorical targets; no regression metrics."""
    y_true = pd.DataFrame({"c": ["A", "B", "A"]})
    y_pred = np.array([["A"], ["A"], ["A"]])

    metrics = _inverse_fold_metrics(None, None, y_true, y_pred)

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "mae" not in metrics
    assert "rmse" not in metrics


def test_inverse_fold_metrics_mixed() -> None:
    """Compute all four metrics when both numeric and categorical targets."""
    y_true_num = pd.DataFrame({"temp": [10.0, 20.0]})
    y_pred_num = np.array([[10.5], [19.5]])
    y_true_cat = pd.DataFrame({"season": ["winter", "summer"]})
    y_pred_cat = np.array([["winter"], ["summer"]])

    metrics = _inverse_fold_metrics(
        y_true_num, y_pred_num, y_true_cat, y_pred_cat
    )

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "accuracy" in metrics
    assert "f1_macro" in metrics


def test_inverse_fold_metrics_nan_rows_excluded() -> None:
    """Rows with NaN numeric ground-truth are excluded from MAE/RMSE."""
    y_true = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
    y_pred = np.array([[1.0], [999.0], [3.0]])

    metrics = _inverse_fold_metrics(y_true, y_pred, None, None)

    assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)


def test_modeling_config_reads_classification_params() -> None:
    """Parse classification_model_type and classifier params from config."""
    config_dict = {
        "feature": "study_tag",
        "modeling": {
            "classification_model_type": "logistic_regression",
            "logistic_regression": {
                "max_iter": 500,
                "solver": "lbfgs",
            },
        },
    }
    model_cfg = modeling_config_from_analysis(config_dict)

    assert model_cfg.classification_model_type == "logistic_regression"
    assert model_cfg.logistic_regression_params is not None
    assert model_cfg.logistic_regression_params["max_iter"] == 500


def test_run_inverse_group_loocv_with_mlflow(tmp_path: Path) -> None:
    """Run inverse LOOCV end-to-end with mixed numeric and categorical targets.

    Uses a synthetic six-sample dataset where abundance columns are
    correlated with metadata so the model can learn meaningful patterns
    even with very few trees.
    """
    abundance = pd.DataFrame(
        {
            "taxon_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "taxon_2": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "taxon_3": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        },
        index=["s1", "s2", "s3", "s4", "s5", "s6"],
    )
    metadata = pd.DataFrame(
        {
            "study_tag": ["A", "A", "B", "B", "C", "C"],
            "temperature": [
                10.0, 11.0, 20.0, 21.0, 30.0, 31.0
            ],
            "season": [
                "winter", "winter",
                "summer", "summer",
                "spring", "spring",
            ],
        },
        index=["s1", "s2", "s3", "s4", "s5", "s6"],
    )

    config = ModelingConfig(
        feature_column="study_tag",
        model_type="random_forest",
        classification_model_type="random_forest",
        random_forest_params={
            "n_estimators": 10,
            "random_state": 0,
            "n_jobs": 1,
        },
        random_forest_classifier_params={
            "n_estimators": 10,
            "random_state": 0,
            "n_jobs": 1,
        },
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
        mlflow_experiment_name="test-inverse-loocv",
    )

    result = run_inverse_group_loocv_with_mlflow(
        metadata, abundance, config
    )

    assert "fold_metrics" in result
    assert "predictions" in result
    assert "feature_importances" in result
    assert "dropped_metadata_columns" in result

    preds_df = result["predictions"]
    assert isinstance(preds_df, pd.DataFrame)
    assert "true__temperature" in preds_df.columns
    assert "pred__temperature" in preds_df.columns
    assert "true__season" in preds_df.columns
    assert "pred__season" in preds_df.columns

    metrics_df = result["fold_metrics"]
    assert isinstance(metrics_df, pd.DataFrame)
    assert "mae" in metrics_df.columns
    assert "accuracy" in metrics_df.columns
    assert len(metrics_df) == 3  # three LOOCV folds

    importances_df = result["feature_importances"]
    assert isinstance(importances_df, pd.DataFrame)
    assert "target_type" in importances_df.columns
    target_types = set(importances_df["target_type"].unique())
    assert "regression" in target_types
    assert "classification" in target_types

