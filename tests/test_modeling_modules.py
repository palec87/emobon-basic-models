"""Tests for EMOBON modeling module helpers."""

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
