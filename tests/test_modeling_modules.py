"""Tests for EMOBON modeling module helpers."""

import pandas as pd

from emobon_models.modeling_config import modeling_config_from_analysis
from emobon_models.modeling_data import prepare_modeling_dataset
from emobon_models.modeling_runner import _group_loocv_masks


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
