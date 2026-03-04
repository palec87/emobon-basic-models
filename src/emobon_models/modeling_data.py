"""Data preparation helpers for metadata-to-abundance models."""

from dataclasses import dataclass

import pandas as pd

from emobon_models.modeling_config import ModelingConfig


@dataclass(slots=True)
class ModelingDataset:
    """Prepared data structures used by model training and evaluation."""

    metadata: pd.DataFrame
    abundance: pd.DataFrame
    groups: pd.Series
    sample_ids: pd.Index
    dropped_metadata_columns: list[str]


def _with_sample_index(
    frame: pd.DataFrame,
    sample_id_column: str | None,
) -> pd.DataFrame:
    """Return a dataframe indexed by sample identifier."""
    if sample_id_column is None:
        return frame.copy()
    if sample_id_column not in frame.columns:
        msg = f"sample_id_column '{sample_id_column}' not present in dataframe"
        raise KeyError(msg)

    indexed = frame.set_index(sample_id_column, drop=True)
    return indexed


def _drop_duplicate_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate sample identifiers while preserving first records."""
    return frame.loc[~frame.index.duplicated(keep="first")]


def _filter_missing_metadata_columns(
    metadata: pd.DataFrame,
    missing_column_threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop metadata columns whose missingness exceeds configured threshold."""
    missing_ratio = metadata.isna().mean(axis=0)
    dropped = (
        missing_ratio[missing_ratio > missing_column_threshold]
        .index
        .tolist()
    )
    filtered = metadata.drop(columns=dropped)
    return filtered, dropped


def prepare_modeling_dataset(
    metadata_df: pd.DataFrame,
    abundance_df: pd.DataFrame,
    config: ModelingConfig,
) -> ModelingDataset:
    """Align metadata and abundance tables and apply modeling filters."""
    metadata = _with_sample_index(metadata_df, config.sample_id_column)
    abundance = _with_sample_index(abundance_df, config.sample_id_column)

    metadata = _drop_duplicate_samples(metadata)
    abundance = _drop_duplicate_samples(abundance)

    shared_ids = metadata.index.intersection(abundance.index)
    if shared_ids.empty:
        msg = (
            "No shared sample identifiers between metadata and "
            "abundance tables"
        )
        raise ValueError(msg)

    metadata = metadata.loc[shared_ids]
    abundance = abundance.loc[shared_ids]

    if config.feature_column not in metadata.columns:
        msg = (
            f"feature column '{config.feature_column}' not "
            "present in metadata"
        )
        raise KeyError(msg)

    metadata = metadata.loc[~metadata[config.feature_column].isna()]
    abundance = abundance.loc[metadata.index]

    filtered_metadata, dropped = _filter_missing_metadata_columns(
        metadata,
        config.missing_column_threshold,
    )

    if filtered_metadata.empty or abundance.empty:
        msg = "No rows available after metadata and abundance alignment"
        raise ValueError(msg)

    groups = filtered_metadata[config.feature_column].astype("string")
    return ModelingDataset(
        metadata=filtered_metadata,
        abundance=abundance,
        groups=groups,
        sample_ids=filtered_metadata.index,
        dropped_metadata_columns=dropped,
    )
