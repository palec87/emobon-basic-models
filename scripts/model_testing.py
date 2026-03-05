"""Run EMOBON preprocessing and modeling from a script.

This script mirrors the notebook workflow and adds cache-first loading for
preprocessed tables to avoid recomputing expensive preprocessing steps.
"""

from pathlib import Path
from typing import Any
from utils.io import (
    load_config,
    load_preprocessed_cache,
    save_preprocessed_cache,
)

import mgnify_methods.paper_modules as pm
from mgnify_methods.metacomp.transforms import apply_transform_method
from mgnify_methods.taxonomy import (
    aggregate_by_taxonomic_level,
    pivot_taxonomic_data, prevalence_cutoff_abund,
    remove_singletons_per_sample
)
from emobon_models.modeling_config import modeling_config_from_analysis
from emobon_models.modeling_runner import run_group_loocv_with_mlflow
from utils.filter import filter_lineage_by_string

from mgnify_methods.utils.logging import get_logger
logger = get_logger(__name__, level="INFO")


def build_preprocessed_tables(
    root_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Run raw-data loading and preprocessing to build model inputs."""
    abundance_emobon, emobon_meta = pm.load_emobon(root_dir, ret="ssu")
    abundance_emobon = pm.process_emobon_data(abundance_emobon, config)

    taxonomy_table = pm.clean_abundance_table(abundance_emobon, config)
    tax_level = config["taxonomy"]["analysis_level"]
    abundance_table = aggregate_by_taxonomic_level(
        taxonomy_table.df,
        level=tax_level,
        dropna=config["preprocessing"]["dropna"],
    )

    abundance_table = pivot_taxonomic_data(abundance_table)

    abundance_table = remove_singletons_per_sample(
        abundance_table,
        skip_columns=0,
    )
    logger.info(f'before filtering: {abundance_table.shape}')
    # remove unclassified sequences such as
    # 'sk__Archaea;k__;p__;c__;o__;f__;g__;s__'
    string_to_filter = config["taxonomy"]["indicators"][tax_level] + ';'
    abundance_table = filter_lineage_by_string(
        abundance_table,
        string_to_filter,
    )
    logger.info(f'after filtering: {abundance_table.shape}')

    abundance_table = prevalence_cutoff_abund(
        abundance_table,
        percent=config["preprocessing"]["prevalence_cutoff"],
        skip_columns=0,
    )

    preprocess_tables: dict[str, Any] = {"emobon": abundance_table}

    if config["transform"]["enabled"]:
        transformed_tables: dict[str, Any] = {}
        for sample_type, table_or_dict in preprocess_tables.items():
            if isinstance(table_or_dict, dict):
                transformed_tables[sample_type] = {
                    tax_level: apply_transform_method(df, config)
                    for tax_level, df in table_or_dict.items()
                }
            else:
                transformed_tables[sample_type] = apply_transform_method(
                    table_or_dict,
                    config,
                )
        preprocess_tables = transformed_tables

    return preprocess_tables, emobon_meta


def main() -> None:
    """Execute cache-aware preprocessing and group-LOOCV modeling."""
    logger.info("Starting model testing workflow")
    root_dir = Path(__file__).resolve().parent.parent
    config = load_config(root_dir)
    logger.info("Configuration loaded")

    cached = load_preprocessed_cache(root_dir, config)
    if cached is None:
        logger.info("No preprocessing cache found; building tables")
        preprocess_tables, emobon_meta = build_preprocessed_tables(
            root_dir,
            config,
        )
        save_preprocessed_cache(
            root_dir,
            config,
            preprocess_tables,
            emobon_meta,
        )
    else:
        logger.info("Using cached preprocessed tables")
        preprocess_tables, emobon_meta = cached

    modeling_config = modeling_config_from_analysis(config)
    abundance_for_model = preprocess_tables["emobon"]
    logger.info("Selected model type: %s", modeling_config.model_type)
    logger.info("Tuning enabled: %s", modeling_config.tuning_enabled)

    # filter metadata manually
    emobon_meta = emobon_meta[config['modeling']['metadata_cols']]
    logger.info("Filtered metadata columns to shape: %s", emobon_meta.shape)

    modeling_results = run_group_loocv_with_mlflow(
        metadata_df=emobon_meta,
        abundance_df=abundance_for_model,
        config=modeling_config,
    )

    print("Modeling summary metrics:")
    print(modeling_results["summary_metrics"])
    logger.info("Model testing workflow completed")


if __name__ == "__main__":
    main()
