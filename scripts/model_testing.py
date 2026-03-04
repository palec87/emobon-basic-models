"""Run EMOBON preprocessing and modeling from a script.

This script mirrors the notebook workflow and adds cache-first loading for
preprocessed tables to avoid recomputing expensive preprocessing steps.
"""

from pathlib import Path
import pickle
from typing import Any

import mgnify_methods.paper_modules as pm
from mgnify_methods.metacomp.transforms import apply_transform_method
from mgnify_methods.taxonomy import aggregate_by_taxonomic_level
from mgnify_methods.taxonomy import pivot_taxonomic_data
from mgnify_methods.taxonomy import prevalence_cutoff_abund
from mgnify_methods.taxonomy import remove_singletons_per_sample

from emobon_models.modeling_config import modeling_config_from_analysis
from emobon_models.modeling_runner import run_group_loocv_with_mlflow


def load_config(root_dir: Path) -> dict[str, Any]:
    """Load analysis configuration from the project config file."""
    config_path = root_dir / "configs" / "model_test.json"
    return pm.config_setup(root_dir, config_path)


def cache_file_paths(
    root_dir: Path,
    config: dict[str, Any],
) -> dict[str, Path]:
    """Build file paths used for caching preprocessed outputs."""
    cache_dir_name = config.get(
        "output",
        {},
    ).get("cache_dir", "analysis_cache")
    cache_dir = root_dir / cache_dir_name / "model_testing"
    return {
        "dir": cache_dir,
        "meta": cache_dir / "emobon_meta.pkl",
        "preprocess": cache_dir / "preprocess_tables.pkl",
    }


def load_preprocessed_cache(
    root_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Any] | None:
    """Load cached preprocessed tables and metadata if available."""
    paths = cache_file_paths(root_dir, config)
    meta_path = paths["meta"]
    preprocess_path = paths["preprocess"]

    if not meta_path.exists() or not preprocess_path.exists():
        return None

    with meta_path.open("rb") as file_obj:
        emobon_meta = pickle.load(file_obj)
    with preprocess_path.open("rb") as file_obj:
        preprocess_tables = pickle.load(file_obj)

    print(f"Loaded preprocessing cache from: {paths['dir']}")
    return preprocess_tables, emobon_meta


def save_preprocessed_cache(
    root_dir: Path,
    config: dict[str, Any],
    preprocess_tables: dict[str, Any],
    emobon_meta: Any,
) -> None:
    """Persist preprocessed outputs to cache files."""
    paths = cache_file_paths(root_dir, config)
    cache_dir = paths["dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    with paths["meta"].open("wb") as file_obj:
        pickle.dump(emobon_meta, file_obj)
    with paths["preprocess"].open("wb") as file_obj:
        pickle.dump(preprocess_tables, file_obj)

    print(f"Saved preprocessing cache to: {cache_dir}")


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

    abundance_table_beta = remove_singletons_per_sample(
        abundance_table,
        skip_columns=0,
    )
    abundance_table_beta = prevalence_cutoff_abund(
        abundance_table_beta,
        percent=config["preprocessing"]["prevalence_cutoff"],
        skip_columns=0,
    )

    preprocess_tables: dict[str, Any] = {"emobon": abundance_table_beta}

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
    root_dir = Path(__file__).resolve().parent.parent
    config = load_config(root_dir)

    cached = load_preprocessed_cache(root_dir, config)
    if cached is None:
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
        preprocess_tables, emobon_meta = cached

    modeling_config = modeling_config_from_analysis(config)
    abundance_for_model = preprocess_tables["emobon"]

    modeling_results = run_group_loocv_with_mlflow(
        metadata_df=emobon_meta,
        abundance_df=abundance_for_model,
        config=modeling_config,
    )

    print("Modeling summary metrics:")
    print(modeling_results["summary_metrics"])


if __name__ == "__main__":
    main()
