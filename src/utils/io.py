from pathlib import Path
import pickle
from typing import Any
import mgnify_methods.paper_modules as pm
from mgnify_methods.utils.logging import get_logger


logger = get_logger(__name__, level="INFO")


def load_config(root_dir: Path, name: str) -> dict[str, Any]:
    """Load analysis configuration from the project config file."""
    config_path = root_dir / "configs" / f"{name}.json"
    logger.info("Loading analysis config from %s", config_path)
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
        logger.info("Preprocessing cache not found at %s", paths["dir"])
        return None

    with meta_path.open("rb") as file_obj:
        emobon_meta = pickle.load(file_obj)
    with preprocess_path.open("rb") as file_obj:
        preprocess_tables = pickle.load(file_obj)

    logger.info("Loaded preprocessing cache from %s", paths["dir"])
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

    logger.info("Saved preprocessing cache to %s", cache_dir)
