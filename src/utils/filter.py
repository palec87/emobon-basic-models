import pandas as pd

from mgnify_methods.utils.logging import get_logger


logger = get_logger(__name__, level="INFO")


def filter_lineage_by_string(
        abundance_df: pd.DataFrame,
        filter_string: str) -> pd.DataFrame:
    """
    Filter abundance table to only include lineages NOT containing a string.

    Args:
        abundance_df: DataFrame with taxonomic lineages as index.
        filter_string: Substring to filter out from lineages.

    Returns:
        Filtered DataFrame with lineages not containing the filter_string.
    """
    logger.info("Filtering lineages containing '%s'", filter_string)
    before_count = len(abundance_df)
    filtered = abundance_df[~abundance_df.index.str.contains(filter_string)]
    logger.info(
        "Filtered lineage table from %d to %d rows",
        before_count,
        len(filtered),
    )
    return filtered
