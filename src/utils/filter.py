import pandas as pd


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
    filtered = abundance_df[~abundance_df.index.str.contains(filter_string)]
    return filtered
