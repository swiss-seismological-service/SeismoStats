import pandas as pd


def _check_required_cols(df: pd.DataFrame,
                         required_cols: list[str]):
    """
    Check if a DataFrame has the required columns.

    Args:
        df : pandas DataFrame
            DataFrame to check.
            required_cols : list of str
            List of required columns.

    Returns:
        bool
            True if the DataFrame has the required columns, False otherwise.
    """

    if not set(required_cols).issubset(set(df.columns)):
        return False
    return True
