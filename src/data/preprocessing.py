
import numpy as np
import pandas as pd


def preprocess_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    replace_infinite: bool = True,
) -> pd.DataFrame:
    """Return a cleaned copy of a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    cleaned = df.copy()

    # Normalize infinite numeric values to missing values.
    if replace_infinite:
        numeric_columns = cleaned.select_dtypes(include=[np.number]).columns
        cleaned[numeric_columns] = cleaned[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )

    # Remove exact duplicate rows if requested.
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned
