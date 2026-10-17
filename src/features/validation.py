
import numpy as np
import pandas as pd


def validate_features(df: pd.DataFrame) -> dict:
    """Return a quality report for an engineered feature DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    numeric_df = df.select_dtypes(include=[np.number])

    infinite_values = {}
    if not numeric_df.empty:
        counts = np.isinf(numeric_df).sum()
        infinite_values = {
            str(column): int(count)
            for column, count in counts.items()
            if count > 0
        }

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {
            str(column): int(count)
            for column, count in df.isna().sum().items()
            if count > 0
        },
        "infinite_values": infinite_values,
        "valid": not bool(infinite_values),
    }
