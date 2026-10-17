
from pathlib import Path

import numpy as np
import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Summarize common data-quality issues in a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {
            str(column): int(count)
            for column, count in df.isna().sum().items()
            if count > 0
        },
        "infinite_values": {},
    }

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        infinite_counts = np.isinf(numeric_df).sum()
        report["infinite_values"] = {
            str(column): int(count)
            for column, count in infinite_counts.items()
            if count > 0
        }

    return report


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and report basic data-quality information."""
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if csv_path.suffix.lower() != ".csv":
        raise ValueError("Expected a .csv file")

    df = pd.read_csv(csv_path)
    return df
