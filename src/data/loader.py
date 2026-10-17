
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the project's raw-data directory."""
    if Path(filename).name != filename:
        raise ValueError("Provide a filename only, not a directory path.")

    file_path = RAW_DATA_DIR / filename

    if not file_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {file_path}")

    return pd.read_csv(file_path)


def save_processed_csv(
    df: pd.DataFrame,
    filename: str,
) -> Path:
    """Save a DataFrame to the processed-data directory."""
    if Path(filename).name != filename or not filename.lower().endswith(".csv"):
        raise ValueError("Provide a CSV filename only, such as cleaned_data.csv.")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)

    return output_path
