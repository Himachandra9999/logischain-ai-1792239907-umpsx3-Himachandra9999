
from pathlib import Path

import pandas as pd

from src.data.loader import load_csv, save_processed_csv
from src.data.preprocessing import preprocess_dataframe
from src.data.validation import validate_dataframe


def process_dataset(filename: str) -> dict:
    """Load, clean, validate, and save a raw CSV dataset."""
    raw_df = load_csv(filename)

    cleaned_df = preprocess_dataframe(raw_df)
    report = validate_dataframe(cleaned_df)

    output_name = f"{Path(filename).stem}_processed.csv"
    output_path = save_processed_csv(cleaned_df, output_name)

    return {
        "input_file": filename,
        "output_file": str(output_path),
        "original_rows": len(raw_df),
        "processed_rows": len(cleaned_df),
        "validation_report": report,
    }
