
import pandas as pd

from src.data.german_credit import load_german_credit


def generate_eda_report() -> dict:
    """Generate basic exploratory statistics for the credit dataset."""
    df = load_german_credit()

    numeric_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(exclude="number")

    report = {
        "shape": df.shape,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": numeric_df.describe().round(2),
        "categorical_unique_counts": categorical_df.nunique(),
        "credit_risk_distribution": (
            df["credit_risk"].value_counts().sort_index()
        ),
    }

    return report


if __name__ == "__main__":
    report = generate_eda_report()

    print("DATASET SHAPE:", report["shape"])
    print("MISSING VALUES:", report["missing_values"])
    print("DUPLICATE ROWS:", report["duplicate_rows"])

    print("\nNUMERIC SUMMARY")
    print(report["numeric_summary"])

    print("\nUNIQUE CATEGORICAL VALUES")
    print(report["categorical_unique_counts"])

    print("\nCREDIT RISK DISTRIBUTION")
    print(report["credit_risk_distribution"])
