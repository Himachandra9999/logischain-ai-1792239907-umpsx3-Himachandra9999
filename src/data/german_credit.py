
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

GERMAN_CREDIT_COLUMNS = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
    "credit_risk",
]


def load_german_credit() -> pd.DataFrame:
    """Load the original UCI German Credit dataset."""
    file_path = RAW_DATA_DIR / "german.data"

    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=GERMAN_CREDIT_COLUMNS,
    )

    # UCI labels: 1 = good credit, 2 = bad credit.
    df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

    return df


def save_german_credit() -> Path:
    """Save the labeled dataset as a processed CSV."""
    df = load_german_credit()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / "german_credit_processed.csv"
    df.to_csv(output_path, index=False)

    return output_path
