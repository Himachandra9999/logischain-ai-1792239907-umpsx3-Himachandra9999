
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import preprocess_dataframe
from src.data.validation import validate_dataframe


def test_validation_reports_missing_and_duplicate_rows():
    df = pd.DataFrame({
        "amount": [100.0, 100.0, np.nan],
        "risk": [0, 0, 1],
    })

    report = validate_dataframe(df)

    assert report["rows"] == 3
    assert report["columns"] == 2
    assert report["duplicate_rows"] == 1
    assert report["missing_values"]["amount"] == 1


def test_preprocessing_removes_duplicates_and_infinities():
    df = pd.DataFrame({
        "amount": [100.0, 100.0, np.inf, -np.inf],
        "risk": [0, 0, 1, 0],
    })

    cleaned = preprocess_dataframe(df)

    assert len(cleaned) == 3
    assert not np.isinf(
        cleaned.select_dtypes(include=[np.number])
    ).any().any()
    assert cleaned["amount"].isna().sum() == 2


def test_preprocessing_does_not_modify_original_dataframe():
    df = pd.DataFrame({"amount": [100.0, np.inf]})

    cleaned = preprocess_dataframe(df)

    assert np.isinf(df["amount"]).sum() == 1
    assert cleaned["amount"].isna().sum() == 1


def test_validation_rejects_non_dataframe_input():
    with pytest.raises(TypeError):
        validate_dataframe([100, 200])
