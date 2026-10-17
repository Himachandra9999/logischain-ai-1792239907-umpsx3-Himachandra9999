
import pandas as pd

from src.models.preprocessing import (
    create_preprocessor,
    split_features_target,
)


def test_split_features_target_excludes_target():
    df = pd.DataFrame({
        "age_years": [25, 40, 35],
        "purpose": ["A40", "A41", "A40"],
        "credit_risk": [0, 1, 0],
    })

    X, y = split_features_target(df)

    assert "credit_risk" not in X.columns
    assert len(X) == len(y) == 3
    assert y.tolist() == [0, 1, 0]


def test_preprocessor_encodes_categorical_features():
    X = pd.DataFrame({
        "age_years": [25, 40, 35],
        "purpose": ["A40", "A41", "A40"],
    })

    preprocessor = create_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] == 3
    assert preprocessor.transform(
        pd.DataFrame({
            "age_years": [30],
            "purpose": ["UNKNOWN"],
        })
    ).shape[1] == 3


def test_missing_target_column_raises_error():
    df = pd.DataFrame({"age_years": [25, 40]})

    try:
        split_features_target(df)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for missing target")
