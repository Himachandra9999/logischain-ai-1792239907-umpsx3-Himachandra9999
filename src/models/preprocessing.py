
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


TARGET_COLUMN = "credit_risk"


def create_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """Build a preprocessing transformer from the input feature columns."""
    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include="number"
    ).columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
    )


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
):
    """Separate predictors from the target, preventing target leakage."""
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y
