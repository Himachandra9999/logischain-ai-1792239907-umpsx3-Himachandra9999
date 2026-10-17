
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.german_credit import load_german_credit
from src.models.preprocessing import (
    create_preprocessor,
    split_features_target,
)
from src.models.random_forest import create_random_forest_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"


def train_credit_model() -> dict:
    """Train and evaluate a baseline German Credit classifier."""
    df = load_german_credit()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("preprocessor", create_preprocessor(X_train)),
        ("classifier", create_random_forest_classifier()),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "dataset": "UCI German Credit",
        "test_records": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["Good credit", "Bad credit"],
            output_dict=True,
            zero_division=0,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "german_credit_rf.joblib")

    with open(
        EVALUATION_DIR / "german_credit_baseline.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    return results


if __name__ == "__main__":
    results = train_credit_model()

    print("Dataset:", results["dataset"])
    print("Test records:", results["test_records"])
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print("Confusion matrix:")
    print(results["confusion_matrix"])
    print("\nClassification report:")
    print(pd.DataFrame(results["classification_report"]).T.round(3))
    print("\nModel saved to: models/german_credit_rf.joblib")
    print("Metrics saved to: evaluation/german_credit_baseline.json")
