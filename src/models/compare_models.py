
import json
from pathlib import Path

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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
EVALUATION_DIR = PROJECT_ROOT / "evaluation"


def compare_models() -> pd.DataFrame:
    """Compare two classifiers using an identical held-out test set."""
    df = load_german_credit()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Random Forest": create_random_forest_classifier(),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
        ),
    }

    results = []

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", create_preprocessor(X_train)),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "bad_credit_precision": report["1"]["precision"],
            "bad_credit_recall": report["1"]["recall"],
            "bad_credit_f1": report["1"]["f1-score"],
        })

    results_df = pd.DataFrame(results).round(4)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVALUATION_DIR / "model_comparison.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results_df.to_dict(orient="records"), file, indent=2)

    return results_df


if __name__ == "__main__":
    comparison = compare_models()
    print("MODEL COMPARISON — UCI GERMAN CREDIT")
    print(comparison.to_string(index=False))
    print("\nResults saved to: evaluation/model_comparison.json")
