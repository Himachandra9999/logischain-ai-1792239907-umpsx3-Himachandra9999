
import json
from pathlib import Path


def test_saved_credit_model_metrics_are_valid():
    metrics_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "german_credit_baseline.json"
    )

    assert metrics_path.is_file(), (
        "Run python -m src.models.train_credit_model first."
    )

    with metrics_path.open(encoding="utf-8") as file:
        results = json.load(file)

    assert results["dataset"] == "UCI German Credit"
    assert results["test_records"] == 200
    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["roc_auc"] <= 1

    matrix = results["confusion_matrix"]
    assert len(matrix) == 2
    assert all(len(row) == 2 for row in matrix)
    assert sum(sum(row) for row in matrix) == 200
