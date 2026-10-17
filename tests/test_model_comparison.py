
from src.models.compare_models import compare_models


def test_model_comparison_returns_expected_metrics():
    results = compare_models()

    assert len(results) == 2
    assert set(results["model"]) == {
        "Random Forest",
        "Gradient Boosting",
    }

    expected_metrics = {
        "accuracy",
        "roc_auc",
        "bad_credit_precision",
        "bad_credit_recall",
        "bad_credit_f1",
    }

    assert expected_metrics.issubset(results.columns)

    for metric in expected_metrics:
        assert results[metric].between(0, 1).all()
