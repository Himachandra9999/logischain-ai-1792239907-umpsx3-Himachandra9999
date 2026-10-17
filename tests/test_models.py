
from src.models import (
    create_random_forest_classifier,
    create_random_forest_regressor,
    create_xgboost_classifier,
    evaluate_classifier,
)


def test_random_forest_classifier_configuration():
    model = create_random_forest_classifier()

    assert model.n_estimators == 200
    assert model.max_depth == 12
    assert model.class_weight == "balanced"


def test_random_forest_regressor_configuration():
    model = create_random_forest_regressor()

    assert model.n_estimators == 200
    assert model.max_depth == 12


def test_xgboost_classifier_configuration():
    model = create_xgboost_classifier()

    assert model.n_estimators == 200
    assert model.max_depth == 5
    assert model.learning_rate == 0.05


def test_classifier_evaluation_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_probability = [0.1, 0.6, 0.8, 0.9]

    results = evaluate_classifier(y_true, y_pred, y_probability)

    assert results["accuracy"] == 0.75
    assert results["recall"] == 1.0
    assert results["roc_auc"] == 1.0
    assert 0 <= results["f1"] <= 1
