
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(y_true, y_pred, y_probability=None) -> dict:
    """Calculate standard metrics for a classification model."""
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_probability is not None:
        probabilities = np.asarray(y_probability)
        if len(np.unique(y_true)) == 2:
            results["roc_auc"] = float(
                roc_auc_score(y_true, probabilities)
            )

    return results
