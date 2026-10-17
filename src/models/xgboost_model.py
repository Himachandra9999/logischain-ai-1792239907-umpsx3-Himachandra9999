
from xgboost import XGBClassifier


def create_xgboost_classifier(
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
) -> XGBClassifier:
    """Create an XGBoost classifier for binary risk prediction."""
    return XGBClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )
