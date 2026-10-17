
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def create_random_forest_classifier(
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 12,
) -> RandomForestClassifier:
    """Create a Random Forest classifier for binary risk prediction."""
    return RandomForestClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
    )


def create_random_forest_regressor(
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 12,
) -> RandomForestRegressor:
    """Create a Random Forest regressor for continuous-value prediction."""
    return RandomForestRegressor(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
    )
