"""Train and evaluate the ten-feature Random Forest baseline."""

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data_loader import (
    BASELINE_FEATURES,
    PROJECT_ROOT,
    TARGET_MAPPING,
    load_data,
)
from src.evaluate import evaluate_model


RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_baseline.joblib"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "baseline_feature_importance.png"


def make_train_test_split(data=None):
    """Create the fixed, stratified holdout split used by all baselines."""
    data = load_data() if data is None else data
    features = data.loc[:, BASELINE_FEATURES]
    target = data["Classes"].map(TARGET_MAPPING).astype(int)
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def create_random_forest() -> RandomForestClassifier:
    """Create the primary, reproducible Random Forest baseline."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def train_baseline():
    """Fit the baseline and return it with holdout data and predictions."""
    data = load_data()
    X_train, X_test, y_train, y_test = make_train_test_split(data)
    model = create_random_forest()
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    return model, (X_test, y_test, metrics["predictions"]), metrics


def plot_feature_importance(model) -> list[tuple[str, float]]:
    """Save and return the baseline model's ranked feature importances."""
    ranked_importances = sorted(
        zip(BASELINE_FEATURES, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    features, importances = zip(*ranked_importances)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.barh(features, importances, color="#287c70")
    axis.invert_yaxis()
    axis.set_xlabel("Mean decrease in impurity")
    axis.set_title("Baseline Random Forest Feature Importance")
    figure.tight_layout()
    figure.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    plt.close(figure)
    return ranked_importances


def build_model_artifact(model) -> dict[str, object]:
    """Bundle the estimator with the preprocessing metadata inference needs."""
    return {
        "model": model,
        "feature_columns": list(BASELINE_FEATURES),
        "target_mapping": TARGET_MAPPING,
        "missing_feature_strategy": "drop rows missing any baseline feature or target",
    }


def main() -> None:
    model, (X_test, y_test, predictions), metrics = train_baseline()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(build_model_artifact(model), MODEL_PATH)
    data = load_data()
    print(f"Clean observations: {len(data)}")
    print(f"Class distribution: {data['Classes'].value_counts().to_dict()}")
    print(f"Features: {', '.join(BASELINE_FEATURES)}")
    print(f"Training rows: {len(data) - len(X_test)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision (fire): {metrics['precision']:.3f}")
    print(f"Recall (fire): {metrics['recall']:.3f}")
    print(f"F1 score (fire): {metrics['f1']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print("\nConfusion matrix (actual rows, predicted columns; not fire, fire):")
    print(metrics["confusion_matrix"])
    probability_preview = X_test.loc[:, []].copy()
    probability_preview["actual"] = y_test.map({0: "not fire", 1: "fire"})
    probability_preview["predicted"] = ["fire" if value else "not fire" for value in predictions]
    probability_preview["fire_probability"] = metrics["fire_probabilities"]
    print("\nFirst ten test-set probability predictions:")
    print(probability_preview.head(10).to_string(index=False))
    print(f"\nSaved model: {MODEL_PATH}")
    print("\nFeature importance (model-specific, not causal):")
    print("Correlated fire-weather indices can share or redistribute importance.")
    for feature, importance in plot_feature_importance(model):
        print(f"{feature}: {importance:.3f}")
    print(f"Saved feature-importance plot: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()