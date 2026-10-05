"""Train and evaluate the raw-weather Random Forest baseline."""

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data_loader import BASELINE_FEATURES, PROJECT_ROOT, load_data


RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_baseline.joblib"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "baseline_feature_importance.png"


def train_baseline():
    """Fit the baseline and return it with holdout data and predictions."""
    data = load_data()
    features = data.loc[:, BASELINE_FEATURES]
    target = data["Classes"].map({"not fire": 0, "fire": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["not fire", "fire"],
            zero_division=0,
        ),
    }
    return model, (X_test, y_test, predictions), metrics


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


def main() -> None:
    model, (X_test, y_test, predictions), metrics = train_baseline()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Training rows: {len(load_data()) - len(X_test)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision (fire): {metrics['precision']:.3f}")
    print(f"Recall (fire): {metrics['recall']:.3f}")
    print(f"F1 score (fire): {metrics['f1']:.3f}")
    print("\nConfusion matrix (actual rows, predicted columns; not fire, fire):")
    print(metrics["confusion_matrix"])
    print("\nClassification report:")
    print(metrics["classification_report"])
    print(f"\nSaved model: {MODEL_PATH}")
    print("\nFeature importance (model-specific, not causal):")
    for feature, importance in plot_feature_importance(model):
        print(f"{feature}: {importance:.3f}")
    print(f"Saved feature-importance plot: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()