"""Compare simple Logistic Regression and Random Forest baselines."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import BASELINE_FEATURES, load_data
from src.train import RANDOM_STATE, TEST_SIZE


def compare_models() -> dict[str, dict[str, object]]:
    """Evaluate both estimators with the same stratified holdout split."""
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

    estimators = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
    }
    results = {}
    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        }
    return results


def main() -> None:
    for name, metrics in compare_models().items():
        print(f"{name}")
        for metric in ("accuracy", "precision", "recall", "f1"):
            print(f"  {metric}: {metrics[metric]:.3f}")
        print("  confusion matrix (actual rows, predicted columns):")
        print(metrics["confusion_matrix"])


if __name__ == "__main__":
    main()