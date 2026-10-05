"""Compare Logistic Regression with the primary Random Forest baseline."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data
from src.evaluate import evaluate_model
from src.train import RANDOM_STATE, create_random_forest, make_train_test_split


def compare_models() -> dict[str, dict[str, object]]:
    """Evaluate both estimators on the same ten-feature stratified split."""
    data = load_data()
    X_train, X_test, y_train, y_test = make_train_test_split(data)

    estimators = {
        "Random Forest": create_random_forest(),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
    }
    results = {}
    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        results[name] = evaluate_model(estimator, X_test, y_test)
    return results


def main() -> None:
    for name, metrics in compare_models().items():
        print(f"{name}")
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            print(f"  {metric}: {metrics[metric]:.3f}")
        print("  confusion matrix (actual rows, predicted columns):")
        print(metrics["confusion_matrix"])


if __name__ == "__main__":
    main()