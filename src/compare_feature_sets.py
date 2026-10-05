"""Compare raw weather with an exploratory fire-weather-index feature set."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data_loader import BASELINE_FEATURES, FIRE_INDEX_FEATURES, load_data
from src.train import RANDOM_STATE, TEST_SIZE


def compare_feature_sets() -> dict[str, dict[str, object]]:
    """Compare both feature sets on identical observations and holdout rows."""
    all_features = (*BASELINE_FEATURES, *FIRE_INDEX_FEATURES)
    data = load_data(feature_columns=all_features)
    target = data["Classes"].map({"not fire": 0, "fire": 1})
    train_indices, test_indices = train_test_split(
        data.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    feature_sets = {
        "Raw weather baseline": BASELINE_FEATURES,
        "Exploratory weather + fire indices": all_features,
    }
    results = {"observations": len(data)}
    for name, features in feature_sets.items():
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        model.fit(data.loc[train_indices, features], target.loc[train_indices])
        predictions = model.predict(data.loc[test_indices, features])
        actual = target.loc[test_indices]
        results[name] = {
            "accuracy": accuracy_score(actual, predictions),
            "precision": precision_score(actual, predictions, zero_division=0),
            "recall": recall_score(actual, predictions, zero_division=0),
            "f1": f1_score(actual, predictions, zero_division=0),
            "confusion_matrix": confusion_matrix(actual, predictions, labels=[0, 1]),
        }
    return results


def main() -> None:
    results = compare_feature_sets()
    print(f"Same complete observations for both experiments: {results['observations']}")
    for name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        print(f"\n{name}")
        for metric in ("accuracy", "precision", "recall", "f1"):
            print(f"  {metric}: {metrics[metric]:.3f}")
        print("  confusion matrix (actual rows, predicted columns):")
        print(metrics["confusion_matrix"])
    print(
        "\nCaution: fire-weather indices encode related weather-danger information. "
        "This experiment is not independent evidence of real-world forecasting ability."
    )


if __name__ == "__main__":
    main()