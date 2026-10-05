"""Compare raw weather with an exploratory fire-weather-index feature set."""

from src.data_loader import BASELINE_FEATURES, WEATHER_FEATURES, load_data
from src.evaluate import evaluate_model
from src.train import create_random_forest, make_train_test_split


def compare_feature_sets() -> dict[str, dict[str, object]]:
    """Compare both feature sets on identical observations and holdout rows."""
    data = load_data()
    X_train, X_test, y_train, y_test = make_train_test_split(data)

    feature_sets = {
        "Weather-only experiment": WEATHER_FEATURES,
        "Weather + fire-weather indices": BASELINE_FEATURES,
    }
    results = {"observations": len(data)}
    for name, features in feature_sets.items():
        model = create_random_forest()
        model.fit(X_train.loc[:, features], y_train)
        results[name] = evaluate_model(model, X_test.loc[:, features], y_test)
    return results


def main() -> None:
    results = compare_feature_sets()
    print(f"Same complete observations for both experiments: {results['observations']}")
    for name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        print(f"\n{name}")
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            print(f"  {metric}: {metrics[metric]:.3f}")
        print("  confusion matrix (actual rows, predicted columns):")
        print(metrics["confusion_matrix"])
    print(
        "\nCaution: fire-weather indices encode related weather-danger information. "
        "This experiment is not independent evidence of real-world forecasting ability."
    )


if __name__ == "__main__":
    main()