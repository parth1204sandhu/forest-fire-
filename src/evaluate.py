"""Reusable classification metrics for held-out model evaluation."""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test) -> dict[str, object]:
    """Evaluate predictions and fire probabilities on an untouched test set."""
    predictions = model.predict(X_test)
    fire_class_index = list(model.classes_).index(1)
    fire_probabilities = model.predict_proba(X_test)[:, fire_class_index]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, predictions, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, predictions, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, fire_probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "predictions": predictions,
        "fire_probabilities": fire_probabilities,
    }