"""Load the saved baseline model and predict fire risk for one observation."""

import argparse
import math
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import BASELINE_FEATURES, PROJECT_ROOT


MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_baseline.joblib"
INPUT_BOUNDS = {
    "temperature": (-20.0, 60.0),
    "humidity": (0.0, 100.0),
    "wind_speed": (0.0, 200.0),
    "rainfall": (0.0, 500.0),
}
RISK_THRESHOLDS = ((0.25, "LOW"), (0.50, "MODERATE"), (0.75, "HIGH"))


def load_model(model_path: str | Path = MODEL_PATH):
    """Load a model previously saved by the training script."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved model not found at {path}. Train it first with: python -m src.train"
        )
    return joblib.load(path)


def classify_risk(fire_probability: float) -> str:
    """Map probability to prototype UI bands; these are not validated warnings."""
    if not math.isfinite(fire_probability) or not 0.0 <= fire_probability <= 1.0:
        raise ValueError("Fire probability must be a finite number from 0 to 1.")
    for threshold, risk_level in RISK_THRESHOLDS:
        if fire_probability < threshold:
            return risk_level
    return "EXTREME"


def _validate_conditions(conditions: dict[str, float]) -> dict[str, float]:
    validated = {}
    for name, (minimum, maximum) in INPUT_BOUNDS.items():
        value = conditions[name]
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a number.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a number.") from error
        if not math.isfinite(numeric_value) or not minimum <= numeric_value <= maximum:
            raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}.")
        validated[name] = numeric_value
    return validated


def predict_fire_risk(
    temperature: float,
    humidity: float,
    wind_speed: float,
    rainfall: float,
    model=None,
) -> dict[str, float | str]:
    """Return fire probability, predicted class, and prototype risk level."""
    conditions = _validate_conditions(
        {
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "rainfall": rainfall,
        }
    )
    model = model if model is not None else load_model()
    observation = pd.DataFrame(
        [[
            conditions["temperature"],
            conditions["humidity"],
            conditions["wind_speed"],
            conditions["rainfall"],
        ]],
        columns=BASELINE_FEATURES,
    )

    fire_class_index = list(model.classes_).index(1)
    fire_probability = float(model.predict_proba(observation)[0][fire_class_index])
    predicted_class = "fire" if int(model.predict(observation)[0]) == 1 else "not fire"
    return {
        "fire_probability": fire_probability,
        "predicted_class": predicted_class,
        "risk_level": classify_risk(fire_probability),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict forest-fire risk.")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature in Celsius")
    parser.add_argument("--humidity", type=float, required=True, help="Relative humidity in percent")
    parser.add_argument("--wind-speed", type=float, required=True, help="Wind speed in km/h")
    parser.add_argument("--rainfall", type=float, required=True, help="Rainfall in mm")
    arguments = parser.parse_args()

    try:
        prediction = predict_fire_risk(
            arguments.temperature,
            arguments.humidity,
            arguments.wind_speed,
            arguments.rainfall,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Fire probability: {prediction['fire_probability']:.2f}")
    print(f"Predicted class: {prediction['predicted_class']}")
    print(f"Risk level: {prediction['risk_level']}")


if __name__ == "__main__":
    main()