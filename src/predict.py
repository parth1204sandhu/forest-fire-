"""Load the saved ten-feature baseline and predict fire risk."""

import argparse
import math
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import BASELINE_FEATURES, PROJECT_ROOT, TARGET_MAPPING


MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_baseline.joblib"
INPUT_BOUNDS = {
    "Temperature": (-20.0, 60.0),
    "RH": (0.0, 100.0),
    "Ws": (0.0, 200.0),
    "Rain": (0.0, 500.0),
    "FFMC": (0.0, 101.0),
    "DMC": (0.0, 1000.0),
    "DC": (0.0, 2000.0),
    "ISI": (0.0, 100.0),
    "BUI": (0.0, 1000.0),
    "FWI": (0.0, 100.0),
}
RISK_THRESHOLDS = ((0.25, "LOW"), (0.50, "MODERATE"), (0.75, "HIGH"))


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the model together with its feature and label metadata."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved model not found at {path}. Train it first with: python -m src.train"
        )
    artifact = joblib.load(path)
    required_keys = {"model", "feature_columns", "target_mapping"}
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise ValueError("Saved model artifact is missing inference metadata; retrain it.")
    return artifact


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
    ffmc: float,
    dmc: float,
    dc: float,
    isi: float,
    bui: float,
    fwi: float,
    model=None,
) -> dict[str, float | str]:
    """Return fire probability, predicted class, and prototype risk level."""
    conditions = _validate_conditions(
        {
            "Temperature": temperature,
            "RH": humidity,
            "Ws": wind_speed,
            "Rain": rainfall,
            "FFMC": ffmc,
            "DMC": dmc,
            "DC": dc,
            "ISI": isi,
            "BUI": bui,
            "FWI": fwi,
        }
    )
    artifact = model if isinstance(model, dict) else None
    if artifact is None and model is None:
        artifact = load_model()
    estimator = artifact["model"] if artifact is not None else model
    feature_columns = (
        artifact["feature_columns"]
        if artifact is not None
        else list(getattr(estimator, "feature_names_in_", BASELINE_FEATURES))
    )
    target_mapping = artifact["target_mapping"] if artifact is not None else TARGET_MAPPING
    observation = pd.DataFrame([[conditions[column] for column in feature_columns]], columns=feature_columns)

    fire_class = target_mapping["fire"]
    fire_class_index = list(estimator.classes_).index(fire_class)
    fire_probability = float(estimator.predict_proba(observation)[0][fire_class_index])
    predicted_value = int(estimator.predict(observation)[0])
    predicted_class = next(
        label for label, encoded_value in target_mapping.items() if encoded_value == predicted_value
    )
    return {
        "fire_probability": fire_probability,
        "predicted_class": predicted_class,
        "prediction": predicted_class.upper(),
        "risk_level": classify_risk(fire_probability),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict forest-fire risk.")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature in Celsius")
    parser.add_argument("--humidity", type=float, required=True, help="Relative humidity in percent")
    parser.add_argument("--wind-speed", type=float, required=True, help="Wind speed in km/h")
    parser.add_argument("--rainfall", type=float, required=True, help="Rainfall in mm")
    parser.add_argument("--ffmc", type=float, required=True, help="Fine Fuel Moisture Code")
    parser.add_argument("--dmc", type=float, required=True, help="Duff Moisture Code")
    parser.add_argument("--dc", type=float, required=True, help="Drought Code")
    parser.add_argument("--isi", type=float, required=True, help="Initial Spread Index")
    parser.add_argument("--bui", type=float, required=True, help="Build Up Index")
    parser.add_argument("--fwi", type=float, required=True, help="Fire Weather Index")
    arguments = parser.parse_args()

    try:
        prediction = predict_fire_risk(
            arguments.temperature,
            arguments.humidity,
            arguments.wind_speed,
            arguments.rainfall,
            arguments.ffmc,
            arguments.dmc,
            arguments.dc,
            arguments.isi,
            arguments.bui,
            arguments.fwi,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Fire probability: {prediction['fire_probability']:.2f}")
    print(f"Predicted class: {prediction['predicted_class']}")
    print(f"Prediction: {prediction['prediction']}")
    print(f"Risk level: {prediction['risk_level']}")


if __name__ == "__main__":
    main()