"""Load and clean the Algerian forest fires dataset."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Algerian_forest_fires_dataset_UPDATE.csv"

BASELINE_FEATURES = ("Temperature", "RH", "Ws", "Rain")
FIRE_INDEX_FEATURES = ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI")
TARGET_COLUMN = "Classes"
NUMERIC_COLUMNS = (
    "day",
    "month",
    "year",
    *BASELINE_FEATURES,
    *FIRE_INDEX_FEATURES,
)
VALID_LABELS = ("fire", "not fire")


def load_data(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    feature_columns: tuple[str, ...] = BASELINE_FEATURES,
) -> pd.DataFrame:
    """Return cleaned observations with valid labels and requested features."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path, skiprows=1)
    data.columns = data.columns.str.strip()

    required_columns = {*feature_columns, TARGET_COLUMN}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    data = data[data[TARGET_COLUMN].isin(VALID_LABELS)]
    data = data.dropna(subset=[*feature_columns, TARGET_COLUMN])

    return data.reset_index(drop=True)