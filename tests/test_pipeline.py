import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app
from src.data_loader import BASELINE_FEATURES, load_data
from src.predict import classify_risk, predict_fire_risk
from src.train import train_baseline


class DataLoadingTests(unittest.TestCase):
    def test_real_dataset_has_clean_labels_and_baseline_features(self):
        data = load_data()

        self.assertEqual(len(data), 243)
        self.assertEqual(data["Classes"].value_counts().to_dict(), {"fire": 137, "not fire": 106})
        self.assertTrue(data[list(BASELINE_FEATURES)].notna().all().all())
        self.assertTrue(all(data[feature].dtype.kind in "if" for feature in BASELINE_FEATURES))

    def test_loader_normalizes_labels_and_drops_invalid_rows(self):
        csv_content = (
            "Region title\n"
            " Temperature , RH , Ws , Rain , Classes \n"
            "32,40,15,0, Fire \n"
            "30,50,10,0,not fire\n"
            "bad,50,10,0,fire\n"
            "30,50,10,,fire\n"
            "30,50,10,0,unknown\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "sample.csv"
            csv_path.write_text(csv_content, encoding="utf-8")
            data = load_data(csv_path)

        self.assertEqual(len(data), 2)
        self.assertEqual(data["Classes"].tolist(), ["fire", "not fire"])
        self.assertEqual(data["Temperature"].dtype.kind, "f")


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _, _ = train_baseline()

    def test_prediction_returns_probability_class_and_risk(self):
        result = predict_fire_risk(34, 31, 18, 0, model=self.model)

        self.assertGreaterEqual(result["fire_probability"], 0)
        self.assertLessEqual(result["fire_probability"], 1)
        self.assertIn(result["predicted_class"], {"fire", "not fire"})
        self.assertIn(result["risk_level"], {"LOW", "MODERATE", "HIGH", "EXTREME"})

    def test_invalid_conditions_are_rejected(self):
        with self.assertRaises(ValueError):
            predict_fire_risk(34, 101, 18, 0, model=self.model)
        with self.assertRaises(ValueError):
            predict_fire_risk(float("nan"), 31, 18, 0, model=self.model)

    def test_risk_bands_cover_probability_boundaries(self):
        self.assertEqual(classify_risk(0.0), "LOW")
        self.assertEqual(classify_risk(0.25), "MODERATE")
        self.assertEqual(classify_risk(0.5), "HIGH")
        self.assertEqual(classify_risk(0.75), "EXTREME")
        with self.assertRaises(ValueError):
            classify_risk(1.1)

    def test_api_returns_prediction_and_rejects_invalid_input(self):
        client = TestClient(app)
        with patch("src.api.get_model", return_value=self.model):
            valid_response = client.post(
                "/predict",
                json={
                    "temperature": 34,
                    "humidity": 31,
                    "wind_speed": 18,
                    "rainfall": 0,
                },
            )
            invalid_response = client.post(
                "/predict",
                json={
                    "temperature": 34,
                    "humidity": 101,
                    "wind_speed": 18,
                    "rainfall": 0,
                },
            )

        self.assertEqual(valid_response.status_code, 200)
        self.assertIn("fire_probability", valid_response.json())
        self.assertEqual(invalid_response.status_code, 422)


if __name__ == "__main__":
    unittest.main()