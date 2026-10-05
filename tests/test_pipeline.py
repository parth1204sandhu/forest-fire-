import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
from fastapi.testclient import TestClient

from src.api import app
from src.data_loader import BASELINE_FEATURES, TARGET_MAPPING, WEATHER_FEATURES, load_data
from src.predict import classify_risk, load_model, predict_fire_risk
from src.train import build_model_artifact, train_baseline


class DataLoadingTests(unittest.TestCase):
    def test_real_dataset_has_clean_labels_and_baseline_features(self):
        data = load_data()

        self.assertEqual(len(data), 243)
        self.assertEqual(data["Classes"].value_counts().to_dict(), {"fire": 137, "not fire": 106})
        self.assertTrue(data[list(BASELINE_FEATURES)].notna().all().all())
        self.assertTrue(all(data[feature].dtype.kind in "if" for feature in BASELINE_FEATURES))

    def test_loader_normalizes_labels_and_drops_invalid_rows(self):
        csv_content = (
            "Bejaia Region Dataset\n"
            "day, Temperature , RH , Ws , Rain , FFMC , DMC , DC , ISI , BUI , FWI , Classes \n"
            "01,32,40,15,0,80,3,8,1,4,1, Fire \n"
            "02,30,50,10,0,70,2,7,0.5,3,0.5,not fire\n"
            "03,bad,50,10,0,80,3,8,1,4,1,fire\n"
            "04,30,50,10,,80,3,8,1,4,1,fire\n"
            "05,30,50,10,0,80,3,8,1,4,1,unknown\n"
            "Sidi-Bel Abbes Region Dataset\n"
            "day,Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,FWI,Classes\n"
            "01,31,55,14,0,86,8,18,5,8,5,NOT FIRE\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "sample.csv"
            csv_path.write_text(csv_content, encoding="utf-8")
            data = load_data(csv_path, feature_columns=BASELINE_FEATURES)

        self.assertEqual(len(data), 3)
        self.assertEqual(data["Classes"].tolist(), ["fire", "not fire", "not fire"])
        self.assertEqual(data["Temperature"].dtype.kind, "f")


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _, _ = train_baseline()

    def test_prediction_returns_probability_class_and_risk(self):
        result = predict_fire_risk(34, 31, 18, 0, 80, 20, 100, 5, 25, 6, model=self.model)

        self.assertGreaterEqual(result["fire_probability"], 0)
        self.assertLessEqual(result["fire_probability"], 1)
        self.assertIn(result["predicted_class"], {"fire", "not fire"})
        self.assertIn(result["risk_level"], {"LOW", "MODERATE", "HIGH", "EXTREME"})

    def test_invalid_conditions_are_rejected(self):
        with self.assertRaises(ValueError):
            predict_fire_risk(34, 101, 18, 0, 80, 20, 100, 5, 25, 6, model=self.model)
        with self.assertRaises(ValueError):
            predict_fire_risk(float("nan"), 31, 18, 0, 80, 20, 100, 5, 25, 6, model=self.model)

    def test_risk_bands_cover_probability_boundaries(self):
        self.assertEqual(classify_risk(0.0), "LOW")
        self.assertEqual(classify_risk(0.25), "MODERATE")
        self.assertEqual(classify_risk(0.5), "HIGH")
        self.assertEqual(classify_risk(0.75), "EXTREME")
        with self.assertRaises(ValueError):
            classify_risk(1.1)

    def test_api_returns_prediction_and_rejects_invalid_input(self):
        client = TestClient(app)
        artifact = build_model_artifact(self.model)
        request = {
            "temperature": 34,
            "humidity": 31,
            "wind_speed": 18,
            "rainfall": 0,
            "ffmc": 80,
            "dmc": 20,
            "dc": 100,
            "isi": 5,
            "bui": 25,
            "fwi": 6,
        }
        with patch("src.api.get_model", return_value=artifact):
            valid_response = client.post(
                "/predict",
                json=request,
            )
            invalid_request = {**request, "humidity": 101}
            invalid_response = client.post(
                "/predict",
                json=invalid_request,
            )

        self.assertEqual(valid_response.status_code, 200)
        self.assertIn("fire_probability", valid_response.json())
        self.assertEqual(valid_response.json()["prediction"], "FIRE")
        self.assertEqual(invalid_response.status_code, 422)

    def test_saved_artifact_contains_feature_and_target_metadata(self):
        artifact = build_model_artifact(self.model)
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "model.joblib"
            joblib.dump(artifact, artifact_path)
            loaded = load_model(artifact_path)

        self.assertEqual(loaded["feature_columns"], list(BASELINE_FEATURES))
        self.assertEqual(loaded["target_mapping"], TARGET_MAPPING)
        self.assertEqual(len(loaded["feature_columns"]), 10)


class TrainingTests(unittest.TestCase):
    def test_training_uses_all_features_and_reports_probability_metrics(self):
        model, (X_test, y_test, predictions), metrics = train_baseline()

        self.assertEqual(X_test.shape[1], 10)
        self.assertEqual(len(predictions), len(y_test))
        self.assertEqual(model.n_estimators, 300)
        self.assertEqual(model.class_weight, "balanced")
        self.assertIn("roc_auc", metrics)
        self.assertEqual(len(metrics["fire_probabilities"]), len(y_test))


if __name__ == "__main__":
    unittest.main()