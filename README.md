🔥 Forest Fire Predictor
An AI/ML project that predicts forest fire risk from environmental conditions and eventually produces geographic and time-based risk forecasts.
Project status: Early ML prototype

🎯 Goal
The long-term goal is:
Weather + Vegetation + Terrain + Historical Fire Data
                         ↓
                    ML Model
                         ↓
                 Fire Probability
                         ↓
               Risk Classification
                         ↓
             Map / Dashboard / Alert
The project is being built incrementally so that the architecture and ML workflow are understood rather than hidden behind generated code.
Current version
The first baseline uses the Algerian Forest Fires Dataset.
Current input features:
- Temperature
- Relative Humidity (RH)
- Wind Speed (Ws)
- Rainfall (Rain)
Target:
fire
not fire
Current cleaned dataset:
- 243 usable observations
- 137 fire observations
- 106 non-fire observations
The initial model is a Random Forest Classifier.
Why only four features initially?
The dataset also contains:
- FFMC
- DMC
- DC
- ISI
- BUI
- FWI
These are fire-weather indices. They are useful, but they are intentionally excluded from the first baseline so we can measure how much predictive information comes from basic environmental conditions alone.
🧠 What we are learning
This project is also a practical way to learn:
- Python
- pandas
- NumPy
- data cleaning
- exploratory data analysis
- machine learning
- model evaluation
- probability prediction
- feature engineering
- APIs
- backend development
- GIS/geospatial data
- deployment
- Linux
- Git/GitHub
📁 Project structure
forest-fire-predictor/
│
├── data/
│   ├── Algerian_forest_fires_dataset_UPDATE.csv
│   └── algerian+forest+fires+dataset.zip
│
├── models/
│   └── random_forest_baseline.joblib  # created by training
│
├── src/
│   ├── api.py
│   ├── compare_feature_sets.py
│   ├── compare_models.py
│   ├── data_loader.py
│   ├── explore.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
│
├── reports/                            # created by training
├── tests/
│   └── test_pipeline.py
│
├── README.md
├── COPILOT.md
├── SKILL.md
└── requirements.txt
🛠️ Setup
1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
2. Install the project dependencies
pip install -r requirements.txt
📊 Dataset
The first dataset is the Algerian Forest Fires Dataset from the UCI Machine Learning Repository.
It contains observations from two Algerian regions during 2012.
The raw dataset contains formatting artifacts such as:
- region title rows
- repeated headers
- whitespace in column names
- whitespace in target labels
- missing values
These are cleaned programmatically without modifying the original CSV.
🧹 Current data-cleaning pipeline
The current preprocessing performs:
1. Skip the first region title and strip column-name whitespace.
2. Convert available numeric columns, coercing invalid values to missing.
3. Trim and normalize target labels to lowercase.
4. Keep only `fire` and `not fire` rows.
5. Drop rows missing a requested feature or target.
Conceptually:
Raw CSV
   ↓
Remove formatting rows
   ↓
Clean column names
   ↓
Clean target labels
   ↓
Convert features to numbers
   ↓
Remove incomplete observations
   ↓
Clean DataFrame
🤖 Baseline ML pipeline
The first model follows:
Clean dataset
      ↓
Select features
      ↓
X = environmental conditions
y = fire / no fire
      ↓
Train/test split
      ↓
Random Forest
      ↓
Predictions
      ↓
Evaluation
The current feature set:
features = [
    "Temperature",
    "RH",
    "Ws",
    "Rain"
]
Target encoding:
fire     → 1
not fire → 0
The shared loader is `src/data_loader.py`. Run commands from the project root:

```bash
python -m src.explore
python -m src.train
python -m src.compare_models
python -m src.compare_feature_sets
python -m src.predict --temperature 34 --humidity 31 --wind-speed 18 --rainfall 0
```

Training saves `models/random_forest_baseline.joblib` and writes the feature-importance plot to `reports/baseline_feature_importance.png`. Predictions load that saved model; they do not retrain it.
📈 Model evaluation
Do not judge the system using accuracy alone.
The training and comparison commands report:
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
For wildfire risk, recall for fire events is particularly important, because missing a genuine fire-risk situation can be more serious than producing a false alarm.
The baseline uses one seeded, stratified 80/20 random split. Its results are educational holdout metrics, not a reliable estimate of future or geographic performance. ROC-AUC, PR-AUC, calibration, and stronger validation remain future work.
🔌 API
Train the model first, then start the local API:

```bash
uvicorn src.api:app --reload
```

Send a `POST` request to `http://127.0.0.1:8000/predict`:

```json
{
      "temperature": 34,
      "humidity": 31,
      "wind_speed": 18,
      "rainfall": 0
}
```

The response includes `fire_probability`, `predicted_class`, and `risk_level`. Invalid or out-of-range input receives HTTP `422`; the model is loaded from disk and is not trained by the API.

Risk categories use the educational thresholds in `src/predict.py`: below 0.25 LOW, below 0.50 MODERATE, below 0.75 HIGH, otherwise EXTREME. These thresholds are not calibrated or validated warning levels.

Run tests with:

```bash
python -m unittest discover -s tests -v
```

The feature-importance plot reflects how this fitted Random Forest used the features; it does not show causality. The separate index experiment intentionally adds FFMC, DMC, DC, ISI, BUI, and FWI only for comparison. These related fire-weather variables may make results look stronger and are not used by the saved baseline or prediction API.

🔮 Future development
Phase 3 — Geographic prediction
Add:
- latitude
- longitude
- elevation
- slope
- vegetation
- NDVI
- historical fire density
Then produce:
🟢 Low
🟡 Moderate
🟠 High
🔴 Extreme
risk maps.
Phase 4 — Future forecasting
The target becomes:
What is the probability of a fire occurring in this area during the next 24–48 hours?

This requires time-aware data such as:
- historical weather
- weather forecasts
- historical fire locations
- vegetation conditions
- terrain
- seasonality
Random train/test splitting should not be treated as sufficient for this stage.
The ML API prototype is implemented. A future frontend can use this architecture:
             Web / Mobile UI
                    ↓
                FastAPI
                    ↓
             Prediction API
                    ↓
              ML Model
                    ↓
          Fire-risk probability
                    ↓
             Risk classification
⚠️ Scientific limitations
The Algerian dataset is useful for learning but is not enough to claim that this system can reliably predict future forest fires in Himachal Pradesh or anywhere else.
A real operational predictor would need geographically relevant, time-stamped data and independent validation.
The eventual system should therefore be presented as:
An AI-based forest fire risk prediction prototype

until it has undergone proper real-world validation.
🚀 Long-term vision
The final project should answer:
"Where is a forest fire most likely to occur, and how risky will the conditions be over the next 24–48 hours?"

Potential final architecture:
Weather Forecast ────────┐
                         │
Satellite / NDVI ────────┤
                         │
Terrain ─────────────────┤
                         ├──→ ML Model ──→ Fire Probability
Historical Fires ────────┤                         │
                         │                         ↓
Season / Time ───────────┘                    Risk Level
                                                   │
                              ┌────────────────────┼─────────────────┐
                              ↓                    ↓                 ↓
                            Map              Dashboard           Alerts
Development philosophy
Build small. Understand the flow. Test every stage. Then add complexity.

