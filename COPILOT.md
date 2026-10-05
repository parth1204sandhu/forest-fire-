# Copilot Instructions — Forest Fire Predictor

## Project goal

This project is an AI/ML-based forest fire **risk predictor**.

The current goal is to learn the complete pipeline:

`historical environmental data → data cleaning → analysis → ML model → probability → risk level → API/dashboard → deployment`

This is a learning project first and a polished application second.

## Current status

- Development environment: Ubuntu/Linux
- Language: Python 3
- Virtual environment: `venv`
- Dataset: Algerian Forest Fires Dataset
- Dataset location: `data/Algerian_forest_fires_dataset_UPDATE.csv`
- Current cleaned dataset: approximately 243 usable observations
- Current baseline model: Random Forest Classifier
- Current baseline features:
  - Temperature
  - RH (relative humidity)
  - Ws (wind speed)
  - Rain
- Target:
  - `fire`
  - `not fire`

## Important project principles

1. **Do not generate the entire application at once.**
   Build it incrementally and keep the architecture understandable.

2. **Explain before adding complexity.**
   When introducing a library, model, API, database, or framework, explain:
   - what it does
   - why we need it
   - where it fits in the architecture

3. **Prefer simple, readable code.**
   This project is being built by a beginner who is learning Python, ML, Linux, networking, and backend development.

4. **Do not hide important logic behind unnecessary abstractions.**
   The user should be able to understand the data flow.

5. **Never silently fabricate data.**
   If a dataset, API, weather source, satellite source, or geographic source is required, say so.

6. **Keep the original dataset untouched.**
   Perform cleaning through Python/data-processing code.

7. **Avoid data leakage.**
   Do not use information that would only be known after or during a fire when claiming to predict future fire risk.

8. **Treat FWI-related variables carefully.**
   `FFMC`, `DMC`, `DC`, `ISI`, `BUI`, and `FWI` are fire-weather indices and may contain information derived from environmental conditions. They must not be casually added to a model and presented as proof of strong forecasting ability.

9. **Use proper train/test separation.**
   Never evaluate on the same observations used for training.

10. **For future forecasting, prefer time-aware validation.**
    Random train/test splitting is acceptable for the first learning baseline, but future versions should consider temporal validation.

## Coding conventions

- Python 3
- Use clear variable names.
- Use functions when logic becomes reusable.
- Keep scripts small and focused.
- Add comments for important ML/data-processing decisions, not every obvious line.
- Prefer `pathlib` for new filesystem code.
- Use `pandas` for tabular data.
- Use `scikit-learn` for baseline ML.
- Save trained models explicitly rather than retraining on every prediction.

## Expected project structure

```text
forest-fire-predictor/
├── data/
│   └── Algerian_forest_fires_dataset_UPDATE.csv
├── models/
├── src/
│   ├── explore.py
│   ├── model.py
│   └── ...
├── venv/
├── README.md
├── copilot.md
└── skill.md
```

## Planned architecture

### Phase 1 — ML foundation
- Load and clean historical data
- Explore distributions and relationships
- Establish a baseline model
- Evaluate accuracy, precision, recall, F1, and confusion matrix
- Generate calibrated/meaningful fire probabilities

### Phase 2 — Better prediction
- Feature engineering
- Compare Random Forest, Logistic Regression, and gradient boosting
- Handle class imbalance if needed
- Use cross-validation
- Investigate temporal validation
- Evaluate probability calibration

### Phase 3 — Geographic risk
- Latitude/longitude
- Terrain/elevation
- Vegetation/NDVI
- Historical fire locations
- Grid-based risk prediction
- Interactive map

### Phase 4 — Forecasting
- Weather forecast data
- 24-hour / 48-hour risk prediction
- Time-based features
- Historical fire frequency
- Separate training/validation periods

### Phase 5 — Application
```text
Frontend
   ↓
FastAPI backend
   ↓
Prediction service
   ↓
Trained ML model
   ↓
Risk probability
```

Possible additions:
- PostgreSQL/Supabase
- authentication
- map visualization
- alerts
- scheduled predictions

### Phase 6 — Deployment
- Docker
- backend deployment
- frontend deployment
- environment variables/secrets
- logging
- monitoring

## ML safety and scientific honesty

This project should be described as a **prototype/research/educational risk prediction system**, not as a certified wildfire warning system.

Do not claim that the model can reliably predict real-world fires unless it has been validated on appropriate, independent, time-separated, geographically relevant data.

Accuracy alone is not enough. For a fire-risk system, pay attention to:
- recall for fire events
- precision
- F1 score
- confusion matrix
- ROC-AUC / PR-AUC where appropriate
- calibration of predicted probabilities
- false negatives

A false negative can be much more consequential than a false positive.

## How Copilot should behave

When asked to implement something:

1. Explain the purpose in 1–3 sentences.
2. Identify which file should change.
3. Make the smallest useful change.
4. Explain how the change connects to the pipeline.
5. Give the command to test it.
6. Do not rewrite unrelated files.
7. Do not introduce a new framework without a reason.
