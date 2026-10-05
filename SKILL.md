# Skill Guide — Forest Fire Predictor

This file defines how to work on the Forest Fire Predictor project efficiently and safely.

---

# 1. Project objective

Build an end-to-end forest fire **risk prediction** system.

The intended pipeline is:

```text
Data
 ↓
Cleaning
 ↓
Exploration
 ↓
Feature Engineering
 ↓
ML Training
 ↓
Evaluation
 ↓
Probability Prediction
 ↓
Risk Level
 ↓
API
 ↓
Map / Dashboard
 ↓
Deployment
```

The project should be built one layer at a time.

---

# 2. Current skill level assumptions

The developer is comfortable with:

- basic programming concepts
- some C++
- beginner Python
- Linux basics
- basic Git

The developer is still learning:

- pandas
- machine learning
- APIs
- databases
- authentication
- deployment
- GIS
- model evaluation

Therefore, solutions should be practical but understandable.

---

# 3. How to explain new code

Whenever adding a significant piece of code, explain:

### What?

What does this code do?

### Why?

Why do we need it?

### Where?

Where does it sit in the project architecture?

### How?

What happens when it runs?

Example:

```text
model.fit(X_train, y_train)
```

Explain:

```text
X_train = environmental conditions
y_train = known fire/no-fire outcomes

The model searches for patterns connecting
environmental conditions to the target.
```

Do not just paste code without context.

---

# 4. Data workflow

The dataset should flow through:

```text
Raw dataset
    ↓
Validation
    ↓
Cleaning
    ↓
Feature selection
    ↓
Train/test split
    ↓
Training
```

Never modify the original raw dataset unless there is a deliberate reason.

Prefer:

```python
df = pd.read_csv(...)
clean_df = ...
```

over permanently editing the source CSV.

---

# 5. Data leakage

Data leakage is one of the most important concepts in this project.

Do not allow the model to see information that would not be available at prediction time.

For example, if predicting tomorrow's fire risk, features calculated using information from after the prediction time cannot be used.

Be especially careful with:

- future observations
- fire occurrence itself
- post-fire satellite information
- derived fire indices
- target-derived features

---

# 6. FWI-related features

The dataset contains:

```text
FFMC
DMC
DC
ISI
BUI
FWI
```

These are important fire-weather indices.

They should be treated carefully because:

```text
Weather conditions
      ↓
Fire-weather calculations
      ↓
FWI-related indices
```

If the model uses these, it may achieve very strong performance because the features already encode fire-danger information.

That is not automatically bad, but it must be explained honestly.

---

# 7. Model development strategy

Start with simple baselines.

Recommended order:

### Model 1

Random Forest using:

```text
Temperature
RH
Ws
Rain
```

### Model 2

Logistic Regression.

Purpose:

Understand whether a simple linear model performs similarly.

### Model 3

Random Forest with additional features.

### Model 4

Gradient boosting.

Compare models using appropriate validation.

---

# 8. Evaluation

Never report only:

```text
Accuracy = 95%
```

Instead report:

```text
Accuracy
Precision
Recall
F1
Confusion Matrix
ROC-AUC
PR-AUC
Calibration
```

For fire detection/risk prediction, pay special attention to:

```text
False Negatives
```

because a false negative means:

```text
Actual fire risk
      ↓
Model says low/no risk
```

---

# 9. Probability vs classification

The final system should preferably output:

```text
Probability = 0.78
```

rather than only:

```text
FIRE
```

Then convert probability into a risk category.

Example prototype thresholds:

```text
0.00–0.24 → LOW
0.25–0.49 → MODERATE
0.50–0.74 → HIGH
0.75–1.00 → EXTREME
```

These are **prototype UI thresholds**, not scientifically validated thresholds. Eventually they should be selected using calibration, validation data, and domain requirements.

---

# 10. Geographic extension

The future model should operate on geographic cells.

Example:

```text
Region
 ↓
Grid cells
 ↓
Each cell gets:
    latitude
    longitude
    weather
    vegetation
    terrain
    historical fire features
 ↓
ML model
 ↓
risk probability
```

The output can become a heatmap.

---

# 11. Forecasting extension

A true predictive system should eventually use future weather forecasts.

Example:

```text
Current time: 10:00 AM

Forecast:
12:00 → 33°C, 31% RH
15:00 → 35°C, 27% RH
18:00 → 32°C, 35% RH

Historical + geographic information
                ↓
             ML model
                ↓
       Future fire probability
```

This is different from simply detecting an existing fire.

---

# 12. Backend architecture

When the ML model is stable:

```text
Frontend
   ↓ HTTP
FastAPI
   ↓
Prediction Service
   ↓
Saved ML Model
   ↓
Probability
```

Example:

```http
POST /predict
```

Request:

```json
{
  "temperature": 34,
  "humidity": 31,
  "wind_speed": 18,
  "rainfall": 0
}
```

Response:

```json
{
  "fire_probability": 0.78,
  "risk": "HIGH"
}
```

---

# 13. Frontend architecture

The frontend should not contain the ML model.

Correct:

```text
Frontend
   ↓
API
   ↓
Python model
```

Incorrect:

```text
Frontend
   ↓
ML model directly
```

This separation makes deployment and maintenance easier.

---

# 14. Database

A database becomes useful when the project needs:

- users
- saved predictions
- historical predictions
- locations
- alerts
- dashboard history

Do not add a database merely because modern applications usually have one.

Add it when persistent application data is actually needed.

---

# 15. Deployment

Possible final architecture:

```text
                 Internet
                    │
             ┌──────┴──────┐
             │             │
          Frontend       API
                           │
                      ML Model
                           │
                       Database
```

Use environment variables for secrets.

Never commit:

```text
API keys
passwords
tokens
database credentials
```

to Git.

---

# 16. Recommended development order

Follow this order:

```text
1. Dataset
2. Cleaning
3. EDA
4. Baseline model
5. Evaluation
6. Probability output
7. Model persistence
8. Prediction script
9. Better features
10. Geographic data
11. Weather forecasting
12. FastAPI
13. Frontend
14. Map
15. Database
16. Alerts
17. Docker
18. Deployment
```

Do not jump directly to step 13.

---

# 17. Git workflow

Commit meaningful milestones.

Example:

```bash
git add .
git commit -m "Clean forest fire dataset"
```

Then:

```bash
git commit -m "Add baseline random forest model"
```

Avoid commits like:

```text
stuff
changes
final final
test
```

Good commit history should tell the story of the project.

---

# 18. Quality checklist

Before considering a feature complete:

- Does it work?
- Can we explain it?
- Is the data flow correct?
- Is there data leakage?
- Is there a test?
- Does it handle missing/invalid input?
- Does it introduce unnecessary complexity?
- Is the result scientifically honest?

---

# 19. Core principle

The goal is not:

> "Generate the biggest AI project possible."

The goal is:

> **Build a real forest-fire risk prediction system while understanding how every major component connects.**

Keep the system simple until complexity is justified.