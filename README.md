🔥 Predictive Forest Fire Risk & Spread Simulation System
An AI-powered prototype for 24-hour forest fire risk prediction, 12-hour fire-spread simulation, and an interactive GIS disaster-management dashboard.
This project follows the provided ISRO-inspired technical blueprint and is being developed incrementally: first a working ML risk model, then spatial prediction, fire-spread simulation, GIS visualization, backend APIs, and finally advanced satellite/physics-informed components.
Current status: Stage 1 — historical environmental dataset + baseline ML model
Long-term target: 24-hour spatial fire-risk prediction + 12-hour fire-spread simulation.

1. Project Vision
Forest fires are difficult to manage because responders need to know both:
1. Where a fire is likely to occur, and
2. How an active fire may spread.
The proposed system therefore has two main intelligence modules:
                    FOREST FIRE SYSTEM
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
       24-HOUR RISK MODEL         12-HOUR SPREAD MODEL
              │                         │
              ↓                         ↓
   Where is fire likely?       How will an active fire spread?
              │                         │
              └────────────┬────────────┘
                           ↓
                    GIS DASHBOARD
2. Main Objectives
Module 1 — 24-Hour Fire Risk Prediction
Predict the probability that a geographical grid cell will experience a fire within the next 24 hours.
Output:
- fire probability
- fire/no-fire classification
- risk level
- spatial fire-risk map
Module 2 — 12-Hour Fire Spread Simulation
Given an active ignition point or current fire boundary, estimate:
- direction of spread
- spread velocity
- fire intensity
- burned area
- predicted boundary
- progression over 12 hours
Module 3 — GIS Dashboard
Provide an actionable interface showing:
- risk layers
- active fires
- predicted fire boundaries
- terrain
- environmental layers
- time-based fire progression
- location-specific predictions
3. Current MVP
The first working prototype uses the Algerian Forest Fires Dataset to establish the ML pipeline.
Current cleaned dataset:
- 243 usable observations
- 137 fire observations
- 106 non-fire observations
- 14 columns
Stage-1 model features:
- Temperature
- RH
- Ws
- Rain
- FFMC
- DMC
- DC
- ISI
- BUI
- FWI
The model deliberately excludes `day`, `month`, and `year` as predictors. `Classes` is encoded as `fire = 1` and `not fire = 0`.
Target:
fire
not fire
Primary baseline: a 300-tree Random Forest with balanced class weights. A scaled Logistic Regression baseline is evaluated on the same fixed, stratified 80/20 split.

The six fire-weather indices are included in this requested Stage-1 feature set, but they are not independent measurements: ISI is derived using FFMC and wind, BUI is derived from DMC/DC, and FWI combines ISI and BUI. Their related information can inflate apparent performance and make individual feature-importance values unstable. The 24-hour forecast in the long-term vision is not the current target: this dataset labels observed `Classes`, and the random holdout does not establish future or geographic forecasting skill.

Stage-1 commands (run from the project root, after installing `requirements.txt`):

```bash
python -m src.explore
python -m src.train
python -m src.compare_models
python -m src.predict --temperature 34 --humidity 31 --wind-speed 18 --rainfall 0 --ffmc 80 --dmc 20 --dc 100 --isi 5 --bui 25 --fwi 6
python -m unittest discover -s tests -v
```

**Local GIS demo:** train the model, then start the existing FastAPI app and open the dashboard:

```bash
python -m src.train
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/`. The dashboard calls the saved model for one environmental risk estimate and animates a separate 12 × 12 grid of 500 m cells. The grid uses an illustrative wind-biased cellular automaton, not a calibrated spread model. Its Shimla coordinates are synthetic and are not linked to the Algerian training observations. Map tiles require an internet connection.

Training saves `models/random_forest_baseline.joblib`, including feature order, target mapping, and missing-value policy. The feature-importance chart is saved at `reports/baseline_feature_importance.png`.

On the current 49-row holdout, the ten-feature Random Forest scored 1.000 for accuracy, precision, recall, F1, and ROC-AUC; its confusion matrix had no errors. Logistic Regression scored 0.939 accuracy, 0.931 precision, 0.964 recall, 0.947 F1, and 0.993 ROC-AUC. The four raw-weather-only experiment scored 0.816 accuracy. These unusually strong index-feature results are a reason for caution, not evidence of reliable operational prediction. False positives can trigger unnecessary response; false negatives can miss a fire event and may be more consequential. Threshold choice trades one kind of error against the other.

Prototype risk display thresholds are: probability `< 0.25` LOW, `< 0.50` MODERATE, `< 0.75` HIGH, and otherwise EXTREME. They are visualization bands only and are not calibrated or scientifically validated wildfire warning thresholds.
This is only the starting point. The final system is intended to use Indian geographic, satellite, meteorological, vegetation, terrain, and historical-fire data.
4. High-Level Architecture
Satellite / Fire Data ──────┐
                            │
Vegetation / Fuel ──────────┤
                            │
Weather ────────────────────┤
                            ├──→ Data Processing
Topography / DEM ───────────┤          │
                            │          ↓
Historical Fires ───────────┘   Feature Engineering
                                       │
                          ┌────────────┴────────────┐
                          ↓                         ↓
                  24h Risk Model             Spread Engine
                          ↓                         ↓
                  Risk Probability          Fire Progression
                          │                         │
                          └────────────┬────────────┘
                                       ↓
                              FastAPI Backend
                                       ↓
                              PostgreSQL/PostGIS
                                       ↓
                              React GIS Dashboard
5. Data Requirements
The final system should combine four major data categories.
5.1 Satellite / Active Fire Data
Potential sources from the blueprint:
- ISRO INSAT-3D / INSAT-3DR
- NASA MODIS
- VIIRS
- Sentinel-2
- Sentinel-1
- Google Earth Engine
- other suitable satellite products
Potential variables:
- thermal anomalies
- active fire detections
- fire radiative power (FRP)
- burn information
5.2 Vegetation and Fuel
Potential variables:
- NDVI
- NDWI
- land-cover type
- vegetation/fuel type
Potential sources:
- ISRO Bhuvan
- Sentinel-2
- Landsat
- Google Earth Engine
5.3 Meteorological Data
Potential variables:
- surface temperature
- relative humidity
- wind speed
- wind direction
- precipitation
5.4 Topographical Data
Potential variables:
- elevation
- slope
- aspect
- terrain characteristics
Potential sources:
- CartoDEM
- SRTM DEM
- other appropriate DEM datasets
6. Spatial and Temporal Processing
The final system should align datasets onto a common spatial grid.
Blueprint target:
500 m × 500 m grid cells
Each cell becomes an ML prediction unit.
Conceptually:
             REGION
┌─────┬─────┬─────┬─────┐
│ C01 │ C02 │ C03 │ C04 │
├─────┼─────┼─────┼─────┤
│ C05 │ C06 │ C07 │ C08 │
├─────┼─────┼─────┼─────┤
│ C09 │ C10 │ C11 │ C12 │
└─────┴─────┴─────┴─────┘

Each cell:
weather + vegetation + terrain + history
                    ↓
              fire probability
Datasets must also be temporally aligned so that training features represent information available at prediction time.
7. Feature Engineering
Planned feature groups:
Weather
- temperature
- relative humidity
- wind speed
- wind direction
- precipitation
Vegetation
- NDVI
- NDWI
- land-cover/fuel type
Terrain
- elevation
- slope
- aspect
Fire-weather
- FFMC
- DMC
- DC
- ISI
- BUI
- FWI
Historical fire information
- previous fire occurrence
- burned area
- fire density
- distance to previous fires
- temporal fire patterns
8. Module 1 — 24-Hour Fire Risk Prediction
Goal
For every geographical grid cell:
P(fire within next 24 hours)
Example:
Cell A → 0.08
Cell B → 0.31
Cell C → 0.76
Cell D → 0.91
These probabilities become a GIS risk layer.
Baseline models
The blueprint recommends:
- Random Forest
- XGBoost
The current prototype starts with Random Forest.
Advanced models
Later possibilities:
- U-Net
- Spatial-Temporal Graph Convolutional Network (ST-GCN)
- other spatial-temporal models
Advanced models should only be introduced after the baseline is working and properly validated.
9. Module 2 — 12-Hour Fire Spread Simulation
Once an ignition point is supplied:
Ignition
   ↓
Current fire state
   ↓
Wind + terrain + fuel
   ↓
Spread simulation
   ↓
Future fire boundary
Inputs:
- initial ignition/fire location
- current fire boundary
- wind speed
- wind direction
- slope
- aspect
- vegetation/fuel type
- terrain
Cellular Automata
The first simulation approach can represent the region as a grid.
Each cell may be:
UNBURNED
BURNING
BURNED
The next state depends on:
- neighboring cells
- wind
- slope
- fuel
- current cell state
Example:
        wind →
┌─────┬─────┬─────┐
│     │ 🔥  │ 🔥  │
├─────┼─────┼─────┤
│     │ 🔥  │     │
├─────┼─────┼─────┤
│     │     │     │
└─────┴─────┴─────┘
Advanced physics-informed model
The blueprint proposes physics-informed neural networks incorporating fire-spread physics, including Rothermel-style fire-spread relationships.
This is a later-stage feature, not part of the first MVP.
10. Expected Outputs
The final platform should produce:
- fire-risk probability for geographic cells
- risk classification
- interactive risk map
- predicted fire boundary
- spread direction
- spread velocity
- relative fire intensity
- burned-area estimate
- up to 12 hours of fire progression
- interactive GIS visualization
11. GIS Dashboard
The dashboard should eventually contain:
┌───────────────────────────────────────┐
│ FOREST FIRE RISK DASHBOARD            │
├───────────────────────────────────────┤
│                                       │
│       INTERACTIVE MAP                 │
│                                       │
│    🟢 🟢 🟡 🟠 🔴                     │
│    🟢 🟡 🟠 🔴 🔴                     │
│    🟡 🟠 🔴 🔴 🔴                     │
│                                       │
├───────────────────────────────────────┤
│ Risk │ Active Fire │ Spread │ Terrain │
├───────────────────────────────────────┤
│ Time: 06:00 → 18:00                   │
│                                       │
│ Predicted boundary: ...               │
│ Probability: 82%                      │
└───────────────────────────────────────┘
Required capabilities:
- interactive geographical map
- fire-risk overlay
- active-fire/ignition visualization
- predicted fire boundaries
- time-based spread animation
- terrain/environmental layers
- location-specific information
- optional 3D terrain visualization
12. Backend
Proposed backend:
FastAPI
Responsibilities:
- expose prediction APIs
- execute ML inference
- receive geographic queries
- start simulations
- handle long-running simulation jobs
- return prediction/simulation results
- communicate with the database
Asynchronous tasks may use:
Celery
when simulations become computationally expensive.
13. Database
Proposed:
PostgreSQL + PostGIS
Potential data:
- geographic grid information
- spatial geometries
- historical fire events
- active fires
- environmental features
- model predictions
- simulation states
- simulation results
14. Technology Stack
Component	Technology
Data Processing	Python, GDAL, Rasterio, GeoPandas
ML	Scikit-learn, XGBoost, PyTorch/TensorFlow
Backend	FastAPI
Async Jobs	Celery
Database	PostgreSQL + PostGIS
Frontend	React
Mapping	Mapbox GL JS or Leaflet
3D	Deck.gl / suitable 3D GIS library
Satellite/GIS	Google Earth Engine, ISRO/NASA/ESA data where appropriate


The exact technology should be chosen based on the requirements of each stage rather than adding every listed technology immediately.
15. Functional Requirements
ID	Requirement
FR-01	Select a geographical region
FR-02	Acquire required environmental datasets
FR-03	Preprocess and align datasets to a common grid
FR-04	Generate fire-risk features
FR-05	Predict 24-hour fire susceptibility
FR-06	Display fire susceptibility on an interactive map
FR-07	Accept an active/simulated ignition point
FR-08	Initialize fire-spread simulation
FR-09	Simulate fire propagation for up to 12 hours
FR-10	Use wind, terrain and fuel-related factors
FR-11	Display predicted fire boundaries
FR-12	Inspect predicted fire progression over time
FR-13	Provide prediction/simulation results through APIs
FR-14	Store relevant spatial and prediction information


16. Non-Functional Requirements
Performance
The system should eventually execute large simulations efficiently.
Scalability
The architecture should support:
- larger geographical regions
- more grid cells
- higher-resolution datasets
Reliability
Identical inputs should produce reproducible results where deterministic models are used.
Visualization
The GIS interface should remain responsive enough for practical inspection.
Extensibility
New:
- satellite sources
- weather sources
- terrain datasets
- ML models
should be integrable without rewriting the entire system.
Computational Efficiency
GPU acceleration may be used later for computationally intensive spread simulations and neural models.
17. Training and Validation
The blueprint proposes historical Indian fire data, particularly covering diverse terrains such as:
- Himalayas
- Western Ghats
and historical burn/fire information.
For the final model, validation should include recent held-out fire events.
Risk model metrics
- accuracy
- precision
- recall
- F1-score
- ROC-AUC
Spread model metrics
- Intersection over Union (IoU)
- predicted vs observed boundary overlap
- area difference
- spatial error
Time-aware and geographically independent validation should be preferred over a simple random split when sufficient data becomes available.
18. Key Challenges
Cloud obstruction
Satellite observations may be unavailable because of clouds.
Possible mitigation:
- Sentinel-1 SAR
- complementary satellite sources
- missing-data strategies
Data imbalance
Fire events may be much rarer than non-fire events.
Possible mitigation:
- class weighting
- SMOTE where appropriate
- careful sampling
- appropriate evaluation metrics
Computational latency
Cellular-automata and spatial simulations may become expensive.
Possible mitigation:
- vectorization
- optimized numerical operations
- GPU acceleration
- asynchronous processing
19. Implementation Roadmap
Stage 1 — ML Risk Prototype
- historical fire + environmental dataset
- preprocessing
- Random Forest/XGBoost baseline
- model evaluation
- initial GIS risk map
Stage 2 — Spread Simulation
- cellular automata
- 12-hour simulation
- backend integration
Stage 3 — Advanced Intelligence
- satellite data
- spatial-temporal models
- physics-informed fire spread
- GPU optimization
20. Six-Month Blueprint
Period	Deliverable
Month 1–2	Automated satellite/weather/DEM data aggregation and preprocessing
Month 3	Historical fire-risk model training
Month 4	Cellular Automata spread simulation + physics calibration
Month 5	FastAPI + GIS frontend
Month 6	Testing, validation and blind back-testing


21. MVP Scope
For a realistic working prototype, implement incrementally:
Stage 1
Historical fire + environmental data → preprocessing → Random Forest/XGBoost → GIS risk map.
Stage 2
Add cellular-automata-based 12-hour spread simulation and backend integration.
Stage 3
Add advanced spatial models, physics-informed learning, additional satellite sources and GPU optimization.
22. Final Deliverables
The complete project aims to contain:
- data ingestion pipeline
- preprocessing pipeline
- 24-hour fire-risk model
- 12-hour fire-spread engine
- physics-informed advanced component
- FastAPI backend
- PostgreSQL/PostGIS database
- React GIS dashboard
- model evaluation/validation pipeline
- technical documentation
- deployment documentation
23. Important Scientific Limitation
The initial Algerian dataset is a development dataset, not the final operational dataset.
The final Indian fire-risk system must use geographically and temporally appropriate data and must be independently validated.
The project should therefore be described as a:
Predictive Forest Fire Risk & Spread Simulation Prototype

until sufficient real-world validation has been completed.
24. Development Philosophy
Build the system one layer at a time:
Dataset
   ↓
Cleaning
   ↓
EDA
   ↓
Baseline ML
   ↓
Validation
   ↓
Spatial features
   ↓
GIS risk map
   ↓
Spread simulation
   ↓
API
   ↓
Database
   ↓
Dashboard
   ↓
Advanced models
   ↓
Deployment
Do not build the entire system in one step.