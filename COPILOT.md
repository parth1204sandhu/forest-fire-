Copilot Instructions — Predictive Forest Fire Risk & Spread Simulation
Read this first
Before changing code, read:
- README.md
- copilot.md
- skill.md
These files define the project requirements and development rules.
1. Project objective
Build an AI-powered forest-fire risk and spread simulation system based on the provided technical blueprint.
The system has two primary ML/simulation modules:
1. 24-hour fire-risk prediction
2. 12-hour fire-spread simulation
and one primary application layer:
3. GIS disaster-management dashboard
2. Development strategy
This is an incremental project.
DO NOT:
- generate the entire application at once
- replace working code unnecessarily
- create a frontend before the ML core works
- introduce PostgreSQL/PostGIS before spatial data is actually required
- introduce deep learning before baseline models are evaluated
- claim that prototype accuracy represents real-world wildfire prediction
- fabricate satellite/weather/fire data
- modify the original raw dataset
DO:
- inspect the repository before editing
- reuse existing working code
- make small changes
- test after each change
- explain major architectural decisions
- preserve reproducibility
- keep modules separated
3. Target architecture
Satellite / Fire
Vegetation / Fuel
Weather
Topography
Historical Fires
        ↓
Data ingestion
        ↓
Cleaning + spatial/temporal alignment
        ↓
Feature engineering
        ↓
24h Risk Model ──────────────┐
                             │
Active Fire + Terrain ──────→ Spread Engine
                             │
                             ↓
                         FastAPI
                             ↓
                       PostgreSQL/PostGIS
                             ↓
                        React GIS UI
4. Current project stage
The current prototype already uses the Algerian Forest Fires Dataset.
Current baseline:
Features:
Temperature
RH
Ws
Rain

Target:
fire / not fire

Model:
Random Forest
The current dataset contains approximately 243 usable rows after cleaning.
Do not throw away this working baseline while adding new features.
5. Risk-model requirements
Baseline
Use Random Forest or XGBoost.
Start with:
Temperature
RH
Ws
Rain
Then separately evaluate:
FFMC
DMC
DC
ISI
BUI
FWI
Do not automatically mix all features together.
These fire-weather indices are derived fire-danger variables, so their inclusion must be scientifically explained.
6. Final risk model inputs
The final system should support:
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
Fire history
- previous fire occurrence
- burned area
- historical fire density
- spatial/temporal fire patterns
Fire-weather indices
- FFMC
- DMC
- DC
- ISI
- BUI
- FWI
7. Spatial design
The blueprint proposes approximately:
500 m × 500 m grid cells
The final risk model should eventually produce:
grid cell → fire probability
Example:
{
  "cell_id": "HIM_001",
  "latitude": 31.7,
  "longitude": 77.1,
  "fire_probability": 0.82,
  "risk_level": "HIGH"
}
Do not implement spatial grids until the project reaches the spatial-data stage.
8. 24-hour prediction
The target is:
probability that a geographic grid cell experiences a fire within the next 24 hours.

The model must distinguish:
prediction time
        ↓
available information
        ↓
next 24 hours
Avoid temporal leakage.
A feature that is only known after the fire occurs cannot be used to claim future prediction.
9. Validation
For the early prototype:
- train/test split is acceptable
- use stratification
- fixed random seed
- report multiple metrics
For the final forecasting system:
Prefer:
- time-based splits
- geographically independent validation
- blind back-testing
- recent held-out fire events
Metrics:
Accuracy
Precision
Recall
F1
ROC-AUC
For spatial spread:
IoU
boundary overlap
burned-area error
spatial displacement
10. Probability output
The model should provide probability rather than only a class.
Use:
model.predict_proba(...)
Then map the probability to a prototype risk level.
Do not present arbitrary thresholds as scientifically validated thresholds.
11. Spread simulation
The second module predicts how an active fire progresses over 12 hours.
Initial implementation:
Cellular Automata
States:
UNBURNED
BURNING
BURNED
Transition factors:
- neighboring cells
- wind direction
- wind speed
- slope
- aspect
- vegetation/fuel
Later, an advanced physics-informed model may incorporate fire-spread physics/Rothermel-style relationships.
Do not implement a neural physics model before the cellular-automata baseline works.
12. GIS dashboard
The eventual frontend should visualize:
- risk layer
- active fires
- ignition points
- predicted fire boundaries
- terrain
- environmental layers
- time progression
- location-specific predictions
Potential mapping technologies:
- Mapbox GL JS
- Leaflet
Potential frontend:
- React
Do not create a complex UI until backend prediction endpoints are stable.
13. Backend
Preferred backend:
FastAPI
Potential endpoints:
GET  /health
POST /predict
POST /risk-map
POST /simulate
GET  /simulation/{id}
Long-running simulations can later use Celery.
14. Database
Preferred:
PostgreSQL + PostGIS
Use it for:
- geographic grids
- geometries
- historical fire events
- predictions
- simulation states
- environmental data references
Do not add it prematurely.
15. Data sources
Potential final sources include:
Satellite
- ISRO INSAT-3D/3DR
- NASA MODIS
- VIIRS
- Sentinel-1
- Sentinel-2
- Google Earth Engine
Vegetation
- Bhuvan
- Sentinel
- Landsat
- NDVI/NDWI products
Weather
- appropriate historical weather datasets
- forecast weather APIs
Terrain
- CartoDEM
- SRTM
- other suitable DEMs
Never pretend to have accessed a source unless it is actually available.
16. Code quality
Use:
- Python 3
- type hints where useful
- small modules
- clear names
- reusable functions
- pathlib
- logging for application services
- configuration through environment variables where appropriate
Avoid:
- giant scripts
- duplicated preprocessing
- hard-coded absolute paths
- secrets in source code
- unnecessary abstractions
17. Required development workflow
For every major task:
Step 1
Inspect the current repository.
Step 2
Identify the smallest change required.
Step 3
Explain:
What are we adding?
Why?
Which files change?
Step 4
Implement it.
Step 5
Run the relevant tests/commands.
Step 6
Report:
Completed
Test result
Next logical step
Do not skip directly to later phases.
18. Current priority
The immediate priority is:
Clean dataset
   ↓
Baseline Random Forest
   ↓
Evaluation
   ↓
Probability prediction
   ↓
Save model
   ↓
Prediction script
   ↓
Feature analysis
   ↓
GIS risk layer
Only after that should the spread simulation be built.
19. Scientific honesty
Never say:
"The model predicts forest fires accurately in India"

unless it has been independently validated on appropriate Indian data.
Use language such as:
"prototype"

"experimental model"

"development dataset"

"risk estimate"

The system is decision support, not a certified emergency-warning system.
20. Final principle
Build the system from simple to advanced:
Random Forest
      ↓
Better features
      ↓
Spatial model
      ↓
GIS
      ↓
Cellular Automata
      ↓
Physics-informed model
      ↓
Operational architecture
Every advanced component must have a clear reason to exist.