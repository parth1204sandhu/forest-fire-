Skill Guide — Forest Fire Risk & Spread Simulation
This document defines the technical skills and implementation rules for building the project.
1. Core engineering skill tree
The project should develop skills in this order:
Python
 ↓
Pandas / NumPy
 ↓
Machine Learning
 ↓
Geospatial Data
 ↓
Model Validation
 ↓
FastAPI
 ↓
PostgreSQL/PostGIS
 ↓
React + GIS
 ↓
Fire Simulation
 ↓
Deep Learning / Physics-Informed ML
 ↓
Deployment
Do not jump to the final technology stack before understanding the earlier layers.
2. Data Engineering
Skills
Learn to:
- load CSV/JSON/raster/vector data
- clean missing values
- normalize column names
- validate data types
- detect corrupt rows
- align timestamps
- align spatial datasets
- create reproducible preprocessing pipelines
Required principle
Never overwrite the raw dataset.
Use:
raw data
   ↓
processing
   ↓
clean/intermediate data
   ↓
model-ready data
3. Machine Learning
First baseline
Use:
Random Forest
Why:
- strong tabular baseline
- handles nonlinear relationships
- little preprocessing required
- provides feature importance
- easy to explain
Second baseline
Use:
Logistic Regression
Purpose:
Understand the difference between a simple linear model and a nonlinear ensemble.
Later
Evaluate:
XGBoost
Gradient Boosting
Then consider spatial/deep-learning approaches.
4. Feature Engineering
Group features by physical meaning.
Weather
Temperature
RH
Wind Speed
Wind Direction
Rainfall
Vegetation
NDVI
NDWI
Land Cover
Fuel Type
Terrain
Elevation
Slope
Aspect
Fire Weather
FFMC
DMC
DC
ISI
BUI
FWI
History
Previous fires
Burned area
Fire density
Distance to historical fire
Seasonality
5. Spatial ML
The final model should operate on geographical cells rather than a single generic row.
Concept:
Location
   ↓
500m × 500m grid cell
   ↓
Environmental feature vector
   ↓
Model
   ↓
Fire probability
This creates a spatial risk surface.
6. Temporal ML
The final target is not simply:
"Was there a fire?"
It should become something closer to:
"Will this cell experience a fire within the next 24 hours?"
Therefore features must represent information available before the prediction window.
Avoid:
future weather
future fire state
post-fire satellite measurements
future-derived features
7. Model Validation
Early stage
Use:
train_test_split(..., stratify=y)
Mature stage
Prefer:
historical period → training
later period       → validation
latest period      → test
Also test geographically separated regions where possible.
8. Risk metrics
A fire-risk model must not be evaluated only by accuracy.
Use:
Precision
Of predicted fires, how many were actual fires?
Recall
Of actual fires, how many did we detect?
F1
Balance between precision and recall.
ROC-AUC
Measures ranking ability across thresholds.
PR-AUC
Especially useful when fire events become rare.
Calibration
Checks whether:
80% probability
actually corresponds roughly to an 80% event frequency under comparable conditions.
9. Risk classification
The model should internally produce:
0.00 → 1.00
Then the application can display categories.
Example prototype:
0.00–0.24 LOW
0.25–0.49 MODERATE
0.50–0.74 HIGH
0.75–1.00 EXTREME
These thresholds must eventually be validated rather than treated as universal scientific standards.
10. GIS Skills
Learn:
- coordinate reference systems
- raster vs vector data
- GeoJSON
- shapefiles
- GeoTIFF
- spatial joins
- raster resampling
- reprojection
- spatial indexing
- geographic grids
Python tools:
GeoPandas
Rasterio
GDAL
Shapely
11. Satellite Data Skills
Eventually learn to work with:
MODIS / VIIRS
Useful for:
- active fire
- thermal anomalies
- fire radiative power
Sentinel-2
Useful for:
- vegetation
- NDVI
- NDWI
- land cover
Sentinel-1
Useful because SAR can provide information even when optical imagery is affected by clouds.
INSAT
Potentially useful for regional meteorological/thermal monitoring depending on available products.
12. Fire-Spread Simulation
Start with Cellular Automata.
Grid:
┌───┬───┬───┐
│ U │ U │ U │
├───┼───┼───┤
│ U │ B │ U │
├───┼───┼───┤
│ U │ U │ U │
└───┴───┴───┘
Where:
U = Unburned
B = Burning
At each timestep:
new_state =
    current state
    + neighborhood
    + wind
    + slope
    + fuel
Eventually:
UNBURNED → BURNING → BURNED
13. Fire Physics
Later learn:
- Rothermel fire spread
- fuel moisture
- rate of spread
- wind influence
- slope influence
- fireline intensity
Physics should be used to constrain or inform advanced models rather than adding complicated equations without validation.
14. Backend Skills
Learn FastAPI concepts:
HTTP
REST
JSON
request validation
response models
routing
dependency injection
authentication
background tasks
Initial endpoint:
POST /predict
Input:
{
  "temperature": 34,
  "humidity": 31,
  "wind_speed": 18,
  "rainfall": 0
}
Output:
{
  "fire_probability": 0.78,
  "risk_level": "HIGH"
}
15. Async Simulation
A 12-hour simulation may take longer than a normal HTTP request.
Eventually:
POST /simulate
       ↓
create job
       ↓
Celery worker
       ↓
simulation
       ↓
store results
       ↓
frontend retrieves results
Do not add Celery until the synchronous simulation works.
16. Database Skills
Learn PostgreSQL first.
Then:
PostGIS
for:
- points
- polygons
- grid cells
- spatial queries
- geometry storage
Possible entities:
regions
grid_cells
fire_events
environmental_observations
predictions
simulations
simulation_states
17. Frontend/GIS Skills
React should consume the backend rather than contain ML logic.
Architecture:
React
  ↓ HTTP
FastAPI
  ↓
ML/Simulation
Mapping:
Leaflet
or:
Mapbox GL JS
The frontend should visualize model outputs, not retrain models.
18. Advanced ML
Only after the tabular/spatial baseline works:
U-Net
Useful for spatial raster prediction/segmentation problems.
ST-GCN
Useful for spatial-temporal relationships between neighboring grid cells over time.
Physics-informed neural networks
Useful for incorporating physical constraints into learned models.
These are advanced components, not MVP requirements.
19. Testing Skills
Test:
Data
- expected columns
- data types
- missing values
- valid target classes
ML
- probability is between 0 and 1
- model loads correctly
- same model/input produces reproducible output where expected
API
- valid request
- missing field
- invalid number
- out-of-range value
- server error handling
Simulation
- ignition point accepted
- grid updates
- simulation terminates
- boundaries remain valid
20. Performance Skills
Early:
NumPy
vectorization
efficient pandas operations
Later:
parallel processing
GPU
PyTorch
CUDA
Avoid Python loops over millions of spatial cells when vectorized operations are possible.
21. Deployment Skills
Final deployment should eventually include:
Frontend
Backend
Database
ML model
Simulation worker
Potentially containerized with:
Docker
Docker Compose
Use environment variables for:
- API keys
- database passwords
- tokens
- service credentials
Never commit secrets.
22. Development order
The recommended skill/project sequence is:
PHASE 1
Python + pandas + dataset
        ↓
PHASE 2
EDA + Random Forest
        ↓
PHASE 3
Model evaluation + probability
        ↓
PHASE 4
Feature engineering
        ↓
PHASE 5
Indian historical/geospatial data
        ↓
PHASE 6
500m spatial grid
        ↓
PHASE 7
GIS risk map
        ↓
PHASE 8
FastAPI
        ↓
PHASE 9
PostgreSQL/PostGIS
        ↓
PHASE 10
Cellular Automata
        ↓
PHASE 11
12-hour spread visualization
        ↓
PHASE 12
Satellite integration
        ↓
PHASE 13
Advanced spatial/deep models
        ↓
PHASE 14
Physics-informed simulation
        ↓
PHASE 15
Deployment
23. Definition of "done"
A component is not complete merely because code exists.
It is complete when:
Code
 ↓
Runs
 ↓
Tested
 ↓
Output validated
 ↓
Documented
 ↓
Connected to next layer
24. Core principle
The project should evolve from:
"Can I predict fire from weather?"
to:
"Can I estimate fire risk for every location?"
and finally:
"Can I estimate where a fire will start,
how likely it is,
and how it will spread over the next 12 hours?"
Build toward that goal without pretending that a small prototype dataset is already an operational wildfire-warning system.