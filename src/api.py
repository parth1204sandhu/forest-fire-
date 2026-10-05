"""FastAPI application for saved-model fire-risk predictions."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model, predict_fire_risk


class PredictionRequest(BaseModel):
    temperature: float = Field(ge=-20, le=60, description="Degrees Celsius")
    humidity: float = Field(ge=0, le=100, description="Relative humidity percent")
    wind_speed: float = Field(ge=0, le=200, description="Wind speed in km/h")
    rainfall: float = Field(ge=0, le=500, description="Rainfall in mm")
    ffmc: float = Field(ge=0, le=101)
    dmc: float = Field(ge=0, le=1000)
    dc: float = Field(ge=0, le=2000)
    isi: float = Field(ge=0, le=100)
    bui: float = Field(ge=0, le=1000)
    fwi: float = Field(ge=0, le=100)


class PredictionResponse(BaseModel):
    fire_probability: float = Field(ge=0, le=1)
    predicted_class: str
    prediction: str
    risk_level: str


app = FastAPI(title="Forest Fire Risk Prediction Prototype")


@lru_cache(maxsize=1)
def get_model():
    """Load the saved model once per application process."""
    return load_model()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model = get_model()
        result = predict_fire_risk(
            temperature=request.temperature,
            humidity=request.humidity,
            wind_speed=request.wind_speed,
            rainfall=request.rainfall,
            ffmc=request.ffmc,
            dmc=request.dmc,
            dc=request.dc,
            isi=request.isi,
            bui=request.bui,
            fwi=request.fwi,
            model=model,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return PredictionResponse(**result)