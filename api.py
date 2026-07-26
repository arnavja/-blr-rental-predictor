"""
api.py
FastAPI prediction server for the Bengaluru Rental Price Predictor.

Run locally:
    uvicorn api:app --reload

Then visit:
    http://127.0.0.1:8000/docs   ← interactive UI to test predictions
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.features import engineer_features, FEATURE_COLS, LOCALITY_TIER

# Load model & scaler once at startup
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "outputs", "xgb_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "outputs", "scaler.pkl")

try:
    model  = joblib.load(MODEL_PATH)
    # scaler is only used by Ridge — XGBoost doesn't need it
    # but we load it in case a future endpoint uses Ridge
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model files not found: {e}\n"
        "Run `python src/train.py` first to generate outputs/xgb_model.pkl"
    )

# App
app = FastAPI(
    title="Bengaluru Rental Price Predictor",
    description=(
        "Predicts monthly rent for Bengaluru flats using XGBoost. "
        "Built on synthetic data reflecting 2023-24 market ranges from "
        "99acres and MagicBricks."
    ),
    version="1.0.0",
)


# Input schema
class RentalInput(BaseModel):
    locality: str = Field(..., description="One of the 20 trained Bengaluru localities")
    bhk: int = Field(..., ge=1, le=5, description="Number of bedrooms")
    area_sqft: float = Field(..., gt=0, description="Carpet area in sq ft")
    furnishing: str = Field(..., description="Unfurnished, Semi-Furnished, or Fully Furnished")
    floor: int = Field(..., ge=0, description="Floor number (0 = ground)")
    total_floors: int = Field(..., ge=1, description="Total floors in building")
    building_age_yrs: float = Field(..., ge=0, description="Age of building in years")
    tenant_preferred: str = Field(..., description="Family, Bachelor, or Any")
    dist_metro_km: float = Field(..., ge=0, description="Distance to nearest metro (km)")
    dist_techpark_km: float = Field(..., ge=0, description="Distance to nearest tech park (km)")
    gym: int = Field(0, ge=0, le=1, description="Gym available (0/1)")
    swimming_pool: int = Field(0, ge=0, le=1)
    power_backup: int = Field(0, ge=0, le=1)
    lift: int = Field(0, ge=0, le=1)
    security: int = Field(0, ge=0, le=1)
    parking: int = Field(0, ge=0, le=1)
    club_house: int = Field(0, ge=0, le=1)

    # Example payload shown in the /docs interactive UI
    model_config = {
        "json_schema_extra": {
            "example": {
                "locality": "Koramangala", "bhk": 2, "area_sqft": 1100,
                "furnishing": "Fully Furnished", "floor": 3, "total_floors": 10,
                "building_age_yrs": 5, "tenant_preferred": "Any",
                "dist_metro_km": 0.8, "dist_techpark_km": 2.5,
                "gym": 1, "swimming_pool": 0, "power_backup": 1,
                "lift": 1, "security": 1, "parking": 1, "club_house": 0,
            }
        }
    }

    @field_validator("locality")
    @classmethod
    def locality_must_be_known(cls, v):
        if v not in LOCALITY_TIER:
            raise ValueError(
                f"Unknown locality '{v}'. "
                f"Valid options: {sorted(LOCALITY_TIER.keys())}"
            )
        return v

    @field_validator("furnishing")
    @classmethod
    def furnishing_must_be_valid(cls, v):
        valid = {"Unfurnished", "Semi-Furnished", "Fully Furnished"}
        if v not in valid:
            raise ValueError(f"furnishing must be one of {valid}, got '{v}'")
        return v

    @field_validator("tenant_preferred")
    @classmethod
    def tenant_must_be_valid(cls, v):
        valid = {"Family", "Bachelor", "Any"}
        if v not in valid:
            raise ValueError(f"tenant_preferred must be one of {valid}, got '{v}'")
        return v


# Output schema
class RentalPrediction(BaseModel):
    predicted_rent: int = Field(..., description="Predicted monthly rent in ₹")
    predicted_rent_formatted: str = Field(..., description="Human-readable rent (e.g. ₹32,500)")
    locality: str
    bhk: int
    area_sqft: float
    furnishing: str
    note: str = "Prediction based on XGBoost model trained on Bengaluru 2023-24 synthetic rental data."


# Routes
@app.get("/", summary="Health check")
def root():
    return {
        "status": "ok",
        "message": "Bengaluru Rental Predictor API is running.",
        "docs": "/docs",
        "predict_endpoint": "/predict"
    }


@app.post("/predict", response_model=RentalPrediction, summary="Predict monthly rent")
def predict(data: RentalInput):
    """
    Takes flat details and returns predicted monthly rent in ₹.

    - All amenity fields (gym, lift, etc.) default to 0 if not provided
    - Locality must be one of the 20 Bengaluru areas the model was trained on
    - Prediction is in rupees (log-space inverse applied automatically)
    """
    try:
        # Convert input to DataFrame (engineer_features expects this)
        df = pd.DataFrame([data.model_dump()])

        # Run feature engineering (same pipeline as training)
        df = engineer_features(df)

        # Slice to exactly the features the model was trained on
        X = df[FEATURE_COLS].values

        # Predict (model outputs log(rent))
        log_pred = model.predict(X)[0]
        rent = int(round(np.exp(log_pred)))

        return RentalPrediction(
            predicted_rent=rent,
            predicted_rent_formatted=f"₹{rent:,}",
            locality=data.locality,
            bhk=data.bhk,
            area_sqft=data.area_sqft,
            furnishing=data.furnishing,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/localities", summary="List all valid localities")
def list_localities():
    """Returns all localities the model knows about, grouped by tier."""
    tiers = {1: "Premium", 2: "Upper-Mid", 3: "Mid", 4: "Budget"}
    grouped = {label: [] for label in tiers.values()}
    for locality, tier in sorted(LOCALITY_TIER.items()):
        grouped[tiers[tier]].append(locality)
    return {"localities_by_tier": grouped}
