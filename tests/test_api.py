"""
test_api.py
-----------
Integration tests for the FastAPI prediction server.
Uses FastAPI's TestClient — no running server needed.

Run:  pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


VALID_PAYLOAD = {
    "locality": "Koramangala",
    "bhk": 2,
    "area_sqft": 1100,
    "furnishing": "Fully Furnished",
    "floor": 3,
    "total_floors": 10,
    "building_age_yrs": 5,
    "tenant_preferred": "Any",
    "dist_metro_km": 0.8,
    "dist_techpark_km": 2.5,
    "gym": 1, "swimming_pool": 0, "power_backup": 1,
    "lift": 1, "security": 1, "parking": 1, "club_house": 0,
}


def test_health_check():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_localities_endpoint():
    r = client.get("/localities")
    assert r.status_code == 200
    tiers = r.json()["localities_by_tier"]
    assert "Koramangala" in tiers["Premium"]
    assert "Electronic City" in tiers["Budget"]


def test_predict_returns_reasonable_rent():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    rent = r.json()["predicted_rent"]
    # Sanity: a 2BHK in Koramangala should be a plausible rent
    assert 10_000 < rent < 200_000


def test_premium_costs_more_than_budget():
    """Same flat, premium locality should predict higher rent than budget."""
    premium = client.post("/predict", json={**VALID_PAYLOAD, "locality": "Koramangala"})
    budget  = client.post("/predict", json={**VALID_PAYLOAD, "locality": "Electronic City"})
    assert premium.json()["predicted_rent"] > budget.json()["predicted_rent"]


def test_more_bhk_costs_more():
    """A 3BHK should generally cost more than a 1BHK, all else equal."""
    small = client.post("/predict", json={**VALID_PAYLOAD, "bhk": 1, "area_sqft": 600})
    large = client.post("/predict", json={**VALID_PAYLOAD, "bhk": 3, "area_sqft": 1600})
    assert large.json()["predicted_rent"] > small.json()["predicted_rent"]


def test_unknown_locality_rejected():
    r = client.post("/predict", json={**VALID_PAYLOAD, "locality": "Devanahalli"})
    assert r.status_code == 422  # validation error


def test_invalid_furnishing_rejected():
    r = client.post("/predict", json={**VALID_PAYLOAD, "furnishing": "Partially Furnished"})
    assert r.status_code == 422


def test_bhk_zero_rejected():
    r = client.post("/predict", json={**VALID_PAYLOAD, "bhk": 0})
    assert r.status_code == 422


def test_negative_area_rejected():
    r = client.post("/predict", json={**VALID_PAYLOAD, "area_sqft": -100})
    assert r.status_code == 422


def test_missing_required_field_rejected():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "locality"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_amenities_default_to_zero():
    """Amenity fields are optional and default to 0."""
    minimal = {
        "locality": "JP Nagar", "bhk": 2, "area_sqft": 900,
        "furnishing": "Semi-Furnished", "floor": 2, "total_floors": 6,
        "building_age_yrs": 8, "tenant_preferred": "Family",
        "dist_metro_km": 2.0, "dist_techpark_km": 4.0,
    }
    r = client.post("/predict", json=minimal)
    assert r.status_code == 200
