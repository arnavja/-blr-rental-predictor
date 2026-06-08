"""
test_features.py
----------------
Unit tests for the feature engineering pipeline.

Run:  pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

from src.features import (
    engineer_features, FEATURE_COLS, TARGET_COL,
    LOCALITY_TIER, FURNISHING_MAP, TENANT_MAP,
)


# ── A reusable valid raw row ──────────────────────────────────────────────────
def make_row(**overrides):
    base = {
        "locality": "Koramangala",
        "bhk": 2,
        "area_sqft": 1000.0,
        "furnishing": "Fully Furnished",
        "floor": 3,
        "total_floors": 10,
        "building_age_yrs": 5,
        "tenant_preferred": "Any",
        "dist_metro_km": 1.0,
        "dist_techpark_km": 2.0,
        "gym": 1, "swimming_pool": 0, "power_backup": 1,
        "lift": 1, "security": 1, "parking": 1, "club_house": 0,
        "monthly_rent": 45000,
    }
    base.update(overrides)
    return pd.DataFrame([base])


# ── Core correctness ──────────────────────────────────────────────────────────
def test_all_feature_cols_present():
    """Output must contain every column the model expects."""
    out = engineer_features(make_row())
    for col in FEATURE_COLS:
        assert col in out.columns, f"Missing engineered feature: {col}"


def test_locality_tier_mapping():
    assert engineer_features(make_row(locality="Koramangala"))["locality_tier"].iloc[0] == 1
    assert engineer_features(make_row(locality="Electronic City"))["locality_tier"].iloc[0] == 4


def test_furnishing_encoding():
    assert engineer_features(make_row(furnishing="Unfurnished"))["furnishing_enc"].iloc[0] == 0
    assert engineer_features(make_row(furnishing="Fully Furnished"))["furnishing_enc"].iloc[0] == 2


def test_area_per_bhk_arithmetic():
    out = engineer_features(make_row(area_sqft=1000, bhk=2))
    assert out["area_per_bhk"].iloc[0] == 500.0


def test_floor_ratio_range():
    out = engineer_features(make_row(floor=5, total_floors=10))
    assert out["floor_ratio"].iloc[0] == pytest.approx(0.5, abs=1e-6)


def test_ground_floor_ratio_is_zero():
    out = engineer_features(make_row(floor=0, total_floors=10))
    assert out["floor_ratio"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_amenity_score_sum():
    # gym=1, power_backup=1, lift=1, security=1, parking=1 → 5
    out = engineer_features(make_row())
    assert out["amenity_score"].iloc[0] == 5


def test_amenity_score_bounds():
    all_on = engineer_features(make_row(gym=1, swimming_pool=1, power_backup=1,
                                        lift=1, security=1, parking=1, club_house=1))
    all_off = engineer_features(make_row(gym=0, swimming_pool=0, power_backup=0,
                                         lift=0, security=0, parking=0, club_house=0))
    assert all_on["amenity_score"].iloc[0] == 7
    assert all_off["amenity_score"].iloc[0] == 0


def test_connectivity_score_is_sum_of_proximities():
    out = engineer_features(make_row())
    expected = out["metro_proximity"].iloc[0] + out["techpark_proximity"].iloc[0]
    assert out["connectivity_score"].iloc[0] == pytest.approx(expected)


def test_age_bucket_boundaries():
    # bins=[-1, 3, 10, 100], labels=[2, 1, 0], right-inclusive
    assert engineer_features(make_row(building_age_yrs=2))["age_bucket"].iloc[0] == 2   # new
    assert engineer_features(make_row(building_age_yrs=3))["age_bucket"].iloc[0] == 2   # boundary → new
    assert engineer_features(make_row(building_age_yrs=7))["age_bucket"].iloc[0] == 1   # mid
    assert engineer_features(make_row(building_age_yrs=20))["age_bucket"].iloc[0] == 0  # old


# ── Safety / purity ───────────────────────────────────────────────────────────
def test_does_not_mutate_input():
    """engineer_features must not modify the caller's DataFrame (it uses .copy())."""
    df = make_row()
    cols_before = set(df.columns)
    _ = engineer_features(df)
    assert set(df.columns) == cols_before, "Input DataFrame was mutated!"


def test_handles_tenant_alias_column():
    """Code accepts either 'tenant' or 'tenant_preferred'."""
    df = make_row()
    df = df.rename(columns={"tenant_preferred": "tenant"})
    out = engineer_features(df)
    assert out["tenant_enc"].iloc[0] == TENANT_MAP["Any"]


# ── Documented edge cases (these currently EXPOSE limitations) ─────────────────
def test_unknown_locality_produces_nan():
    """
    Known limitation: unknown localities map to NaN silently.
    The API layer guards against this with validation — but the raw
    feature function does not. This test documents that behaviour.
    """
    out = engineer_features(make_row(locality="Devanahalli"))
    assert pd.isna(out["locality_tier"].iloc[0])
