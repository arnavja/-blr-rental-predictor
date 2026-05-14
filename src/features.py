"""
features.py
-----------
Feature engineering for the Bengaluru Rental Price Predictor.

Key ideas:
  1. Log-transform the target (rent is right-skewed — a ₹5k difference
     at ₹10k matters more than at ₹50k. Log makes errors symmetric.)
  2. Interaction features — area × BHK, premium locality × furnishing
  3. Locality tier encoding (instead of raw one-hot — avoids high cardinality)
  4. Amenity count as a single score
"""

import pandas as pd
import numpy as np


# ── Locality tier map ────────────────────────────────────────────────────────
# Instead of 20 one-hot columns, encode locality as a tier (1-4).
# This reduces dimensionality and teaches the model ordinal locality value.

LOCALITY_TIER = {
    # Tier 1 — Premium
    "Koramangala": 1, "Indiranagar": 1, "Sadashivanagar": 1,
    "Vasanth Nagar": 1,

    # Tier 2 — Upper-mid
    "Jayanagar": 2, "HSR Layout": 2, "BTM Layout": 2,
    "Whitefield": 2, "Sarjapur Road": 2,

    # Tier 3 — Mid
    "JP Nagar": 3, "Banashankari": 3, "Marathahalli": 3,
    "Bellandur": 3, "Mahadevapura": 3, "Yelahanka": 3,

    # Tier 4 — Budget
    "Electronic City": 4, "Hennur": 4, "Bannerghatta Rd": 4,
    "Begur": 4, "Attibele": 4,
}

FURNISHING_MAP = {"Unfurnished": 0, "Semi-Furnished": 1, "Fully Furnished": 2}
TENANT_MAP     = {"Family": 0, "Bachelor": 1, "Any": 2}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw rental DataFrame, returns feature matrix and target.
    """
    df = df.copy()

    # ── Encode categoricals ──────────────────────────────────────────────────
    df["locality_tier"] = df["locality"].map(LOCALITY_TIER)
    df["furnishing_enc"] = df["furnishing"].map(FURNISHING_MAP)
    df["tenant_enc"]     = df["tenant"].map(TENANT_MAP) if "tenant" in df.columns \
                           else df["tenant_preferred"].map(TENANT_MAP)

    # ── Floor ratio (more meaningful than raw floor number) ─────────────────
    # 0.0 = ground floor, 1.0 = top floor
    df["floor_ratio"] = df["floor"] / (df["total_floors"] + 1e-9)

    # ── Amenity score ────────────────────────────────────────────────────────
    # Sum of all binary amenity columns → single "quality of building" score
    amenity_cols = ["gym", "swimming_pool", "power_backup",
                    "lift", "security", "parking", "club_house"]
    df["amenity_score"] = df[amenity_cols].sum(axis=1)

    # ── Interaction features ─────────────────────────────────────────────────
    # Area per BHK: studio-sized 2BHK should cost less than spacious 2BHK
    df["area_per_bhk"] = df["area_sqft"] / df["bhk"]

    # Tier × Furnishing: furnished flat in premium area >> furnished flat in budget area
    df["tier_x_furnishing"] = df["locality_tier"] * df["furnishing_enc"]

    # Proximity score: inverse of distance (closer = higher score)
    df["metro_proximity"]    = 1 / (df["dist_metro_km"] + 0.5)
    df["techpark_proximity"] = 1 / (df["dist_techpark_km"] + 0.5)

    # Combined connectivity score
    df["connectivity_score"] = df["metro_proximity"] + df["techpark_proximity"]

    # Age bucket: new (≤3yr), mid (4-10yr), old (>10yr)
    df["age_bucket"] = pd.cut(
        df["building_age_yrs"],
        bins=[-1, 3, 10, 100],
        labels=[2, 1, 0]   # newer = higher label
    ).astype(int)

    return df


# Columns used as model input
FEATURE_COLS = [
    "locality_tier",
    "bhk",
    "area_sqft",
    "area_per_bhk",
    "furnishing_enc",
    "floor_ratio",
    "total_floors",
    "age_bucket",
    "building_age_yrs",
    "tenant_enc",
    "amenity_score",
    "gym",
    "swimming_pool",
    "power_backup",
    "lift",
    "security",
    "parking",
    "club_house",
    "dist_metro_km",
    "dist_techpark_km",
    "metro_proximity",
    "techpark_proximity",
    "connectivity_score",
    "tier_x_furnishing",
]

TARGET_COL = "monthly_rent"
