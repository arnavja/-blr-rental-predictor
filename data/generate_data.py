"""
generate_data.py
----------------
Generates a realistic synthetic Bengaluru rental dataset.

Why synthetic?
  Real rental data from 99acres/MagicBricks requires scraping (ToS issues)
  and is dynamic. Synthetic data lets us control distributions while keeping
  them realistic — rent ranges are calibrated to actual 2023-24 Bengaluru
  market rates from public listings.

Run:
    python data/generate_data.py
Output:
    data/bengaluru_rentals.csv  (~3000 rows)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000

# ─────────────────────────────────────────────────────────────
#  LOCALITY PROFILES
#  Each entry: (base_rent_1bhk, multiplier_2bhk, multiplier_3bhk,
#               dist_metro_km, dist_techpark_km, premium_score)
#  premium_score influences amenities probability
# ─────────────────────────────────────────────────────────────

LOCALITY_PROFILES = {
    # Premium
    "Koramangala":      (22000, 1.7, 2.5, 1.2, 2.1, 0.85),
    "Indiranagar":      (24000, 1.8, 2.6, 0.8, 3.5, 0.88),
    "Sadashivanagar":   (28000, 1.9, 2.8, 3.5, 6.0, 0.90),
    "Jayanagar":        (18000, 1.6, 2.3, 1.5, 5.0, 0.75),
    "Vasanth Nagar":    (26000, 1.8, 2.7, 1.0, 4.0, 0.87),

    # Upper-mid
    "HSR Layout":       (16000, 1.65, 2.4, 2.5, 1.8, 0.70),
    "BTM Layout":       (14000, 1.6, 2.3, 2.2, 2.5, 0.65),
    "JP Nagar":         (13000, 1.55, 2.2, 3.0, 5.5, 0.60),
    "Banashankari":     (12000, 1.5, 2.1, 3.5, 6.0, 0.55),
    "Yelahanka":        (11000, 1.5, 2.1, 4.0, 8.0, 0.50),

    # Mid
    "Marathahalli":     (13000, 1.6, 2.3, 3.5, 0.8, 0.62),
    "Whitefield":       (14000, 1.65, 2.4, 4.5, 0.5, 0.65),
    "Sarjapur Road":    (13500, 1.6, 2.3, 4.0, 1.2, 0.63),
    "Bellandur":        (13000, 1.55, 2.2, 3.8, 1.0, 0.60),
    "Mahadevapura":     (11500, 1.5, 2.1, 3.2, 1.5, 0.55),

    # Budget
    "Electronic City":  (9000,  1.5, 2.1, 5.5, 0.3, 0.40),
    "Hennur":           (9500,  1.45, 2.0, 4.5, 7.0, 0.38),
    "Bannerghatta Rd":  (10000, 1.5, 2.1, 4.0, 4.5, 0.42),
    "Begur":            (8500,  1.4, 1.9, 5.0, 3.5, 0.35),
    "Attibele":         (7500,  1.4, 1.9, 8.0, 2.5, 0.30),
}

LOCALITIES = list(LOCALITY_PROFILES.keys())

# Locality weights — popular areas appear more in listings
LOCALITY_WEIGHTS = [
    0.07, 0.07, 0.03, 0.04, 0.02,   # premium
    0.08, 0.07, 0.05, 0.04, 0.03,   # upper-mid
    0.07, 0.08, 0.06, 0.05, 0.04,   # mid
    0.06, 0.04, 0.05, 0.03, 0.02,   # budget
]

BHK_OPTIONS = [1, 2, 3]
BHK_WEIGHTS = [0.25, 0.50, 0.25]

FURNISHING = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
FURNISHING_WEIGHTS = [0.30, 0.40, 0.30]
FURNISHING_MULTIPLIER = {"Unfurnished": 1.0, "Semi-Furnished": 1.18, "Fully Furnished": 1.38}

TENANT = ["Family", "Bachelor", "Any"]
TENANT_WEIGHTS = [0.40, 0.25, 0.35]


def generate_dataset(n: int = 3000) -> pd.DataFrame:
    rows = []

    localities = np.random.choice(LOCALITIES, size=n, p=LOCALITY_WEIGHTS)
    bhks       = np.random.choice(BHK_OPTIONS, size=n, p=BHK_WEIGHTS)
    furnishing = np.random.choice(FURNISHING, size=n, p=FURNISHING_WEIGHTS)
    tenants    = np.random.choice(TENANT, size=n, p=TENANT_WEIGHTS)

    for i in range(n):
        loc  = localities[i]
        bhk  = bhks[i]
        furn = furnishing[i]
        ten  = tenants[i]

        base, m2, m3, d_metro, d_tech, premium = LOCALITY_PROFILES[loc]

        # Base rent by BHK
        if bhk == 1:
            base_rent = base
        elif bhk == 2:
            base_rent = base * m2
        else:
            base_rent = base * m3

        # Area (sq ft) — correlated with BHK
        if bhk == 1:
            area = np.random.normal(550, 80)
        elif bhk == 2:
            area = np.random.normal(950, 120)
        else:
            area = np.random.normal(1400, 180)
        area = max(300, area)

        # Building age
        building_age = np.random.choice(
            [1, 3, 6, 10, 15, 20],
            p=[0.10, 0.20, 0.25, 0.20, 0.15, 0.10]
        )
        age_discount = 1.0 - (building_age * 0.008)  # older = cheaper

        # Floor
        total_floors = np.random.choice([3, 5, 7, 10, 15, 20], p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
        floor        = np.random.randint(0, total_floors + 1)
        # Ground floor slight discount, higher floors slight premium
        floor_factor = 0.95 if floor == 0 else (1.0 + (floor / total_floors) * 0.05)

        # Amenities — probability driven by locality premium score
        p = premium
        gym           = int(np.random.random() < p * 0.6)
        swimming_pool = int(np.random.random() < p * 0.4)
        power_backup  = int(np.random.random() < p * 0.8)
        lift          = int(np.random.random() < p * 0.9)
        security      = int(np.random.random() < p * 0.85)
        parking       = int(np.random.random() < p * 0.75)
        club_house    = int(np.random.random() < p * 0.35)

        amenity_bonus = (
            gym * 1200 +
            swimming_pool * 1800 +
            power_backup * 500 +
            lift * 400 +
            security * 600 +
            parking * 800 +
            club_house * 1000
        )

        # Metro and tech park proximity discount/premium
        metro_factor    = 1.0 + max(0, (3.0 - d_metro) * 0.025)   # closer = more expensive
        techpark_factor = 1.0 + max(0, (4.0 - d_tech) * 0.018)

        # Furnishing multiplier
        furn_mult = FURNISHING_MULTIPLIER[furn]

        # Compose rent
        rent = (
            base_rent
            * furn_mult
            * age_discount
            * floor_factor
            * metro_factor
            * techpark_factor
        ) + amenity_bonus

        # Add noise (±12%)
        noise = np.random.normal(1.0, 0.12)
        rent  = max(5000, rent * noise)
        rent  = round(rent / 500) * 500   # round to nearest 500

        # Jitter distances slightly per listing
        dist_metro   = round(max(0.2, d_metro + np.random.normal(0, 0.3)), 1)
        dist_techpark = round(max(0.2, d_tech + np.random.normal(0, 0.5)), 1)

        rows.append({
            "locality":           loc,
            "bhk":                bhk,
            "area_sqft":          round(area),
            "furnishing":         furn,
            "floor":              floor,
            "total_floors":       total_floors,
            "building_age_yrs":   building_age,
            "tenant_preferred":   ten,
            "gym":                gym,
            "swimming_pool":      swimming_pool,
            "power_backup":       power_backup,
            "lift":               lift,
            "security":           security,
            "parking":            parking,
            "club_house":         club_house,
            "dist_metro_km":      dist_metro,
            "dist_techpark_km":   dist_techpark,
            "monthly_rent":       int(rent),
        })

    df = pd.DataFrame(rows)
    print(f"Generated {len(df)} rental listings.")
    print(f"Rent range: ₹{df['monthly_rent'].min():,} – ₹{df['monthly_rent'].max():,}")
    print(f"Median rent: ₹{df['monthly_rent'].median():,.0f}")
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(N)
    df.to_csv("data/bengaluru_rentals.csv", index=False)
    print("\nSaved → data/bengaluru_rentals.csv")
    print(df.head())
