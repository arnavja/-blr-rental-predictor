# 🏙️ Bengaluru Rental Price Predictor

Predict the **monthly rent (₹)** of a residential flat in Bengaluru based on locality, BHK, area, furnishing, amenities, and proximity to metro stations and tech parks.

---

## 🎯 Problem Statement

Bengaluru's rental market is one of India's most dynamic — rents vary dramatically based on proximity to tech corridors (Whitefield, Electronic City, Marathahalli) and premium neighbourhoods (Koramangala, Indiranagar). 

This project builds a **regression model** to estimate fair rental price from flat attributes — useful for:
- Tenants checking if a listing is overpriced
- Landlords pricing their property competitively
- Real estate platforms building automated valuation tools

---

## 📦 Dataset

- **Source**: Synthetically generated with realistic Bengaluru market distributions (calibrated to 2023-24 public listing data from 99acres, MagicBricks)
- **Size**: 3,000 rental listings across 20 localities
- **Features**: 18 raw features → 24 engineered features

### Localities covered

| Tier | Localities |
|---|---|
| Premium | Koramangala, Indiranagar, Sadashivanagar, Vasanth Nagar |
| Upper-Mid | HSR Layout, BTM Layout, Whitefield, Sarjapur Road, Jayanagar |
| Mid | Marathahalli, Bellandur, JP Nagar, Mahadevapura, Yelahanka |
| Budget | Electronic City, Hennur, Bannerghatta Rd, Begur, Attibele |

---

## 🛠️ Feature Engineering

| Feature | Description |
|---|---|
| `locality_tier` | Ordinal encoding (1=premium → 4=budget) instead of 20 one-hot columns |
| `area_per_bhk` | Area ÷ BHK — a 400 sq ft 2BHK is very different from 1000 sq ft 2BHK |
| `floor_ratio` | Floor ÷ total floors — relative position in building |
| `amenity_score` | Sum of 7 binary amenities (gym, pool, parking, etc.) |
| `metro_proximity` | 1 / (distance to metro + 0.5) — inverse distance for non-linearity |
| `techpark_proximity` | 1 / (distance to tech park + 0.5) |
| `connectivity_score` | metro_proximity + techpark_proximity |
| `tier_x_furnishing` | Interaction: furnished in premium area >> furnished in budget area |
| `age_bucket` | Binned building age: new / mid / old |

**Target transformation**: Log(rent) — rent is right-skewed. Modelling on log scale makes errors scale-invariant and improves all model metrics.

---

## 🤖 Models Compared

| Model | Notes |
|---|---|
| Ridge Regression | Baseline — linear, interpretable, fast |
| Random Forest | Ensemble — captures non-linearity, robust |
| XGBoost (tuned) | Best performer — gradient boosting + RandomizedSearchCV |

---

## 📊 Results

| Model | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| Ridge Regression | ₹5,840 | ₹4,210 | 0.821 | 18.4% |
| Random Forest | ₹3,920 | ₹2,780 | 0.912 | 12.1% |
| **XGBoost (tuned)** | **₹3,240** | **₹2,310** | **0.938** | **10.3%** |

> XGBoost predicts rent within ~₹2,300 on average — strong enough for a pricing tool.

---

## 🔑 Key Design Decisions (Interview-Ready)

1. **Log-transform the target** — Rent is right-skewed. Log makes the regression problem symmetric and reduces the influence of outliers like luxury flats.
2. **Locality tier instead of one-hot** — 20 one-hot columns add noise. Ordinal tier encoding preserves the hierarchy while reducing dimensionality.
3. **Inverse distance features** — `1 / distance` captures the non-linear "premium" for being very close to metro or tech park.
4. **Interaction feature** — `tier × furnishing` because a fully furnished flat in Koramangala commands a much higher premium than furnished in Electronic City.
5. **RandomizedSearchCV** — Faster than GridSearchCV for large hyperparameter spaces; 25 random samples across 7 parameters.
6. **SHAP for explainability** — Goes beyond feature importance to show *how* each feature pushes a specific prediction up or down.

---

## 🚀 How to Run

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/blr-rental-predictor
cd blr-rental-predictor
pip install -r requirements.txt

# 2. Generate dataset
python data/generate_data.py

# 3. Train all models
python src/train.py
```

---

## 📁 Project Structure

```
blr-rental-predictor/
├── README.md
├── requirements.txt
├── data/
│   ├── generate_data.py      # synthetic data generation with realistic BLR distributions
│   └── bengaluru_rentals.csv # generated dataset (3000 rows)
├── src/
│   ├── features.py           # all feature engineering
│   ├── train.py              # full pipeline
│   └── evaluate.py           # metrics, plots, SHAP
└── outputs/                  # all saved plots and models
    ├── eda_overview.png
    ├── correlation_heatmap.png
    ├── actual_vs_pred_*.png
    ├── residuals_*.png
    ├── shap_summary.png
    └── xgb_model.pkl
```

---

## 💡 What I Learned

- Log-transforming skewed targets significantly improves regression performance across all models
- Inverse distance features (`1/dist`) outperform raw distance because the value of being near a metro drops off non-linearly
- SHAP revealed that `area_per_bhk` matters more than raw `area_sqft` — a cramped 2BHK is penalised
- Locality tier was the single most important feature, confirming that location dominates Bengaluru rental pricing

---

## 🔭 Future Work

- Scrape live listings from 99acres for real-time data
- Add school/hospital proximity features
- Build a Streamlit app for interactive rent estimation
- Extend to other cities (Mumbai, Pune, Hyderabad)
