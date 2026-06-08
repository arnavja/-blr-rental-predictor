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

## 🌐 Prediction API (FastAPI)

The trained model is served as a REST API so anyone can get a rent prediction over HTTP — no Python required on the client side.

```bash
# Start the API locally
uvicorn api:app --reload

# Open the interactive docs (try predictions in the browser):
#   http://127.0.0.1:8000/docs
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/` | Health check |
| `GET`  | `/localities` | List all 20 valid localities, grouped by tier |
| `POST` | `/predict` | Predict monthly rent from flat details |

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "locality": "Koramangala", "bhk": 2, "area_sqft": 1100,
    "furnishing": "Fully Furnished", "floor": 3, "total_floors": 10,
    "building_age_yrs": 5, "tenant_preferred": "Any",
    "dist_metro_km": 0.8, "dist_techpark_km": 2.5,
    "gym": 1, "power_backup": 1, "lift": 1, "security": 1, "parking": 1
  }'
```

```json
{ "predicted_rent": 58048, "predicted_rent_formatted": "₹58,048", ... }
```

The API reuses the **exact same** `engineer_features()` pipeline as training — guaranteeing no training/serving skew. Inputs are validated with Pydantic (unknown localities, invalid furnishing, `bhk=0`, negative area are all rejected with clear errors).

---

## ✅ Testing

```bash
pytest tests/ -v
```

24 tests covering:
- **Feature engineering** — correctness of every engineered feature, boundary conditions (age buckets, ground floor), input immutability (`.copy()` purity), and documented edge cases (unknown locality → NaN)
- **API** — health checks, prediction sanity (premium > budget, 3BHK > 1BHK), and validation rejection of bad inputs

---

## 🚀 Deployment

Two ready-to-use options:

**Render (free, one-click):** push to GitHub → New + → Blueprint → select repo. `render.yaml` is auto-detected.

**Docker:**
```bash
docker build -t blr-rental-predictor .
docker run -p 8000:8000 blr-rental-predictor
```

---

## 📁 Project Structure

```
blr-rental-predictor/
├── README.md
├── requirements.txt
├── api.py                    # FastAPI prediction server
├── Dockerfile                # containerised deployment
├── render.yaml               # one-click Render deploy config
├── data/
│   ├── generate_data.py      # synthetic data generation with realistic BLR distributions
│   └── bengaluru_rentals.csv # generated dataset (3000 rows)
├── src/
│   ├── features.py           # all feature engineering (shared by train + API)
│   ├── train.py              # full training pipeline
│   └── evaluate.py           # metrics, plots, SHAP
├── tests/
│   ├── test_features.py      # unit tests for feature engineering
│   └── test_api.py           # integration tests for the API
└── outputs/                  # saved plots and models
    ├── eda_overview.png
    ├── correlation_heatmap.png
    ├── actual_vs_pred_*.png
    ├── residuals_*.png
    ├── shap_summary.png
    ├── scaler.pkl
    └── xgb_model.pkl         # committed so the API runs without retraining
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
