"""
train.py
End-to-end pipeline for the Bengaluru Rental Price Predictor.

Run:
    python data/generate_data.py   # create dataset first
    python src/train.py            # train all models

Steps:
  1. Load data
  2. EDA — rent distribution, locality heatmap, correlation
  3. Feature engineering
  4. Train/test split (random OK here — no time dependency)
  5. Train 3 models: Linear Regression → Random Forest → XGBoost
  6. Hyperparameter tune XGBoost with RandomizedSearchCV
  7. Evaluate: RMSE, MAE, R², MAPE
  8. SHAP explainability
  9. Save best model
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import joblib

from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_squared_error, r2_score
from xgboost                 import XGBRegressor

from src.features import engineer_features, FEATURE_COLS, TARGET_COL
from src.evaluate  import (regression_metrics, print_summary_table,
                            plot_actual_vs_predicted, plot_residuals, plot_shap)

os.makedirs("outputs", exist_ok=True)


#  STEP 1 — LOAD DATA

def load_data(path: str = "data/bengaluru_rentals.csv") -> pd.DataFrame:
    print(f"\n[1/6] Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"  Shape : {df.shape}")
    print(f"  Cols  : {list(df.columns)}")
    print(f"\n  Rent stats (₹):")
    print(df["monthly_rent"].describe().apply(lambda x: f"₹{x:,.0f}").to_string())
    return df


#  STEP 2 — EDA

def run_eda(df: pd.DataFrame):
    print("\n[2/6] Running EDA...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bengaluru Rental Market — EDA", fontsize=15, fontweight="bold")

    # Plot 1: Rent distribution
    axes[0, 0].hist(df["monthly_rent"], bins=60, color="#2563EB",
                    edgecolor="white", alpha=0.85)
    axes[0, 0].set_title("Rent Distribution")
    axes[0, 0].set_xlabel("Monthly Rent (₹)")
    axes[0, 0].xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Log-rent distribution (more normal → better for modelling)
    axes[0, 1].hist(np.log(df["monthly_rent"]), bins=60, color="#16A34A",
                    edgecolor="white", alpha=0.85)
    axes[0, 1].set_title("Log(Rent) Distribution — more symmetric")
    axes[0, 1].set_xlabel("log(Monthly Rent)")
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Median rent by locality (top 15)
    med_rent = (df.groupby("locality")["monthly_rent"]
                  .median()
                  .sort_values(ascending=False))
    axes[1, 0].barh(med_rent.index[::-1], med_rent.values[::-1],
                    color="#9333EA", alpha=0.85)
    axes[1, 0].set_title("Median Rent by Locality")
    axes[1, 0].xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
    axes[1, 0].grid(axis="x", alpha=0.3)

    # Plot 4: Rent by BHK × furnishing
    pivot = df.groupby(["bhk", "furnishing"])["monthly_rent"].median().unstack()
    pivot.plot(kind="bar", ax=axes[1, 1], color=["#F59E0B", "#2563EB", "#16A34A"],
               edgecolor="white", alpha=0.85)
    axes[1, 1].set_title("Median Rent by BHK × Furnishing")
    axes[1, 1].set_xlabel("BHK")
    axes[1, 1].yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
    axes[1, 1].legend(title="Furnishing", fontsize=9)
    axes[1, 1].grid(axis="y", alpha=0.3)
    axes[1, 1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig("outputs/eda_overview.png", dpi=150)
    plt.show()
    print("  Saved → outputs/eda_overview.png")

    # Correlation heatmap
    num_cols = ["area_sqft", "bhk", "building_age_yrs",
                "dist_metro_km", "dist_techpark_km", "monthly_rent"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=150)
    plt.show()
    print("  Saved → outputs/correlation_heatmap.png")


#  STEP 3 — PREPARE FEATURES

def prepare(df: pd.DataFrame):
    print("\n[3/6] Engineering features...")
    df = engineer_features(df)
    X  = df[FEATURE_COLS].values
    y  = df[TARGET_COL].values

    # Log-transform target
    # Why? Rent is right-skewed. Log makes errors scale-invariant:
    # being off by ₹5k on a ₹10k flat is very different from ₹5k on ₹80k.
    y_log = np.log(y)

    X_train, X_test, y_train, y_test, y_log_train, y_log_test = train_test_split(
        X, y, y_log, test_size=0.20, random_state=42
    )

    print(f"  Train : {len(X_train)} samples")
    print(f"  Test  : {len(X_test)} samples")
    print(f"  Features: {len(FEATURE_COLS)}")
    return X_train, X_test, y_train, y_test, y_log_train, y_log_test


#  STEP 4 — TRAIN MODELS

def train_models(X_train, X_test, y_train, y_test, y_log_train, y_log_test):
    print("\n[4/6] Training models...")
    results = {}

    # Model 1: Ridge Regression (baseline)
    print("\n  Training Ridge Regression (baseline)...")
    scaler      = StandardScaler()
    Xtr_scaled  = scaler.fit_transform(X_train)
    Xte_scaled  = scaler.transform(X_test)

    ridge = Ridge(alpha=10.0)
    ridge.fit(Xtr_scaled, y_log_train)
    ridge_log_pred = ridge.predict(Xte_scaled)
    ridge_pred     = np.exp(ridge_log_pred)    # back to ₹

    m = regression_metrics("Ridge Regression", y_test, ridge_pred)
    plot_actual_vs_predicted("Ridge Regression", y_test, ridge_pred)
    results["Ridge Regression"] = m

    # Model 2: Random Forest
    print("\n  Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_log_train)
    rf_log_pred = rf.predict(X_test)
    rf_pred     = np.exp(rf_log_pred)

    m = regression_metrics("Random Forest", y_test, rf_pred)
    plot_actual_vs_predicted("Random Forest", y_test, rf_pred)
    results["Random Forest"] = m

    # Model 3: XGBoost + RandomizedSearchCV
    print("\n  Training XGBoost with hyperparameter search...")
    param_dist = {
        "n_estimators":      [200, 300, 400, 500],
        "max_depth":         [3, 4, 5, 6],
        "learning_rate":     [0.01, 0.03, 0.05, 0.08],
        "subsample":         [0.7, 0.8, 0.9],
        "colsample_bytree":  [0.6, 0.7, 0.8],
        "reg_alpha":         [0, 0.1, 0.5],
        "reg_lambda":        [1, 2, 5],
    }
    xgb_base = XGBRegressor(eval_metric="rmse", random_state=42, verbosity=0)
    search    = RandomizedSearchCV(
        xgb_base, param_dist,
        n_iter=25,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    search.fit(X_train, y_log_train)
    xgb = search.best_estimator_
    print(f"  Best params: {search.best_params_}")

    xgb_log_pred = xgb.predict(X_test)
    xgb_pred     = np.exp(xgb_log_pred)

    m = regression_metrics("XGBoost (tuned)", y_test, xgb_pred)
    plot_actual_vs_predicted("XGBoost (tuned)", y_test, xgb_pred)
    plot_residuals("XGBoost (tuned)", y_test, xgb_pred)
    results["XGBoost (tuned)"] = m

    return results, xgb, scaler, xgb_pred, y_test


#  STEP 5 — SHAP EXPLAINABILITY

def explain(xgb_model, X_test):
    print("\n[5/6] SHAP explainability...")
    plot_shap(xgb_model, X_test, FEATURE_COLS)


#  STEP 6 — SAVE & SUMMARISE

def save_and_summarise(results, xgb_model, scaler):
    print("\n[6/6] Summary & saving...")
    print_summary_table(results)

    joblib.dump(xgb_model, "outputs/xgb_model.pkl")
    joblib.dump(scaler,    "outputs/scaler.pkl")
    print("\n  Saved → outputs/xgb_model.pkl")
    print("  Saved → outputs/scaler.pkl")
    print("\n✅ Done. Check outputs/ for all plots and saved model.")


#  MAIN

if __name__ == "__main__":
    df = load_data("data/bengaluru_rentals.csv")
    run_eda(df)
    X_train, X_test, y_train, y_test, y_log_train, y_log_test = prepare(df)
    results, xgb, scaler, xgb_pred, y_test = train_models(
        X_train, X_test, y_train, y_test, y_log_train, y_log_test
    )
    explain(xgb, X_test)
    save_and_summarise(results, xgb, scaler)
