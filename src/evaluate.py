"""
evaluate.py
-----------
Evaluation utilities for regression: RMSE, MAE, R², residual plots,
prediction vs actual plot, and SHAP explainability.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  REGRESSION METRICS
# ─────────────────────────────────────────────

def regression_metrics(name: str, y_true, y_pred) -> dict:
    """
    RMSE: penalises large errors more (squared). Good for outlier-sensitivity.
    MAE:  average absolute error. Easier to interpret in ₹ terms.
    R²:   fraction of variance explained. 1.0 = perfect, 0.0 = predicting mean.
    MAPE: mean absolute % error — intuitive for non-technical audiences.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{'='*48}")
    print(f"  {name}")
    print(f"{'='*48}")
    print(f"  RMSE  : ₹{rmse:>10,.0f}")
    print(f"  MAE   : ₹{mae:>10,.0f}")
    print(f"  R²    :  {r2:>10.4f}")
    print(f"  MAPE  :  {mape:>9.2f}%")

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def print_summary_table(results: dict):
    print("\n" + "="*62)
    print(f"  {'Model':<25} {'RMSE':>10} {'MAE':>10} {'R²':>7} {'MAPE':>7}")
    print("="*62)
    for name, m in results.items():
        print(f"  {name:<25} ₹{m['rmse']:>8,.0f} ₹{m['mae']:>8,.0f} "
              f"{m['r2']:>6.3f} {m['mape']:>6.1f}%")
    print("="*62)


# ─────────────────────────────────────────────
#  ACTUAL vs PREDICTED PLOT
# ─────────────────────────────────────────────

def plot_actual_vs_predicted(name: str, y_true, y_pred, save=True):
    """
    Perfect model → all points on the diagonal.
    Scatter above diagonal → model underestimates.
    Scatter below diagonal → model overestimates.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, s=18, color="#2563EB", edgecolors="none")

    lims = [min(y_true.min(), y_pred.min()) * 0.95,
            max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual Rent (₹)", fontsize=12)
    ax.set_ylabel("Predicted Rent (₹)", fontsize=12)
    ax.set_title(f"Actual vs Predicted — {name}", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, f"actual_vs_pred_{name.replace(' ', '_').lower()}.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────
#  RESIDUAL PLOT
# ─────────────────────────────────────────────

def plot_residuals(name: str, y_true, y_pred, save=True):
    """
    Residuals (actual - predicted) should be:
    - Centred around 0 (no systematic bias)
    - Evenly spread (no heteroscedasticity)
    - No pattern (model has captured all structure)
    """
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs predicted
    ax1.scatter(y_pred, residuals, alpha=0.3, s=15, color="#16A34A", edgecolors="none")
    ax1.axhline(0, color="red", lw=1.5, linestyle="--")
    ax1.set_xlabel("Predicted Rent (₹)")
    ax1.set_ylabel("Residual (₹)")
    ax1.set_title("Residuals vs Predicted")
    ax1.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
    ax1.grid(alpha=0.3)

    # Distribution of residuals
    ax2.hist(residuals, bins=50, color="#9333EA", edgecolor="white", alpha=0.8)
    ax2.axvline(0, color="red", lw=1.5, linestyle="--")
    ax2.set_xlabel("Residual (₹)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Residual Analysis — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, f"residuals_{name.replace(' ', '_').lower()}.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────
#  SHAP EXPLAINABILITY
# ─────────────────────────────────────────────

def plot_shap(model, X_test, feature_names: list, save=True):
    """
    SHAP (SHapley Additive exPlanations) shows HOW each feature
    pushes predictions up or down for each individual listing.

    This is model-agnostic and grounded in game theory.
    It answers: 'Why did the model predict ₹32,000 for this flat?'

    Most student projects stop at feature importance.
    SHAP takes you a level deeper — interviewers love this.
    """
    try:
        import shap
    except ImportError:
        print("  shap not installed. Run: pip install shap")
        return

    print("\n  Computing SHAP values (this may take ~30 seconds)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot — beeswarm
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title("SHAP Feature Impact (XGBoost)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.show()
