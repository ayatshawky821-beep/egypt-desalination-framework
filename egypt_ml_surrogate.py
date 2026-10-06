"""
egypt_ml_surrogate.py

Machine-learning surrogate models layered on top of the physics-based
desalination_plant_design.py framework.

WHY: The physics-based model (Sections 2 and Appendix A of the paper) is
cheap to evaluate once, but sweeping it across many design variables at once
(TDS x recovery x capacity x water type, for optimization or Monte Carlo
uncertainty analysis) means re-running the full pretreatment -> filtration
-> economics pipeline thousands of times. This script:
  1. Uses the physics-based framework itself as a "ground truth" generator,
     sampling a wide range of feed TDS, system recovery, plant capacity, and
     water type (N_SAMPLES design points, expanded from an earlier 2,000-
     sample version to 6,000 for better coverage of the design space).
  2. Trains and compares THREE supervised regression models on two targets
     (LCOW and specific energy consumption, SEC): a Random Forest (the
     original baseline), a Gradient Boosting Regressor, and a Multi-Layer
     Perceptron (MLP) neural network, each evaluated with k-fold cross-
     validation and a held-out test set (R^2, MAE, RMSE).
  3. Reports which model generalizes best and which inputs matter most
     (permutation/Gini feature importance).
  4. Demonstrates a two-tier "surrogate for breadth, physics for
     verification" optimization pattern: the fastest validated surrogate
     screens a large grid, and the physics-based model verifies the land-
     area constraint only for the most promising candidates.

This demonstrates how an AI/ML tool can accelerate design-space exploration
built on top of an engineering-grounded model, without inventing new
physics: every surrogate here is only ever as good as the physics-based
training data it learns from, and its accuracy is reported honestly rather
than assumed. For a genuinely adaptive, sample-efficient AI search over the
real physics model (not a surrogate), see ai_design_optimizer.py.

Run:
    python3 egypt_ml_surrogate.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from desalination_plant_design import (
    FeedWaterQuality,
    PlantCapacity,
    PlantFlows,
    ReverseOsmosisSystem,
    EconomicAnalysis,
)

plt.rcParams.update({
    "figure.figsize": (7, 6),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10.5,
})

RNG = np.random.default_rng(42)
N_SAMPLES = 6000  # expanded from an earlier 2,000-sample version


# --------------------------------------------------------------------------
# 1. Generate training data from the physics-based framework
# --------------------------------------------------------------------------

def sample_design_point():
    """Draw one random, physically plausible design point and evaluate the
    physics-based framework on it, returning (features, [LCOW, SEC])."""
    is_seawater = RNG.random() < 0.3

    if is_seawater:
        tds = RNG.uniform(28000, 45000)
        recovery = RNG.uniform(0.35, 0.55)
        water_type = "seawater"
    else:
        tds = RNG.uniform(500, 15000)
        recovery = RNG.uniform(0.50, 0.90)
        water_type = "brackish"

    capacity_m3d = float(10 ** RNG.uniform(np.log10(2000), np.log10(150000)))  # log-uniform

    feed = FeedWaterQuality(tds_mg_l=tds, water_type=water_type)
    cap = PlantCapacity(permeate_flow_m3d=capacity_m3d, recovery=recovery)
    flows = PlantFlows.from_capacity(cap)

    ro = ReverseOsmosisSystem(
        flow_m3d=flows.feed_to_membranes_m3d,
        permeate_m3d=flows.permeate_m3d,
        concentrate_m3d=flows.concentrate_m3d,
        feed_tds_mg_l=tds,
        water_type=water_type,
    ).design()
    sec_lo, sec_hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
    sec_mid = (sec_lo + sec_hi) / 2

    econ = EconomicAnalysis(
        capacity_m3d=capacity_m3d,
        water_type=water_type,
        specific_energy_kwh_m3=sec_mid,
    ).run_full_analysis()
    lcow = econ["lcow"]["lcow_usd_m3"]

    features = [tds, recovery, capacity_m3d, 1.0 if is_seawater else 0.0]
    targets = [lcow, sec_mid]
    return features, targets


def build_dataset(n=N_SAMPLES):
    X, Y = [], []
    for _ in range(n):
        feats, targets = sample_design_point()
        X.append(feats)
        Y.append(targets)
    return np.array(X), np.array(Y)  # Y columns: [LCOW, SEC]


FEATURE_NAMES = ["Feed TDS (mg/L)", "Recovery (fraction)", "Capacity (m3/d)", "Is seawater (0/1)"]
TARGET_NAMES = ["LCOW (USD/m3)", "SEC (kWh/m3)"]


# --------------------------------------------------------------------------
# 2. Train and compare three model families (Random Forest, Gradient
#    Boosting, and a Neural Network), each on both targets.
# --------------------------------------------------------------------------

def build_models():
    """Returns {model_name: unfitted estimator or pipeline}. The neural
    network is wrapped in a StandardScaler pipeline since MLPs are
    sensitive to unscaled inputs (TDS ~10^4 vs. recovery ~10^0), unlike the
    tree-based models which are scale-invariant."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        ),
        "Neural Network (MLP)": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                alpha=1e-4, max_iter=2000, random_state=42, early_stopping=True,
                n_iter_no_change=20,
            ),
        ),
    }


def train_and_compare(target_idx=0, target_name="LCOW (USD/m3)"):
    """Trains all three model families on one target column, reports 5-fold
    cross-validated R^2 (on the training split) and held-out test-set R^2 /
    MAE / RMSE for each, and returns the best-performing fitted model by
    test R^2 along with the comparison table."""
    X, Y = build_dataset()
    y = Y[:, target_idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    fitted = {}
    for name, model in build_models().items():
        t0 = time.time()
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        model.fit(X_train, y_train)
        train_time_s = time.time() - t0

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5

        results[name] = {
            "cv_r2_mean": cv_scores.mean(), "cv_r2_std": cv_scores.std(),
            "test_r2": r2, "test_mae": mae, "test_rmse": rmse,
            "train_time_s": train_time_s,
        }
        fitted[name] = model

    print(f"\n=== Model comparison for target: {target_name} ===")
    print(f"{'Model':<22} {'CV R2 (mean+-std)':<22} {'Test R2':<10} {'Test MAE':<12} {'Test RMSE':<12} {'Fit time (s)'}")
    for name, r in results.items():
        print(f"{name:<22} {r['cv_r2_mean']:.4f}+-{r['cv_r2_std']:.4f}      "
              f"{r['test_r2']:<10.4f} {r['test_mae']:<12.4f} {r['test_rmse']:<12.4f} {r['train_time_s']:.2f}")

    best_name = max(results, key=lambda n: results[n]["test_r2"])
    print(f"Best model by held-out test R^2: {best_name}")

    return fitted, results, best_name, X_test, y_test, {n: fitted[n].predict(X_test) for n in fitted}


# --------------------------------------------------------------------------
# 3. Plots: parity plot (best model) and model-comparison bar chart
# --------------------------------------------------------------------------

def plot_parity(y_test, y_pred, model_name, r2, mae, target_name, filename):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.35, s=12, color="tab:blue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel(f"Physics-based {target_name} — ground truth")
    ax.set_ylabel(f"{model_name} predicted {target_name}")
    ax.set_title(f"{model_name} Surrogate vs. Physics-Based Model\n"
                 f"R² = {r2:.4f}, MAE = {mae:.3f} (test set, n={len(y_test)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_model_comparison(results, target_name, filename):
    names = list(results.keys())
    r2s = [results[n]["test_r2"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["tab:green" if v == max(r2s) else "tab:gray" for v in r2s]
    ax.bar(names, r2s, color=colors)
    for i, v in enumerate(r2s):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("Held-out test R²")
    ax.set_ylim(min(0.9, min(r2s) - 0.02), 1.012)
    ax.set_title(f"Surrogate Model Comparison — {target_name}", pad=14)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_feature_importance(model, filename):
    """Only meaningful for the tree-based models (Gini/impurity-based);
    skipped if the best model is the MLP, which has no direct equivalent
    without a separate permutation-importance pass."""
    if not hasattr(model, "feature_importances_"):
        print(f"Skipping feature importance plot for {filename}: "
              f"best model has no native feature_importances_ attribute.")
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(importances)), importances[order], color="tab:green")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in order], rotation=20, ha="right")
    ax.set_ylabel("Feature importance")
    ax.set_title("Which Inputs Drive LCOW Most?\n(best tree-based surrogate, Gini importance)")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


# --------------------------------------------------------------------------
# 4. Optimization example: use the best validated surrogate to search a
#    design space far larger than the hand-picked sweeps in Sections 4-5,
#    subject to a real-world constraint (brine-disposal land area), then
#    verify the top candidate against the physics-based model directly.
# --------------------------------------------------------------------------

def optimize_with_surrogate(model):
    from desalination_plant_design import BrineManagementSystem

    pond_budget_ha = 50.0  # illustrative land constraint for a constrained site
    n = 40
    tds_grid = np.linspace(1000, 15000, n)
    recovery_grid = np.linspace(0.50, 0.90, n)
    capacity_grid = np.linspace(2000, 60000, n)
    TDS, REC, CAP = np.meshgrid(tds_grid, recovery_grid, capacity_grid, indexing="ij")
    X_grid = np.column_stack([TDS.ravel(), REC.ravel(), CAP.ravel(), np.zeros(TDS.size)])

    lcow_pred = model.predict(X_grid)  # vectorized: one call for all 64,000 points
    order = np.argsort(lcow_pred)  # cheapest-predicted first

    best = None
    checked = 0
    for idx in order:
        tds, recovery, capacity = X_grid[idx, 0], X_grid[idx, 1], X_grid[idx, 2]
        feed = FeedWaterQuality(tds_mg_l=tds, water_type="brackish")
        cap = PlantCapacity(permeate_flow_m3d=capacity, recovery=recovery)
        flows = PlantFlows.from_capacity(cap)
        ro = ReverseOsmosisSystem(
            flow_m3d=flows.feed_to_membranes_m3d, permeate_m3d=flows.permeate_m3d,
            concentrate_m3d=flows.concentrate_m3d, feed_tds_mg_l=tds, water_type="brackish",
        ).design()
        pond = BrineManagementSystem(
            concentrate_flow_m3d=flows.concentrate_m3d,
            concentrate_tds_mg_l=ro["concentrate_tds_mg_l_est"],
            feed_tds_mg_l=tds, is_coastal_site=False, is_arid_climate=True,
        ).design_evaporation_ponds_alternative().design["evaporation_pond_option"]["required_pond_area_hectares"]
        checked += 1
        if pond <= pond_budget_ha:
            best = {"tds": tds, "recovery": recovery, "capacity": capacity,
                    "lcow_pred": lcow_pred[idx], "pond_ha": pond, "candidates_checked": checked}
            break
    return best, pond_budget_ha, X_grid.shape[0]


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # --- LCOW target: compare Random Forest, Gradient Boosting, and MLP ---
    fitted_lcow, results_lcow, best_lcow_name, X_test, y_test_lcow, preds_lcow = \
        train_and_compare(target_idx=0, target_name=TARGET_NAMES[0])
    plot_model_comparison(results_lcow, TARGET_NAMES[0], "curve10b_model_comparison_lcow.png")
    plot_parity(y_test_lcow, preds_lcow[best_lcow_name], best_lcow_name,
                results_lcow[best_lcow_name]["test_r2"], results_lcow[best_lcow_name]["test_mae"],
                TARGET_NAMES[0], "curve10_ml_surrogate_parity.png")
    plot_feature_importance(fitted_lcow[best_lcow_name], "curve11_ml_feature_importance.png")

    # --- SEC target: same three-way comparison ---
    fitted_sec, results_sec, best_sec_name, X_test2, y_test_sec, preds_sec = \
        train_and_compare(target_idx=1, target_name=TARGET_NAMES[1])
    plot_model_comparison(results_sec, TARGET_NAMES[1], "curve12_model_comparison_sec.png")
    plot_parity(y_test_sec, preds_sec[best_sec_name], best_sec_name,
                results_sec[best_sec_name]["test_r2"], results_sec[best_sec_name]["test_mae"],
                TARGET_NAMES[1], "curve13_sec_surrogate_parity.png")

    # --- Surrogate-accelerated, physics-verified optimization (LCOW) ---
    print("\nRunning constrained optimization search (best LCOW surrogate, physics-verified)...")
    best, pond_budget_ha, n_grid = optimize_with_surrogate(fitted_lcow[best_lcow_name])
    print(f"Grid size searched: {n_grid:,} combinations (surrogate-predicted in one vectorized call)")
    print(f"Pond-area budget constraint: <= {pond_budget_ha} ha")
    if best:
        print(f"Candidates checked against physics model before first feasible hit: {best['candidates_checked']}")
        print("Best feasible design found:")
        print(f"  Feed TDS:        {best['tds']:.0f} mg/L")
        print(f"  Recovery:        {best['recovery']*100:.1f}%")
        print(f"  Capacity:        {best['capacity']:.0f} m3/day")
        print(f"  Predicted LCOW:  ${best['lcow_pred']:.3f}/m3")
        print(f"  Pond area:       {best['pond_ha']:.1f} ha")
    else:
        print("No feasible design found within the pond-area budget.")

    print("\nML surrogate model training, comparison, evaluation, and optimization example complete.")
