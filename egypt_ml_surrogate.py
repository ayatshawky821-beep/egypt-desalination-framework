"""
egypt_ml_surrogate.py
 
Machine-learning surrogate model layered on top of the physics-based
desalination_plant_design.py framework.
 
WHY: The physics-based model (Sections 2 and Appendix A of the paper) is
cheap to evaluate once, but sweeping it across many design variables at once
(TDS x recovery x capacity x water type, for optimization or Monte Carlo
uncertainty analysis) means re-running the full pretreatment -> filtration
-> economics pipeline thousands of times. This script instead:
  1. Uses the physics-based framework itself as a "ground truth" generator,
     sampling a wide range of feed TDS, system recovery, plant capacity, and
     water type.
  2. Trains a Random Forest regression model (a standard supervised ML
     method for tabular, nonlinear, non-smooth functions -- appropriate here
     because the underlying rules include piecewise recovery thresholds and
     a clipped economies-of-scale factor) to predict Levelized Cost of Water
     (LCOW) directly from those four inputs.
  3. Evaluates the surrogate's accuracy (R^2, MAE, RMSE) on a held-out test
     set, and reports which inputs matter most (feature importance).
 
This demonstrates how an AI/ML tool can accelerate design-space exploration
built on top of an engineering-grounded model, without inventing new
physics: the surrogate is only ever as good as the physics-based training
data it learns from.
 
Run:
    python3 egypt_ml_surrogate.py
"""
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
N_SAMPLES = 2000
 
 
# --------------------------------------------------------------------------
# 1. Generate training data from the physics-based framework
# --------------------------------------------------------------------------
 
def sample_design_point():
    """Draw one random, physically plausible design point and evaluate the
    physics-based framework on it, returning (features, target)."""
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
    return features, lcow
 
 
def build_dataset(n=N_SAMPLES):
    X, y = [], []
    for _ in range(n):
        feats, target = sample_design_point()
        X.append(feats)
        y.append(target)
    return np.array(X), np.array(y)
 
 
FEATURE_NAMES = ["Feed TDS (mg/L)", "Recovery (fraction)", "Capacity (m3/d)", "Is seawater (0/1)"]
 
 
# --------------------------------------------------------------------------
# 2. Train and evaluate the Random Forest surrogate
# --------------------------------------------------------------------------
 
def train_and_evaluate():
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    model = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
 
    print(f"Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    print(f"R^2  = {r2:.4f}")
    print(f"MAE  = {mae:.4f} USD/m3")
    print(f"RMSE = {rmse:.4f} USD/m3")
 
    return model, X_test, y_test, y_pred, r2, mae, rmse
 
 
# --------------------------------------------------------------------------
# 3. Plots: parity plot and feature importance
# --------------------------------------------------------------------------
 
def plot_parity(y_test, y_pred, r2, mae):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.4, s=14, color="tab:blue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Physics-based LCOW (USD/m³) — ground truth")
    ax.set_ylabel("ML surrogate predicted LCOW (USD/m³)")
    ax.set_title(f"Random Forest Surrogate vs. Physics-Based Model\n"
                 f"R² = {r2:.4f}, MAE = ${mae:.3f}/m³ (test set, n={len(y_test)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig("curve10_ml_surrogate_parity.png")
    plt.close(fig)
    print("Saved curve10_ml_surrogate_parity.png")
 
 
def plot_feature_importance(model):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(importances)), importances[order], color="tab:green")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in order], rotation=20, ha="right")
    ax.set_ylabel("Random Forest feature importance")
    ax.set_title("Which Inputs Drive LCOW Most?\n(Random Forest surrogate, Gini importance)")
    fig.tight_layout()
    fig.savefig("curve11_ml_feature_importance.png")
    plt.close(fig)
    print("Saved curve11_ml_feature_importance.png")
 
 
# --------------------------------------------------------------------------
# 4. Optimization example: use the trained surrogate to search a design
#    space far larger than the hand-picked sweeps in Sections 4-5 of the
#    paper, subject to a real-world constraint (brine-disposal land area).
# --------------------------------------------------------------------------
 
def optimize_with_surrogate(model):
    """Search a design space far larger than the hand-picked sweeps in
    Sections 4-5 using the trained surrogate for the expensive part (LCOW
    prediction across ~50,000 combinations, vectorized and near-instant),
    then verify the evaporation-pond land constraint with the physics-based
    model only for the most promising candidates. This two-tier pattern
    (cheap surrogate for global search, physics model for final
    verification) is the realistic way a surrogate adds optimization value
    beyond curve-fitting a known function."""
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
    model, X_test, y_test, y_pred, r2, mae, rmse = train_and_evaluate()
    plot_parity(y_test, y_pred, r2, mae)
    plot_feature_importance(model)
 
    print("\nRunning constrained optimization search (surrogate-accelerated)...")
    best, pond_budget_ha, n_grid = optimize_with_surrogate(model)
    print(f"Grid size searched: {n_grid:,} combinations (surrogate-predicted in one vectorized call)")
    print(f"Pond-area budget constraint: <= {pond_budget_ha} ha")
    if best:
        print(f"Candidates checked against physics model before first feasible hit: {best['candidates_checked']}")
        print(f"Best feasible design found:")
        print(f"  Feed TDS:        {best['tds']:.0f} mg/L")
        print(f"  Recovery:        {best['recovery']*100:.1f}%")
        print(f"  Capacity:        {best['capacity']:.0f} m3/day")
        print(f"  Predicted LCOW:  ${best['lcow_pred']:.3f}/m3")
        print(f"  Pond area:       {best['pond_ha']:.1f} ha")
    else:
        print("No feasible design found within the pond-area budget.")
 
    print("\nML surrogate model training, evaluation, and optimization example complete.")
 