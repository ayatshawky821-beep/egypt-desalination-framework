"""
ai_deep_surrogate_large_scale.py

A deeper neural-network surrogate purpose-built for the LARGE-SCALE and
REGIONAL/NATIONAL-SCALE capacity tiers (100,000-600,000 m3/day) used by the
companion multi-objective (ai_multi_objective_design.py) and multi-plant
(ai_multi_plant_rl.py) modules in this paper.

WHY A SEPARATE SURROGATE FROM egypt_ml_surrogate.py: that module's training
data was sampled with capacity log-uniform over 2,000-150,000 m3/day (the
range relevant to the companion single-plant paper). A surrogate trained on
that range should not be trusted to extrapolate up to 600,000 m3/day --
tree ensembles in particular cannot predict outside the range of values
seen in training by construction. This module instead samples training
data DIRECTLY and ONLY within 100,000-600,000 m3/day, so the resulting
surrogate is valid exactly where this paper's large-scale and regional-
network studies need it, and is explicit about not claiming validity
outside that range.

Architecture: a deeper MLP (3 hidden layers: 128-64-32) than the 64-32
network in egypt_ml_surrogate.py, reflecting the larger, more capacity-
dominated regime this surrogate targets; compared against Gradient
Boosting as in the companion module, so the "deep learning" claim is
validated against a strong non-deep baseline rather than asserted alone.

Run:
    python3 ai_deep_surrogate_large_scale.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from desalination_plant_design import (
    FeedWaterQuality, PlantCapacity, PlantFlows, ReverseOsmosisSystem, EconomicAnalysis,
)

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10.5})

RNG = np.random.default_rng(7)
N_SAMPLES = 4000
CAPACITY_RANGE_M3D = (100_000.0, 600_000.0)  # spans both the "large municipal" and
                                               # "regional/national-scale" tiers of this paper


def sample_large_scale_point():
    is_seawater = RNG.random() < 0.3
    if is_seawater:
        tds = RNG.uniform(28000, 45000)
        recovery = RNG.uniform(0.35, 0.55)
        water_type = "seawater"
    else:
        tds = RNG.uniform(500, 15000)
        recovery = RNG.uniform(0.50, 0.90)
        water_type = "brackish"
    capacity = RNG.uniform(*CAPACITY_RANGE_M3D)  # linear-uniform: this range spans only 6x,
                                                   # not the 75x range of the original surrogate,
                                                   # so log-uniform sampling is unnecessary here

    feed = FeedWaterQuality(tds_mg_l=tds, water_type=water_type)
    cap = PlantCapacity(permeate_flow_m3d=capacity, recovery=recovery)
    flows = PlantFlows.from_capacity(cap)
    ro = ReverseOsmosisSystem(
        flow_m3d=flows.feed_to_membranes_m3d, permeate_m3d=flows.permeate_m3d,
        concentrate_m3d=flows.concentrate_m3d, feed_tds_mg_l=tds, water_type=water_type,
    ).design()
    sec_lo, sec_hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
    sec_mid = (sec_lo + sec_hi) / 2
    econ = EconomicAnalysis(capacity_m3d=capacity, water_type=water_type, specific_energy_kwh_m3=sec_mid).run_full_analysis()
    lcow = econ["lcow"]["lcow_usd_m3"]

    return [tds, recovery, capacity, 1.0 if is_seawater else 0.0], lcow


def build_dataset(n=N_SAMPLES):
    X, y = [], []
    for _ in range(n):
        feats, target = sample_large_scale_point()
        X.append(feats)
        y.append(target)
    return np.array(X), np.array(y)


FEATURE_NAMES = ["Feed TDS (mg/L)", "Recovery (fraction)", "Capacity (m3/d)", "Is seawater (0/1)"]


def train_and_compare():
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    models = {
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=7),
        "Deep MLP (128-64-32)": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
                         alpha=1e-4, max_iter=3000, random_state=7, early_stopping=True, n_iter_no_change=25),
        ),
    }

    results, fitted, preds = {}, {}, {}
    for name, model in models.items():
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "cv_r2_mean": cv.mean(), "cv_r2_std": cv.std(),
            "test_r2": r2_score(y_test, y_pred),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        }
        fitted[name] = model
        preds[name] = y_pred

    print(f"\n=== Large-scale surrogate comparison (capacity {CAPACITY_RANGE_M3D[0]:,.0f}-{CAPACITY_RANGE_M3D[1]:,.0f} m3/day) ===")
    print(f"{'Model':<24}{'CV R2':<20}{'Test R2':<10}{'Test MAE':<12}{'Test RMSE'}")
    for name, r in results.items():
        print(f"{name:<24}{r['cv_r2_mean']:.4f}+-{r['cv_r2_std']:.4f}    {r['test_r2']:<10.4f}{r['test_mae']:<12.4f}{r['test_rmse']:.4f}")
    best_name = max(results, key=lambda n: results[n]["test_r2"])
    print(f"Best model: {best_name}")
    return fitted, results, best_name, X_test, y_test, preds


def plot_parity(y_test, y_pred, model_name, r2, mae, filename):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.35, s=14, color="tab:purple")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Physics-based LCOW (USD/m3) — ground truth")
    ax.set_ylabel(f"{model_name} predicted LCOW (USD/m3)")
    ax.set_title(f"Large-Scale Surrogate ({model_name}) vs. Physics-Based Model\n"
                 f"R2={r2:.4f}, MAE=${mae:.4f}/m3 (100k-600k m3/day range, n={len(y_test)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_extrapolation_check(fitted_small_range_model, large_scale_model, model_name, filename):
    """Sanity check: does the ORIGINAL (small-range) surrogate silently
    produce plausible-looking but wrong predictions above its training
    range, compared to the new large-scale-trained model and the real
    physics model? This directly tests the extrapolation warning already
    given in Section 7.2 of the companion paper, rather than just asserting
    it."""
    capacities = np.linspace(100_000, 600_000, 60)
    feed = FeedWaterQuality(tds_mg_l=5000, water_type="brackish")
    recovery = 0.80

    physics_lcow = []
    for cap_val in capacities:
        cap = PlantCapacity(permeate_flow_m3d=cap_val, recovery=recovery)
        flows = PlantFlows.from_capacity(cap)
        ro = ReverseOsmosisSystem(flow_m3d=flows.feed_to_membranes_m3d, permeate_m3d=flows.permeate_m3d,
                                   concentrate_m3d=flows.concentrate_m3d, feed_tds_mg_l=5000, water_type="brackish").design()
        sec_lo, sec_hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
        econ = EconomicAnalysis(capacity_m3d=cap_val, water_type="brackish", specific_energy_kwh_m3=(sec_lo+sec_hi)/2).run_full_analysis()
        physics_lcow.append(econ["lcow"]["lcow_usd_m3"])

    X_check = np.column_stack([np.full(60, 5000.0), np.full(60, recovery), capacities, np.zeros(60)])
    small_range_pred = fitted_small_range_model.predict(X_check)
    large_scale_pred = large_scale_model.predict(X_check)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(capacities, physics_lcow, "k-", linewidth=2.5, label="Physics-based model (ground truth)")
    ax.plot(capacities, small_range_pred, "r--", linewidth=1.8,
             label="Original surrogate (trained only on 2,000-150,000 m3/day)")
    ax.plot(capacities, large_scale_pred, "-", color="tab:purple", linewidth=1.8,
             label=f"New large-scale surrogate ({model_name})")
    ax.axvline(150_000, color="gray", linestyle=":", linewidth=1)
    ax.text(152_000, ax.get_ylim()[0] + 0.02, "original surrogate's\ntraining range ends here", fontsize=8, color="gray")
    ax.set_xlabel("Plant capacity (m3/day)")
    ax.set_ylabel("LCOW (USD/m3)")
    ax.set_title("Extrapolation Check: Does the Original Surrogate Mislead Above 150,000 m3/day?")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


if __name__ == "__main__":
    fitted, results, best_name, X_test, y_test, preds = train_and_compare()
    plot_parity(y_test, preds[best_name], best_name, results[best_name]["test_r2"], results[best_name]["test_mae"],
                "ai_large_scale_surrogate_parity.png")

    # --- Extrapolation sanity check against the ORIGINAL small-range surrogate ---
    print("\nTraining the original (small-range, 2,000-150,000 m3/day) surrogate for comparison...")
    try:
        from egypt_ml_surrogate import build_dataset as build_small_dataset
        X_small, Y_small = build_small_dataset()
        from sklearn.ensemble import GradientBoostingRegressor as GBR
        small_model = GBR(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
        small_model.fit(X_small, Y_small[:, 0])
        plot_extrapolation_check(small_model, fitted[best_name], best_name, "ai_extrapolation_check.png")
    except Exception as e:
        print(f"Extrapolation check skipped ({e}); large-scale surrogate training above is unaffected.")

    print("\nLarge-scale deep surrogate training and extrapolation check complete.")
