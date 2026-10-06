"""
ai_design_optimizer.py

A genuinely adaptive AI-based design optimizer: Bayesian Optimization (a
Gaussian Process surrogate plus an Expected Improvement acquisition
function) searching DIRECTLY against the real physics-based framework in
desalination_plant_design.py -- not against a pre-trained ML surrogate.

HOW THIS DIFFERS FROM egypt_ml_surrogate.py's optimize_with_surrogate():
That function trains a surrogate once on thousands of random samples, then
ranks a fixed grid by the surrogate's prediction. It is fast but its
accuracy is capped by how well the one-shot surrogate generalizes, and it
only ever evaluates points that happen to fall on the pre-defined grid.

This module instead treats each evaluation of the physics-based framework
as expensive (a reasonable stand-in for coupling this optimizer to a
genuinely expensive subroutine, such as a membrane-vendor projection run or
a hydraulic network solve). It builds its OWN internal Gaussian Process
surrogate that is updated after every single evaluation, and uses that
surrogate's predicted mean AND uncertainty (not just the mean) to decide
which untested design point is most worth evaluating next -- balancing
exploitation (points near the current best) against exploration (points
the model is still uncertain about). This is the standard "Bayesian
Optimization" pattern used for expensive black-box engineering
optimization (e.g. process design, hyperparameter tuning, experimental
design).

OPTIMIZATION PROBLEM SOLVED HERE (two free design variables):
Given a fixed feed-water preset (TDS, water type) and a target water-demand
range (a minimum required capacity up to a site/budget ceiling, e.g. a
planner open to modestly oversizing the plant to capture economies of
scale), find the (system recovery, plant capacity) pair that MINIMIZES the
Levelized Cost of Water (LCOW), subject to:
  - recovery staying within the physically sane bounds used elsewhere in
    this framework (0.50-0.90 brackish, 0.35-0.55 seawater),
  - capacity staying within the planner's acceptable demand range, and
  - the resulting evaporation-pond area (for brine disposal) staying under
    a configurable site land-area budget -- which depends on BOTH
    variables jointly (concentrate flow scales with capacity; concentrate
    TDS and hence pond intensity scales with recovery), making this a
    genuinely two-dimensional, non-separable search rather than two
    independent 1-D problems.
Infeasible points (pond area over budget) return a smooth penalty rather
than crashing the optimizer, so the Gaussian Process learns the shape of
the infeasible region rather than failing on it.

A fair, matched-budget comparison against brute-force grid search is
included: at the same number of physics-model evaluations, grid search in
2-D only achieves sqrt(N) resolution per axis, while Bayesian Optimization
concentrates its budget adaptively -- the gap between the two should widen
noticeably compared to the 1-D case.

Run:
    python3 ai_design_optimizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm

from desalination_plant_design import (
    FeedWaterQuality,
    PlantCapacity,
    PlantFlows,
    ReverseOsmosisSystem,
    EconomicAnalysis,
    BrineManagementSystem,
    EGYPT_BRACKISH_PRESETS,
)

plt.rcParams.update({
    "figure.figsize": (7.5, 5),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10.5,
})


# --------------------------------------------------------------------------
# 1. The expensive black-box objective: physics-based LCOW at a given
#    (recovery, capacity) pair, with a penalty for violating the pond-area
#    land constraint.
# --------------------------------------------------------------------------

class DesignObjective:
    """Wraps the physics-based framework as a 2-D black-box function of
    (recovery, capacity) for a fixed feed preset and pond-area budget.
    Internally normalizes both variables to [0, 1] for the GP, since
    recovery (~0.5-0.9) and capacity (~thousands of m3/day) live on very
    different scales."""

    def __init__(self, feed: FeedWaterQuality, capacity_range_m3d, pond_budget_ha=50.0,
                 is_coastal_site: bool = False):
        self.feed = feed
        self.cap_lo, self.cap_hi = capacity_range_m3d
        self.pond_budget_ha = pond_budget_ha
        self.is_coastal_site = is_coastal_site
        self.history = []  # list of dicts: recovery, capacity, lcow, pond_ha, feasible, score

    def bounds_unit(self):
        """Both variables normalized to [0, 1] for the optimizer."""
        return np.array([[0.0, 1.0], [0.0, 1.0]])

    def recovery_bounds(self):
        if self.feed.water_type == "seawater":
            return (0.35, 0.55)
        return (0.50, 0.90)

    def _denormalize(self, x_unit):
        """x_unit = [u_recovery, u_capacity], each in [0,1]."""
        r_lo, r_hi = self.recovery_bounds()
        recovery = r_lo + x_unit[0] * (r_hi - r_lo)
        capacity = self.cap_lo + x_unit[1] * (self.cap_hi - self.cap_lo)
        return recovery, capacity

    def evaluate_unit(self, x_unit) -> float:
        recovery, capacity = self._denormalize(x_unit)
        return self.evaluate(recovery, capacity)

    def evaluate(self, recovery: float, capacity_m3d: float) -> float:
        cap = PlantCapacity(permeate_flow_m3d=capacity_m3d, recovery=recovery)
        flows = PlantFlows.from_capacity(cap)
        ro = ReverseOsmosisSystem(
            flow_m3d=flows.feed_to_membranes_m3d, permeate_m3d=flows.permeate_m3d,
            concentrate_m3d=flows.concentrate_m3d, feed_tds_mg_l=self.feed.tds_mg_l,
            water_type=self.feed.water_type,
        ).design()
        sec_lo, sec_hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
        sec_mid = (sec_lo + sec_hi) / 2

        econ = EconomicAnalysis(
            capacity_m3d=capacity_m3d, water_type=self.feed.water_type,
            specific_energy_kwh_m3=sec_mid,
        ).run_full_analysis()
        lcow = econ["lcow"]["lcow_usd_m3"]

        pond_ha = BrineManagementSystem(
            concentrate_flow_m3d=flows.concentrate_m3d,
            concentrate_tds_mg_l=ro["concentrate_tds_mg_l_est"],
            feed_tds_mg_l=self.feed.tds_mg_l, is_coastal_site=self.is_coastal_site,
            is_arid_climate=True,
        ).design_evaporation_ponds_alternative().design["evaporation_pond_option"]["required_pond_area_hectares"]

        feasible = pond_ha <= self.pond_budget_ha
        score = lcow if feasible else lcow + 0.08 * max(0.0, pond_ha - self.pond_budget_ha)

        self.history.append({"recovery": recovery, "capacity": capacity_m3d, "lcow": lcow,
                              "pond_ha": pond_ha, "feasible": feasible, "score": score})
        return score


# --------------------------------------------------------------------------
# 2. Bayesian Optimization loop (N-dimensional): GP surrogate + Expected
#    Improvement, with a multi-start continuous optimizer for the
#    acquisition function (more faithful to real BO than a fixed grid).
# --------------------------------------------------------------------------

def expected_improvement(X_candidates, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    improvement = y_best - mu - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-9] = 0.0
    return ei


def propose_next_point(gp, y_best, bounds_unit, X_done, rng, n_candidates=3000):
    """Dense random candidate search over the unit square for the point
    maximizing Expected Improvement, skipping anything too close to an
    already-evaluated point (avoids wasting evaluations on near-duplicates)."""
    candidates = rng.uniform(bounds_unit[:, 0], bounds_unit[:, 1], size=(n_candidates, 2))
    ei = expected_improvement(candidates, gp, y_best)
    if len(X_done):
        dists = np.linalg.norm(candidates[:, None, :] - X_done[None, :, :], axis=2).min(axis=1)
        ei[dists < 0.02] = -np.inf
    return candidates[np.argmax(ei)]


def bayesian_optimize_2d(objective: DesignObjective, n_init=8, n_iter=22, seed=42):
    rng = np.random.default_rng(seed)
    bounds_unit = objective.bounds_unit()

    X = rng.uniform(0, 1, size=(n_init, 2))
    y = np.array([objective.evaluate_unit(x) for x in X])

    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=[0.2, 0.2], length_scale_bounds=(1e-2, 2.0), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6,
                                   n_restarts_optimizer=4, random_state=seed)

    for i in range(n_iter):
        gp.fit(X, y)
        y_best = y.min()
        next_x = propose_next_point(gp, y_best, bounds_unit, X, rng)
        next_y = objective.evaluate_unit(next_x)
        X = np.vstack([X, next_x])
        y = np.append(y, next_y)

    best_idx = np.argmin(y)
    recovery, capacity = objective._denormalize(X[best_idx])
    best = {"recovery": recovery, "capacity": capacity, "score": float(y[best_idx])}
    return X, y, best, gp


# --------------------------------------------------------------------------
# 3. Plots
# --------------------------------------------------------------------------

def plot_convergence(history, preset_name, filename):
    scores = [h["score"] for h in history]
    running_best = np.minimum.accumulate(scores)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(scores) + 1), scores, "o", alpha=0.4, color="tab:blue", label="Each evaluation")
    ax.plot(range(1, len(scores) + 1), running_best, "-", color="tab:red", linewidth=2, label="Best found so far")
    ax.set_xlabel("Physics-model evaluation number")
    ax.set_ylabel("Objective score (LCOW, USD/m3; penalized if infeasible)")
    ax.set_title(f"Bayesian Optimization Convergence — {preset_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_search_map(objective, X, y, preset_name, filename):
    """2-D map of where Bayesian Optimization chose to sample, colored by
    objective score, over the (recovery, capacity) design space -- shows
    the adaptive concentration of samples near the optimum."""
    r_lo, r_hi = objective.recovery_bounds()
    recoveries = r_lo + X[:, 0] * (r_hi - r_lo)
    capacities = objective.cap_lo + X[:, 1] * (objective.cap_hi - objective.cap_lo)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(recoveries * 100, capacities, c=y, cmap="viridis_r", s=60, edgecolor="black", linewidth=0.5)
    best_idx = np.argmin(y)
    ax.scatter(recoveries[best_idx] * 100, capacities[best_idx], marker="*", s=400,
               color="red", edgecolor="black", linewidth=1, zorder=5, label="Best found")
    fig.colorbar(sc, ax=ax, label="Objective score (LCOW, USD/m3)")
    ax.set_xlabel("System recovery (%)")
    ax.set_ylabel("Plant capacity (m3/day)")
    ax.set_title(f"Where Bayesian Optimization Sampled — {preset_name}\n"
                 f"(point color = LCOW; note concentration near the optimum)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


# --------------------------------------------------------------------------
# 4. Matched-budget grid-search baseline for a fair efficiency comparison
# --------------------------------------------------------------------------

def grid_search_baseline_2d(objective: DesignObjective, n_total_evals):
    """Evenly-spaced sqrt(N) x sqrt(N) grid using (approximately) the same
    total evaluation budget as the Bayesian search."""
    n_per_axis = max(2, int(round(n_total_evals ** 0.5)))
    r_lo, r_hi = objective.recovery_bounds()
    recoveries = np.linspace(r_lo, r_hi, n_per_axis)
    capacities = np.linspace(objective.cap_lo, objective.cap_hi, n_per_axis)
    best_score = np.inf
    for r in recoveries:
        for c in capacities:
            s = objective.evaluate(float(r), float(c))
            best_score = min(best_score, s)
    return best_score, n_per_axis * n_per_axis


# --------------------------------------------------------------------------
# 5. Run for several presets and summarize
# --------------------------------------------------------------------------

def run_for_preset(preset_key, nominal_capacity_m3d=20000.0, capacity_flex=0.30,
                    pond_budget_ha=50.0, is_coastal_site=False, n_init=8, n_iter=22):
    feed = EGYPT_BRACKISH_PRESETS[preset_key]
    cap_range = (nominal_capacity_m3d * (1 - capacity_flex), nominal_capacity_m3d * (1 + capacity_flex))
    print(f"\n{'='*72}\nOptimizing (recovery, capacity) for preset: {preset_key}")
    print(f"TDS={feed.tds_mg_l:.0f} mg/L | capacity range={cap_range[0]:.0f}-{cap_range[1]:.0f} m3/day "
          f"(+-{capacity_flex*100:.0f}% around a {nominal_capacity_m3d:.0f} m3/day nominal demand)")

    objective = DesignObjective(feed, cap_range, pond_budget_ha, is_coastal_site)
    X, y, best, gp = bayesian_optimize_2d(objective, n_init=n_init, n_iter=n_iter)
    n_bo_evals = n_init + n_iter

    print(f"Bayesian Optimization: {n_bo_evals} physics-model evaluations")
    print(f"  Best recovery:   {best['recovery']*100:.2f}%")
    print(f"  Best capacity:   {best['capacity']:.0f} m3/day")
    print(f"  Best LCOW/score: ${best['score']:.4f}/m3")

    objective_grid = DesignObjective(feed, cap_range, pond_budget_ha, is_coastal_site)
    grid_best, n_grid_evals = grid_search_baseline_2d(objective_grid, n_bo_evals)
    gap_pct = (grid_best - best["score"]) / best["score"] * 100
    print(f"Grid search baseline ({n_grid_evals} evaluations, {int(n_grid_evals**0.5)}x{int(n_grid_evals**0.5)}): ${grid_best:.4f}/m3")
    print(f"Bayesian Optimization advantage: {gap_pct:+.2f}% "
          f"({'better' if gap_pct > 0 else 'worse or equal'} than grid search at a comparable budget)")

    preset_label = preset_key.replace("_", " ").title()
    plot_convergence(objective.history, preset_label, f"ai_opt_convergence_{preset_key}.png")
    plot_search_map(objective, X, y, preset_label, f"ai_opt_search_map_{preset_key}.png")

    return best, n_bo_evals, grid_best, n_grid_evals


if __name__ == "__main__":
    summary = []
    for preset_key in ["el_moghra_aquifer", "nile_delta_shallow", "sinai_coastal"]:
        best, n_evals, grid_best, n_grid_evals = run_for_preset(preset_key)
        summary.append((preset_key, best, n_evals, grid_best, n_grid_evals))

    print(f"\n{'='*72}\nSUMMARY — 2-D Bayesian Optimization (recovery x capacity) across presets")
    print(f"{'Preset':<22} {'Recovery':<10} {'Capacity':<12} {'BO LCOW':<12} {'Grid LCOW':<12} {'BO advantage'}")
    for preset_key, best, n_evals, grid_best, n_grid_evals in summary:
        adv = (grid_best - best["score"]) / best["score"] * 100
        print(f"{preset_key:<22} {best['recovery']*100:<10.2f} {best['capacity']:<12.0f} "
              f"${best['score']:<11.4f} ${grid_best:<11.4f} {adv:+.2f}%")

    print("\nAI-based Bayesian design optimization (2-D) complete.")
