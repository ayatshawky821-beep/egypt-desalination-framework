"""
ai_multi_objective_design.py

Multi-objective AI design optimization for a LARGE-SCALE brackish water
RO plant (100,000-300,000 m3/day), comparing two AI paradigms on the same
problem at a matched evaluation budget:

  (A) NSGA-II -- a genetic algorithm (evolutionary AI), implemented from
      scratch (fast non-dominated sorting + crowding distance + simulated
      binary crossover + polynomial mutation), the standard method for
      finding a Pareto front without scalarizing multiple objectives into
      one number.

  (B) Multi-objective Bayesian Optimization (MOBO) via weighted-Chebyshev
      scalarization -- for each of many random weight vectors, a separate
      single-objective Bayesian search (Gaussian Process + Expected
      Improvement, as in ai_design_optimizer.py) is run against the
      Chebyshev-scalarized objective, and the union of all runs' best
      points approximates the Pareto front. This re-uses the adaptive,
      sample-efficient search from the companion single-plant paper, now
      extended to multiple objectives.

THREE OBJECTIVES (all minimized):
  1. LCOW (USD/m3)              -- economic cost
  2. SEC (kWh/m3)                -- energy intensity
  3. Evaporation-pond area (ha)  -- land footprint for brine disposal

TWO DECISION VARIABLES: system recovery and plant capacity, both within
large-scale bounds (100,000-300,000 m3/day capacity; preset-dependent
recovery bounds).

This reuses the validated physics-based framework (desalination_plant_
design.py) and the same Egyptian feed-water presets as the companion
single-plant paper -- the AI methodology is new, the engineering ground
truth is not reinvented.

Run:
    python3 ai_multi_objective_design.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm

from desalination_plant_design import (
    FeedWaterQuality, PlantCapacity, PlantFlows, ReverseOsmosisSystem,
    EconomicAnalysis, BrineManagementSystem, EGYPT_BRACKISH_PRESETS,
)

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10.5})

LARGE_SCALE_CAPACITY_RANGE = (100_000.0, 300_000.0)  # m3/day, "large municipal plant" tier


# ===========================================================================
# 1. Shared problem definition: 2 decision variables -> 3 objectives
# ===========================================================================

class LargeScaleDesignProblem:
    def __init__(self, feed: FeedWaterQuality, capacity_range=LARGE_SCALE_CAPACITY_RANGE,
                 is_coastal_site: bool = False):
        self.feed = feed
        self.cap_lo, self.cap_hi = capacity_range
        self.is_coastal_site = is_coastal_site
        self.n_evals = 0

    def recovery_bounds(self):
        return (0.35, 0.55) if self.feed.water_type == "seawater" else (0.50, 0.90)

    def bounds(self):
        r_lo, r_hi = self.recovery_bounds()
        return np.array([[r_lo, r_hi], [self.cap_lo, self.cap_hi]])

    def evaluate(self, recovery: float, capacity_m3d: float):
        """Returns (LCOW, SEC, pond_area_ha), all to be minimized."""
        self.n_evals += 1
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

        return lcow, sec_mid, pond_ha


# ===========================================================================
# 2. NSGA-II (implemented from scratch: no AI/optimization library available
#    in this environment, so the canonical algorithm -- Deb et al. 2002 --
#    is reproduced directly: fast non-dominated sorting, crowding distance,
#    binary tournament selection, simulated binary crossover (SBX), and
#    polynomial mutation)
# ===========================================================================

def dominates(a, b):
    """True if objective vector a Pareto-dominates b (minimization)."""
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(F):
    n = len(F)
    S = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance(F_front):
    n = len(F_front)
    if n == 0:
        return np.array([])
    dist = np.zeros(n)
    m = F_front.shape[1]
    for obj in range(m):
        order = np.argsort(F_front[:, obj])
        dist[order[0]] = dist[order[-1]] = np.inf
        obj_range = F_front[order[-1], obj] - F_front[order[0], obj]
        if obj_range == 0:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (F_front[order[k + 1], obj] - F_front[order[k - 1], obj]) / obj_range
    return dist


def tournament_select(pop_X, ranks, crowd, rng):
    i, j = rng.integers(0, len(pop_X), 2)
    if ranks[i] < ranks[j]:
        return pop_X[i]
    if ranks[j] < ranks[i]:
        return pop_X[j]
    return pop_X[i] if crowd[i] > crowd[j] else pop_X[j]


def sbx_crossover(p1, p2, bounds, rng, eta=15, prob=0.9):
    if rng.random() > prob:
        return p1.copy(), p2.copy()
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-12:
            continue
        x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        lo, hi = bounds[i]
        u = rng.random()
        beta = 1 + 2 * (x1 - lo) / (x2 - x1 + 1e-12)
        alpha = 2 - beta ** -(eta + 1)
        beta_q = (u * alpha) ** (1 / (eta + 1)) if u <= 1 / alpha else (1 / (2 - u * alpha)) ** (1 / (eta + 1))
        c1[i] = np.clip(0.5 * ((x1 + x2) - beta_q * (x2 - x1)), lo, hi)
        c2[i] = np.clip(0.5 * ((x1 + x2) + beta_q * (x2 - x1)), lo, hi)
    return c1, c2


def polynomial_mutation(x, bounds, rng, eta=20, prob=None):
    x = x.copy()
    prob = prob if prob is not None else 1.0 / len(x)
    for i in range(len(x)):
        if rng.random() > prob:
            continue
        lo, hi = bounds[i]
        delta1 = (x[i] - lo) / (hi - lo)
        delta2 = (hi - x[i]) / (hi - lo)
        u = rng.random()
        if u < 0.5:
            deltaq = (2 * u + (1 - 2 * u) * (1 - delta1) ** (eta + 1)) ** (1 / (eta + 1)) - 1
        else:
            deltaq = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta2) ** (eta + 1)) ** (1 / (eta + 1))
        x[i] = np.clip(x[i] + deltaq * (hi - lo), lo, hi)
    return x


def nsga2(problem: LargeScaleDesignProblem, pop_size=40, n_gen=25, seed=42):
    rng = np.random.default_rng(seed)
    bounds = problem.bounds()

    def eval_pop(X):
        return np.array([problem.evaluate(x[0], x[1]) for x in X])

    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, 2))
    F = eval_pop(X)
    history_best_lcow = [F[:, 0].min()]

    for gen in range(n_gen):
        fronts = fast_non_dominated_sort(F)
        ranks = np.zeros(len(X), dtype=int)
        crowd = np.zeros(len(X))
        for rank, front in enumerate(fronts):
            ranks[front] = rank
            crowd[front] = crowding_distance(F[front])

        # --- generate offspring ---
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_select(X, ranks, crowd, rng)
            p2 = tournament_select(X, ranks, crowd, rng)
            c1, c2 = sbx_crossover(p1, p2, bounds, rng)
            c1 = polynomial_mutation(c1, bounds, rng)
            c2 = polynomial_mutation(c2, bounds, rng)
            offspring.extend([c1, c2])
        offspring = np.array(offspring[:pop_size])
        F_off = eval_pop(offspring)

        # --- combine parent + offspring, select next generation ---
        X_combined = np.vstack([X, offspring])
        F_combined = np.vstack([F, F_off])
        fronts = fast_non_dominated_sort(F_combined)

        new_X, new_F = [], []
        for front in fronts:
            if len(new_X) + len(front) <= pop_size:
                new_X.extend(X_combined[front])
                new_F.extend(F_combined[front])
            else:
                remaining = pop_size - len(new_X)
                cd = crowding_distance(F_combined[front])
                order = np.argsort(cd)[::-1]
                chosen = [front[i] for i in order[:remaining]]
                new_X.extend(X_combined[chosen])
                new_F.extend(F_combined[chosen])
                break
        X, F = np.array(new_X), np.array(new_F)
        history_best_lcow.append(F[:, 0].min())

    final_fronts = fast_non_dominated_sort(F)
    pareto_X, pareto_F = X[final_fronts[0]], F[final_fronts[0]]
    return pareto_X, pareto_F, problem.n_evals, history_best_lcow


# ===========================================================================
# 3. Multi-objective Bayesian Optimization via weighted-Chebyshev
#    scalarization, matched to NSGA-II's total evaluation budget.
# ===========================================================================

def chebyshev_scalarize(f_vec, weights, ideal):
    return np.max(weights * np.abs(f_vec - ideal))


def mobo_chebyshev(problem: LargeScaleDesignProblem, total_budget, n_weight_vectors=10, seed=42):
    rng = np.random.default_rng(seed)
    bounds = problem.bounds()
    evals_per_weight = max(3, total_budget // n_weight_vectors)

    # Rough ideal point: evaluate a small warm-up sample to estimate per-objective minima
    warmup_X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(8, 2))
    warmup_F = np.array([problem.evaluate(x[0], x[1]) for x in warmup_X])
    ideal = warmup_F.min(axis=0)

    all_X, all_F = [warmup_X], [warmup_F]

    for w_idx in range(n_weight_vectors):
        w = rng.dirichlet(np.ones(3))  # random weights summing to 1, one per objective
        n_init = max(2, evals_per_weight // 3)
        n_iter = evals_per_weight - n_init

        X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, 2))
        F = np.array([problem.evaluate(x[0], x[1]) for x in X])
        y_scalar = np.array([chebyshev_scalarize(f, w, ideal) for f in F])

        X_unit = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        kernel = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=[0.2, 0.2], length_scale_bounds=(1e-2, 2.0), nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6, n_restarts_optimizer=2, random_state=seed + w_idx)

        for it in range(max(0, n_iter)):
            gp.fit(X_unit, y_scalar)
            y_best = y_scalar.min()
            candidates = rng.uniform(0, 1, size=(1500, 2))
            mu, sigma = gp.predict(candidates, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            improvement = y_best - mu - 0.01
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            if len(X_unit):
                dists = np.linalg.norm(candidates[:, None, :] - X_unit[None, :, :], axis=2).min(axis=1)
                ei[dists < 0.02] = -np.inf
            next_unit = candidates[np.argmax(ei)]
            next_x = bounds[:, 0] + next_unit * (bounds[:, 1] - bounds[:, 0])
            next_f = np.array(problem.evaluate(next_x[0], next_x[1]))
            X = np.vstack([X, next_x])
            F = np.vstack([F, next_f])
            y_scalar = np.append(y_scalar, chebyshev_scalarize(next_f, w, ideal))
            X_unit = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

        all_X.append(X)
        all_F.append(F)

    X_all = np.vstack(all_X)
    F_all = np.vstack(all_F)

    # Extract the non-dominated front from everything evaluated across all weight vectors
    fronts = fast_non_dominated_sort(F_all)
    pareto_X, pareto_F = X_all[fronts[0]], F_all[fronts[0]]
    return pareto_X, pareto_F, problem.n_evals


# ===========================================================================
# 4. Pareto front comparison metric: hypervolume (2-objective projection,
#    LCOW x SEC, for simplicity and interpretability) relative to a shared
#    reference point, at matched evaluation budgets.
# ===========================================================================

def hypervolume_2d(F_front, ref_point):
    """Exact 2-D hypervolume (LCOW, SEC) via the standard sort-and-sum method."""
    pts = F_front[:, :2]
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_y = ref_point[1]
    for x, y in pts:
        if y < prev_y:
            hv += (ref_point[0] - x) * (prev_y - y)
            prev_y = y
    return max(hv, 0.0)


# ===========================================================================
# 5. Plots
# ===========================================================================

def plot_pareto_comparison(F_nsga2, F_mobo, preset_name, filename):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(F_nsga2[:, 0], F_nsga2[:, 1], c="tab:blue", s=50, alpha=0.75,
               edgecolor="black", linewidth=0.4, label=f"NSGA-II Pareto front (n={len(F_nsga2)})")
    ax.scatter(F_mobo[:, 0], F_mobo[:, 1], c="tab:orange", s=50, alpha=0.75, marker="s",
               edgecolor="black", linewidth=0.4, label=f"MOBO Pareto front (n={len(F_mobo)})")
    ax.set_xlabel("LCOW (USD/m3)")
    ax.set_ylabel("SEC (kWh/m3)")
    ax.set_title(f"Pareto Fronts: NSGA-II vs. Multi-Objective Bayesian Optimization\n{preset_name}, large-scale capacity range")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_nsga2_convergence(history, preset_name, filename):
    fig, ax = plt.subplots()
    ax.plot(history, "-o", color="tab:green", markersize=3)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best LCOW in population (USD/m3)")
    ax.set_title(f"NSGA-II Convergence — {preset_name}")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_pareto_3d_projection(F_nsga2, preset_name, filename):
    """3-objective front shown as LCOW-SEC colored by pond area, since a
    true 3-D scatter is harder to read in print."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(F_nsga2[:, 0], F_nsga2[:, 1], c=F_nsga2[:, 2], cmap="plasma_r",
                     s=70, edgecolor="black", linewidth=0.4)
    fig.colorbar(sc, ax=ax, label="Evaporation pond area (ha)")
    ax.set_xlabel("LCOW (USD/m3)")
    ax.set_ylabel("SEC (kWh/m3)")
    ax.set_title(f"NSGA-II Pareto Front, 3 Objectives — {preset_name}\n(color = pond area; 2-D projection of a 3-objective front)")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


# ===========================================================================
if __name__ == "__main__":
    summary = []
    for preset_key in ["el_moghra_aquifer", "nile_delta_shallow"]:
        feed = EGYPT_BRACKISH_PRESETS[preset_key]
        print(f"\n{'='*72}\nLarge-scale multi-objective design: {preset_key} "
              f"(TDS={feed.tds_mg_l:.0f} mg/L, capacity {LARGE_SCALE_CAPACITY_RANGE[0]:,.0f}-{LARGE_SCALE_CAPACITY_RANGE[1]:,.0f} m3/day)")

        # --- NSGA-II ---
        problem_a = LargeScaleDesignProblem(feed)
        pareto_X_a, pareto_F_a, n_evals_a, history = nsga2(problem_a, pop_size=40, n_gen=25)
        print(f"NSGA-II: {n_evals_a} physics-model evaluations, Pareto front size = {len(pareto_F_a)}")

        # --- MOBO (matched evaluation budget) ---
        problem_b = LargeScaleDesignProblem(feed)
        pareto_X_b, pareto_F_b, n_evals_b = mobo_chebyshev(problem_b, total_budget=n_evals_a, n_weight_vectors=10)
        print(f"MOBO:     {n_evals_b} physics-model evaluations, Pareto front size = {len(pareto_F_b)}")

        ref_point = np.array([
            max(pareto_F_a[:, 0].max(), pareto_F_b[:, 0].max()) * 1.05,
            max(pareto_F_a[:, 1].max(), pareto_F_b[:, 1].max()) * 1.05,
        ])
        hv_a = hypervolume_2d(pareto_F_a, ref_point)
        hv_b = hypervolume_2d(pareto_F_b, ref_point)
        print(f"Hypervolume (LCOW x SEC, shared reference point): NSGA-II = {hv_a:.5f}, MOBO = {hv_b:.5f}")
        better = "NSGA-II" if hv_a > hv_b else "MOBO"
        print(f"Larger hypervolume (better Pareto front coverage): {better}")

        preset_label = preset_key.replace("_", " ").title()
        plot_pareto_comparison(pareto_F_a, pareto_F_b, preset_label, f"mo_pareto_comparison_{preset_key}.png")
        plot_nsga2_convergence(history, preset_label, f"mo_nsga2_convergence_{preset_key}.png")
        plot_pareto_3d_projection(pareto_F_a, preset_label, f"mo_pareto_3obj_{preset_key}.png")

        summary.append((preset_key, n_evals_a, len(pareto_F_a), n_evals_b, len(pareto_F_b), hv_a, hv_b, better))

    print(f"\n{'='*72}\nSUMMARY — NSGA-II vs. MOBO at matched evaluation budgets")
    print(f"{'Preset':<22} {'NSGA-II evals':<14} {'NSGA-II |PF|':<14} {'MOBO evals':<12} {'MOBO |PF|':<10} {'HV(NSGA2)':<11} {'HV(MOBO)':<10} {'Winner'}")
    for row in summary:
        preset_key, na, pa, nb, pb, hva, hvb, winner = row
        print(f"{preset_key:<22} {na:<14} {pa:<14} {nb:<12} {pb:<10} {hva:<11.5f} {hvb:<10.5f} {winner}")

    print("\nMulti-objective AI design optimization (NSGA-II vs. MOBO) complete.")
