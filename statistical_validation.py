"""
statistical_validation.py

Re-runs this paper's two headline comparisons (NSGA-II vs. MOBO hypervolume;
RL vs. greedy vs. random allocation cost) across MULTIPLE random seeds,
reporting mean +- standard deviation and a paired statistical significance
test, rather than the single-seed point estimates in the main results.
Also runs a basic hyperparameter sensitivity sweep for each method.

WHY THIS MATTERS: a single-seed result cannot distinguish a genuine,
reproducible effect from noise. This script answers, for each headline
claim: does it hold up across seeds, and is the difference statistically
significant (Wilcoxon signed-rank for paired per-seed comparisons, alpha
= 0.05)?

Run:
    python3 statistical_validation.py
(Runtime: several minutes -- this deliberately repeats expensive
Bayesian-optimization and NSGA-II runs 20x each.)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu

from ai_multi_objective_design import (
    LargeScaleDesignProblem, nsga2, mobo_chebyshev, hypervolume_2d,
)
from ai_multi_plant_rl import train_reinforce, run_episode
from desalination_plant_design import EGYPT_BRACKISH_PRESETS

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10.5})

N_SEEDS = 20


# ===========================================================================
# 1. NSGA-II vs. MOBO: hypervolume across N_SEEDS independent runs
# ===========================================================================

def run_multiseed_nsga2_vs_mobo(preset_key="el_moghra_aquifer", n_seeds=N_SEEDS):
    feed = EGYPT_BRACKISH_PRESETS[preset_key]
    hv_nsga2, hv_mobo = [], []

    for seed in range(n_seeds):
        problem_a = LargeScaleDesignProblem(feed)
        pareto_X_a, pareto_F_a, n_evals_a, _ = nsga2(problem_a, pop_size=40, n_gen=25, seed=seed)

        problem_b = LargeScaleDesignProblem(feed)
        pareto_X_b, pareto_F_b, n_evals_b = mobo_chebyshev(problem_b, total_budget=n_evals_a, n_weight_vectors=10, seed=seed)

        ref_point = np.array([
            max(pareto_F_a[:, 0].max(), pareto_F_b[:, 0].max()) * 1.05,
            max(pareto_F_a[:, 1].max(), pareto_F_b[:, 1].max()) * 1.05,
        ])
        hv_a = hypervolume_2d(pareto_F_a, ref_point)
        hv_b = hypervolume_2d(pareto_F_b, ref_point)
        hv_nsga2.append(hv_a)
        hv_mobo.append(hv_b)
        print(f"  seed {seed:2d}: NSGA-II HV={hv_a:.5f}  MOBO HV={hv_b:.5f}  diff={hv_a-hv_b:+.6f}")

    return np.array(hv_nsga2), np.array(hv_mobo)


def summarize_and_test(name_a, vals_a, name_b, vals_b):
    print(f"\n{name_a}: mean={vals_a.mean():.5f}  std={vals_a.std(ddof=1):.5f}  "
          f"[min={vals_a.min():.5f}, max={vals_a.max():.5f}]")
    print(f"{name_b}: mean={vals_b.mean():.5f}  std={vals_b.std(ddof=1):.5f}  "
          f"[min={vals_b.min():.5f}, max={vals_b.max():.5f}]")
    diff = vals_a - vals_b
    if np.allclose(diff, 0):
        print("All paired differences are exactly zero -- Wilcoxon test is undefined (identical results every seed).")
        return None
    try:
        stat, p = wilcoxon(vals_a, vals_b)
        print(f"Wilcoxon signed-rank test: statistic={stat:.3f}, p-value={p:.4f}  "
              f"-- {'SIGNIFICANT (p<0.05)' if p < 0.05 else 'NOT significant (p>=0.05)'} difference")
        return p
    except ValueError as e:
        print(f"Wilcoxon test could not be computed: {e}")
        return None


# ===========================================================================
# 2. RL vs. greedy vs. random: multi-seed training + evaluation
# ===========================================================================

def run_multiseed_rl_comparison(n_seeds=N_SEEDS):
    rl_lcows, random_lcows = [], []
    greedy_lcow = None  # deterministic; computed once

    for seed in range(n_seeds):
        policy, _, _ = train_reinforce(n_episodes=600, lr=0.08, seed=seed)
        rng_eval = np.random.default_rng(1000 + seed)
        _, rl_lcow, _, _, _ = run_episode(policy, rng_eval)
        rl_lcows.append(rl_lcow)

        if greedy_lcow is None:
            rng_g = np.random.default_rng(999)
            _, greedy_lcow, _, _, _ = run_episode(None, rng_g, greedy=True)

        rng_r = np.random.default_rng(2000 + seed)
        _, r_lcow, _, _, _ = run_episode(None, rng_r, random_policy=True)
        random_lcows.append(r_lcow)
        print(f"  seed {seed:2d}: RL LCOW={rl_lcow:.4f}  Random LCOW={r_lcow:.4f}")

    return np.array(rl_lcows), np.array(random_lcows), greedy_lcow


# ===========================================================================
# 3. Hyperparameter sensitivity
# ===========================================================================

def nsga2_hyperparam_sensitivity(preset_key="el_moghra_aquifer"):
    feed = EGYPT_BRACKISH_PRESETS[preset_key]
    configs = [(20, 15), (40, 25), (80, 25), (40, 50)]
    print(f"\n{'pop_size':<10}{'n_gen':<8}{'n_evals':<10}{'|Pareto|':<10}{'Hypervolume (vs fixed ref)'}")
    results = []
    # Fixed reference point computed from the largest/most thorough run first
    ref_problem = LargeScaleDesignProblem(feed)
    ref_X, ref_F, _, _ = nsga2(ref_problem, pop_size=80, n_gen=50, seed=0)
    ref_point = np.array([ref_F[:, 0].max() * 1.1, ref_F[:, 1].max() * 1.1])
    for pop_size, n_gen in configs:
        problem = LargeScaleDesignProblem(feed)
        pareto_X, pareto_F, n_evals, _ = nsga2(problem, pop_size=pop_size, n_gen=n_gen, seed=0)
        hv = hypervolume_2d(pareto_F, ref_point)
        results.append((pop_size, n_gen, n_evals, len(pareto_F), hv))
        print(f"{pop_size:<10}{n_gen:<8}{n_evals:<10}{len(pareto_F):<10}{hv:.5f}")
    return results


def reinforce_hyperparam_sensitivity():
    configs = [(150, 0.08), (600, 0.08), (600, 0.02), (600, 0.15)]
    print(f"\n{'n_episodes':<12}{'lr':<8}{'RL LCOW (seed=42)'}")
    results = []
    for n_episodes, lr in configs:
        policy, _, _ = train_reinforce(n_episodes=n_episodes, lr=lr, seed=42)
        rng_eval = np.random.default_rng(999)
        _, lcow, _, _, _ = run_episode(policy, rng_eval)
        results.append((n_episodes, lr, lcow))
        print(f"{n_episodes:<12}{lr:<8}{lcow:.4f}")
    return results


# ===========================================================================
# 4. Plots
# ===========================================================================

def plot_boxplot_comparison(vals_a, name_a, vals_b, name_b, greedy_val, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    data = [vals_a, vals_b]
    bp = ax.boxplot(data, labels=[name_a, name_b], patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], ['tab:blue', 'tab:orange']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    if greedy_val is not None:
        ax.axhline(greedy_val, color='tab:green', linestyle='--', linewidth=2, label=f'Greedy (deterministic) = {greedy_val:.4f}')
        ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


def run_nsga2_mobo_chunk(seed_start, seed_end, preset_key="el_moghra_aquifer",
                          results_file="nsga2_mobo_results.json"):
    """Runs seeds [seed_start, seed_end) and appends to a JSON results file,
    so the full N_SEEDS sweep (expensive mainly due to MOBO's per-seed GP
    fitting, ~100s/seed) can be executed across several shorter calls
    instead of one long-running process."""
    import json, os
    feed = EGYPT_BRACKISH_PRESETS[preset_key]

    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
    else:
        results = {}

    for seed in range(seed_start, seed_end):
        if str(seed) in results:
            print(f"  seed {seed}: already done, skipping")
            continue
        problem_a = LargeScaleDesignProblem(feed)
        pareto_X_a, pareto_F_a, n_evals_a, _ = nsga2(problem_a, pop_size=40, n_gen=25, seed=seed)

        problem_b = LargeScaleDesignProblem(feed)
        pareto_X_b, pareto_F_b, n_evals_b = mobo_chebyshev(problem_b, total_budget=n_evals_a, n_weight_vectors=10, seed=seed)

        ref_point = np.array([
            max(pareto_F_a[:, 0].max(), pareto_F_b[:, 0].max()) * 1.05,
            max(pareto_F_a[:, 1].max(), pareto_F_b[:, 1].max()) * 1.05,
        ])
        hv_a = hypervolume_2d(pareto_F_a, ref_point)
        hv_b = hypervolume_2d(pareto_F_b, ref_point)
        results[str(seed)] = {"hv_nsga2": float(hv_a), "hv_mobo": float(hv_b)}
        print(f"  seed {seed:2d}: NSGA-II HV={hv_a:.5f}  MOBO HV={hv_b:.5f}  diff={hv_a-hv_b:+.6f}")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    print("=" * 72)
    print(f"1. NSGA-II vs. MOBO hypervolume across {N_SEEDS} seeds (El Moghra Aquifer)")
    print("=" * 72)
    hv_nsga2, hv_mobo = run_multiseed_nsga2_vs_mobo()
    p_value_mo = summarize_and_test("NSGA-II hypervolume", hv_nsga2, "MOBO hypervolume", hv_mobo)
    plot_boxplot_comparison(hv_nsga2, "NSGA-II", hv_mobo, "MOBO", None,
                             f"Hypervolume Across {N_SEEDS} Seeds: NSGA-II vs. MOBO",
                             "Hypervolume (LCOW x SEC)", "stat_nsga2_vs_mobo_boxplot.png")

    print("\n" + "=" * 72)
    print(f"2. RL vs. greedy vs. random across {N_SEEDS} seeds (multi-plant allocation)")
    print("=" * 72)
    rl_lcows, random_lcows, greedy_lcow = run_multiseed_rl_comparison()
    print(f"\nGreedy (deterministic): LCOW = {greedy_lcow:.4f}")
    p_value_rl = summarize_and_test("RL LCOW", rl_lcows, "Random LCOW", random_lcows)
    plot_boxplot_comparison(rl_lcows, "RL (REINFORCE)", random_lcows, "Random", greedy_lcow,
                             f"System LCOW Across {N_SEEDS} Seeds: RL vs. Random vs. Greedy",
                             "Total system LCOW (USD/m3)", "stat_rl_vs_random_boxplot.png")

    print("\n" + "=" * 72)
    print("3. NSGA-II hyperparameter sensitivity (population size x generations)")
    print("=" * 72)
    nsga2_sensitivity = nsga2_hyperparam_sensitivity()

    print("\n" + "=" * 72)
    print("4. REINFORCE hyperparameter sensitivity (episodes x learning rate)")
    print("=" * 72)
    reinforce_sensitivity = reinforce_hyperparam_sensitivity()

    print("\n" + "=" * 72)
    print("STATISTICAL VALIDATION SUMMARY")
    print("=" * 72)
    print(f"NSGA-II vs MOBO hypervolume: mean diff = {(hv_nsga2-hv_mobo).mean():.6f}, "
          f"Wilcoxon p = {p_value_mo if p_value_mo is not None else 'N/A (identical every seed)'}")
    print(f"RL vs Random LCOW: mean diff = {(rl_lcows-random_lcows).mean():.5f}, "
          f"Wilcoxon p = {p_value_rl if p_value_rl is not None else 'N/A'}")
    print(f"RL mean vs greedy (deterministic): {rl_lcows.mean():.4f} vs {greedy_lcow:.4f}  "
          f"(gap = {(rl_lcows.mean()-greedy_lcow)/greedy_lcow*100:+.2f}%)")
