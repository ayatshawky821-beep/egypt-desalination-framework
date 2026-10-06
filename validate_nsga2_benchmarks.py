"""
validate_nsga2_benchmarks.py

Validates the from-scratch NSGA-II implementation in ai_multi_objective_design.py
against ZDT1 and ZDT2 (Zitzler-Deb-Thiele 2000), the standard synthetic
benchmark suite for multi-objective evolutionary algorithms. Both have a
known, closed-form true Pareto front, so the algorithm's output can be
checked against ground truth BEFORE trusting it on the real (no-known-
answer) desalination design problem in ai_multi_objective_design.py.

ZDT1: convex true front f2 = 1 - sqrt(f1), f1 in [0,1]
ZDT2: non-convex (concave) true front f2 = 1 - f1^2, f1 in [0,1]
Testing both checks that the implementation does not have a bias toward
convex fronts (a known failure mode of some weighted-sum-based methods,
though NOT expected for genuine NSGA-II, which is precisely the property
this validation checks for).

Metric: Generational Distance (GD) -- mean Euclidean distance from each
found point to its nearest point on the TRUE front. Lower is better;
GD -> 0 as the found front converges to the true front.

Run:
    python3 validate_nsga2_benchmarks.py
"""

import numpy as np
import matplotlib.pyplot as plt

from ai_multi_objective_design import (
    fast_non_dominated_sort, crowding_distance, tournament_select,
    sbx_crossover, polynomial_mutation,
)

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10.5})

N_VAR = 30  # standard ZDT dimensionality


def zdt1(x):
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (N_VAR - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.array([f1, f2])


def zdt2(x):
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (N_VAR - 1)
    f2 = g * (1 - (f1 / g) ** 2)
    return np.array([f1, f2])


def true_front_zdt1(n=200):
    f1 = np.linspace(0, 1, n)
    return np.column_stack([f1, 1 - np.sqrt(f1)])


def true_front_zdt2(n=200):
    f1 = np.linspace(0, 1, n)
    return np.column_stack([f1, 1 - f1 ** 2])


def generational_distance(found_F, true_F):
    """Mean distance from each found point to its nearest true-front point."""
    dists = []
    for f in found_F:
        d = np.min(np.linalg.norm(true_F - f, axis=1))
        dists.append(d)
    return np.mean(dists)


def inverted_generational_distance(found_F, true_F):
    """Mean distance from each TRUE-front point to its nearest found point
    -- penalizes gaps/poor coverage of the found front, complementing GD."""
    dists = []
    for t in true_F:
        d = np.min(np.linalg.norm(found_F - t, axis=1))
        dists.append(d)
    return np.mean(dists)


def nsga2_generic(objective_fn, n_var, bounds, pop_size=100, n_gen=100, seed=42):
    """Generic NSGA-II loop (same algorithm core as ai_multi_objective_design.py's
    nsga2(), but parameterized for an arbitrary objective function and
    dimensionality rather than the 2-variable desalination problem, so it
    can run ZDT1/ZDT2 at their standard 30-variable dimensionality)."""
    rng = np.random.default_rng(seed)
    bounds_arr = np.array(bounds)

    def eval_pop(X):
        return np.array([objective_fn(x) for x in X])

    X = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1], size=(pop_size, n_var))
    F = eval_pop(X)

    for gen in range(n_gen):
        fronts = fast_non_dominated_sort(F)
        ranks = np.zeros(len(X), dtype=int)
        crowd = np.zeros(len(X))
        for rank, front in enumerate(fronts):
            ranks[front] = rank
            crowd[front] = crowding_distance(F[front])

        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_select(X, ranks, crowd, rng)
            p2 = tournament_select(X, ranks, crowd, rng)
            c1, c2 = sbx_crossover(p1, p2, bounds_arr, rng)
            c1 = polynomial_mutation(c1, bounds_arr, rng)
            c2 = polynomial_mutation(c2, bounds_arr, rng)
            offspring.extend([c1, c2])
        offspring = np.array(offspring[:pop_size])
        F_off = eval_pop(offspring)

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

    final_fronts = fast_non_dominated_sort(F)
    return F[final_fronts[0]]


def plot_validation(found_F, true_F, name, gd, igd, filename):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(true_F[:, 0], true_F[:, 1], "k-", linewidth=2, label="True Pareto front (analytical)")
    ax.scatter(found_F[:, 0], found_F[:, 1], c="tab:red", s=18, alpha=0.7, label=f"NSGA-II found front (n={len(found_F)})")
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_title(f"NSGA-II Validation on {name}\nGD = {gd:.5f}, IGD = {igd:.5f} (lower is better; 0 = perfect)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


if __name__ == "__main__":
    bounds = [(0.0, 1.0)] * N_VAR

    print("=" * 70)
    print("Validating NSGA-II on ZDT1 (convex true front)")
    found = nsga2_generic(zdt1, N_VAR, bounds, pop_size=100, n_gen=100, seed=42)
    true_f = true_front_zdt1()
    gd = generational_distance(found, true_f)
    igd = inverted_generational_distance(found, true_f)
    print(f"Pareto front size: {len(found)}")
    print(f"Generational Distance (GD): {gd:.5f}  [reference: well-converged NSGA-II typically achieves GD < 0.01 on ZDT1]")
    print(f"Inverted GD (coverage): {igd:.5f}")
    plot_validation(found, true_f, "ZDT1", gd, igd, "zdt1_validation.png")

    print("\n" + "=" * 70)
    print("Validating NSGA-II on ZDT2 (non-convex true front)")
    found2 = nsga2_generic(zdt2, N_VAR, bounds, pop_size=100, n_gen=100, seed=42)
    true_f2 = true_front_zdt2()
    gd2 = generational_distance(found2, true_f2)
    igd2 = inverted_generational_distance(found2, true_f2)
    print(f"Pareto front size: {len(found2)}")
    print(f"Generational Distance (GD): {gd2:.5f}  [reference: well-converged NSGA-II typically achieves GD < 0.01 on ZDT2]")
    print(f"Inverted GD (coverage): {igd2:.5f}")
    plot_validation(found2, true_f2, "ZDT2", gd2, igd2, "zdt2_validation.png")

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print(f"ZDT1: GD={gd:.5f}, IGD={igd:.5f} -- {'PASS' if gd < 0.01 and igd < 0.01 else 'MARGINAL/FAIL'} against the GD<0.01 reference")
    print(f"ZDT2: GD={gd2:.5f}, IGD={igd2:.5f} -- {'PASS' if gd2 < 0.01 and igd2 < 0.01 else 'MARGINAL/FAIL'} against the GD<0.01 reference")
    print("If both pass: the NSGA-II implementation correctly finds both convex and")
    print("non-convex Pareto fronts, supporting its use on the real desalination")
    print("design problem (which has unknown front shape) in ai_multi_objective_design.py.")
