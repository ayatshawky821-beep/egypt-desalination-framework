"""
analysis_no_ai_small_budget.py   (v1.2.0)

Companion to analysis_no_ai_baselines.py. The AI methods are usually motivated by expensive
evaluations, which means SMALL evaluation budgets. Here every method gets about 50, 100, 200 or 400
evaluations on the same three-objective problem, and is scored by hypervolume share (same fixed
reference, same dense-grid denominator as analysis_no_ai_baselines.py).
  grid: floor(sqrt(budget))^2 points; random: budget points; NSGA-II: pop x (gens+1) = budget;
  MOBO: total_budget argument = budget (it also spends 8 warm-up evaluations, so it uses slightly more).
Run: python3 analysis_no_ai_small_budget.py OUT.json N_SEEDS preset1 [preset2 ...]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[_v] = '1'
import sys, json, time, math, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from ai_multi_objective_design import LargeScaleDesignProblem, nsga2, mobo_chebyshev
from analysis_no_ai_baselines import evals, grid_pts, reference, hv
from desalination_plant_design import EGYPT_BRACKISH_PRESETS

NSGA = {50: (10, 4), 100: (20, 4), 200: (20, 9), 400: (40, 9)}   # (pop, gens): pop*(gens+1) = budget


def one(args):
    preset, seed = args
    feed = EGYPT_BRACKISH_PRESETS[preset]
    ref, hv_dense = reference(preset)
    out = {"preset": preset, "seed": seed, "budgets": {}}
    for B, (pop, gens) in NSGA.items():
        row = {}
        def rec(name, F, n): row[name] = {"hv_share": hv(F, ref) / hv_dense, "evals": int(n)}
        prob = LargeScaleDesignProblem(feed); b = prob.bounds(); n = int(math.isqrt(B))
        rec("grid", evals(prob, *grid_pts(prob, n)), prob.n_evals)
        prob = LargeScaleDesignProblem(feed); rng = np.random.default_rng(seed)
        rec("random", evals(prob, rng.uniform(*b[0], B), rng.uniform(*b[1], B)), prob.n_evals)
        prob = LargeScaleDesignProblem(feed); _, Fa, na, _ = nsga2(prob, pop_size=pop, n_gen=gens, seed=seed); rec("nsga2", Fa, na)
        prob = LargeScaleDesignProblem(feed); _, Fb, nb = mobo_chebyshev(prob, total_budget=B, n_weight_vectors=10, seed=seed); rec("mobo", Fb, nb)
        out["budgets"][B] = row
    print(preset, seed, {B: {k: round(v["hv_share"], 3) for k, v in r.items()} for B, r in out["budgets"].items()}, flush=True)
    return out


if __name__ == "__main__":
    outf, n_seeds, presets = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    jobs = [(p, s) for p in presets for s in range(n_seeds)]
    res = []
    with Pool(2) as pool:
        for o in pool.imap_unordered(one, jobs):
            res.append(o); json.dump(res, open(outf, "w"), default=float)
