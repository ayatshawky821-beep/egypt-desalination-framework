"""
analysis_no_ai_baselines.py   (v1.2.0)

Conventional (no-AI) design workflows on the SAME three-objective problem and the SAME
evaluation budget as NSGA-II and MOBO (decision variables: recovery, capacity; objectives:
LCOW, SEC, pond area; ai_multi_objective_design.LargeScaleDesignProblem).

Methods (all use the physics model only):
  rule_of_thumb  one design: the framework's default recovery rule at mid capacity (1 evaluation)
  manual_sweep   5 x 5 parametric sweep, as in a conventional parametric study (25 evaluations)
  grid           32 x 32 full-factorial grid (1,024 evaluations, ~ NSGA-II budget of 1,040)
  random         1,040 uniform random designs (seeded)
  nsga2_40, mobo the AI methods, as in analysis_hypervolume_multi_preset.py (re-run here so that
                 every method is scored against the same reference)
Score: exact 3-D hypervolume against a FIXED reference point = 1.05 x the worst value of each
objective over a dense 81 x 81 grid, expressed as a share of the dense-grid front's hypervolume
(an estimate of the achievable maximum). Wall-clock time per method is recorded.

Run: python3 analysis_no_ai_baselines.py OUT.json N_SEEDS preset1 [preset2 ...]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[_v] = '1'
import sys, json, time, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from ai_multi_objective_design import LargeScaleDesignProblem, nsga2, mobo_chebyshev
from analysis_hypervolume_multi_preset import hv3d
from desalination_plant_design import EGYPT_BRACKISH_PRESETS, default_recovery_for


def pareto(F):
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dom = np.all(F >= F[i], axis=1) & np.any(F > F[i], axis=1)
            keep &= ~dom
    return F[keep]


def hv(F, ref):
    return hv3d(pareto(F), ref)


def evals(prob, R, C):
    return np.array([prob.evaluate(r, c) for r, c in zip(R, C)])


def grid_pts(prob, n):
    b = prob.bounds()
    r, c = np.meshgrid(np.linspace(*b[0], n), np.linspace(*b[1], n))
    return r.ravel(), c.ravel()


def reference(preset):
    prob = LargeScaleDesignProblem(EGYPT_BRACKISH_PRESETS[preset])
    F = evals(prob, *grid_pts(prob, 81))
    ref = F.max(0) * 1.05
    return ref, hv(F, ref)


def one(args):
    preset, seed = args
    feed = EGYPT_BRACKISH_PRESETS[preset]
    ref, hv_dense = reference(preset)
    out = {"preset": preset, "seed": seed, "hv_dense": hv_dense, "ref": ref.tolist(), "methods": {}}

    def rec(name, F, secs, n):
        out["methods"][name] = {"hv_share": hv(F, ref) / hv_dense, "seconds": secs, "evals": int(n)}

    prob = LargeScaleDesignProblem(feed); b = prob.bounds()
    t = time.time()
    F = evals(prob, [default_recovery_for(feed)], [0.5 * (b[1, 0] + b[1, 1])]); rec("rule_of_thumb", F, time.time() - t, 1)
    prob = LargeScaleDesignProblem(feed); t = time.time()
    F = evals(prob, *grid_pts(prob, 5)); rec("manual_sweep_5x5", F, time.time() - t, prob.n_evals)
    prob = LargeScaleDesignProblem(feed); t = time.time()
    F = evals(prob, *grid_pts(prob, 32)); rec("grid_32x32", F, time.time() - t, prob.n_evals)
    prob = LargeScaleDesignProblem(feed); rng = np.random.default_rng(seed); t = time.time()
    R = rng.uniform(*b[0], 1040); C = rng.uniform(*b[1], 1040)
    F = evals(prob, R, C); rec("random_1040", F, time.time() - t, prob.n_evals)
    prob = LargeScaleDesignProblem(feed); t = time.time()
    _, Fa, na, _ = nsga2(prob, pop_size=40, n_gen=25, seed=seed); rec("nsga2_pop40", Fa, time.time() - t, na)
    prob = LargeScaleDesignProblem(feed); t = time.time()
    _, Fb, nb = mobo_chebyshev(prob, total_budget=na, n_weight_vectors=10, seed=seed); rec("mobo", Fb, time.time() - t, nb)
    print(preset, seed, {k: round(v["hv_share"], 3) for k, v in out["methods"].items()}, flush=True)
    return out


if __name__ == "__main__":
    outf, n_seeds, presets = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    jobs = [(p, s) for p in presets for s in range(n_seeds)]
    res = []
    with Pool(2) as pool:
        for o in pool.imap_unordered(one, jobs):
            res.append(o); json.dump(res, open(outf, "w"), default=float)
