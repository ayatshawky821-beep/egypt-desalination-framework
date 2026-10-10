"""
analysis_hypervolume_multi_preset.py   (v1.2.0)

Replaces the 3-seed / 1-preset three-objective hypervolume comparison by many seeds
x several Egyptian presets. For each (preset, seed):
  * NSGA-II pop 40 x 25 generations, and pop 100 x 10 generations (~same budget)
  * MOBO (Chebyshev scalarisation) at the same evaluation budget as NSGA-II pop 40
Hypervolume is exact in 2-D (LCOW, SEC) and 3-D (+ pond area); the reference point is
1.05 x the worst value of each objective over the fronts being compared (per seed).

Run:  python3 analysis_hypervolume_multi_preset.py OUT.json N_SEEDS preset1 [preset2 ...]
Writes OUT.json after every run, so a stopped job keeps its results.
"""
import sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
from ai_multi_objective_design import LargeScaleDesignProblem, nsga2, mobo_chebyshev, hypervolume_2d
from desalination_plant_design import EGYPT_BRACKISH_PRESETS


def hv3d(P, ref):
    P = P[np.all(P < ref, axis=1)]
    P = P[np.argsort(P[:, 2])]
    total = 0.0
    for i in range(len(P)):
        z_next = P[i + 1, 2] if i + 1 < len(P) else ref[2]
        total += (z_next - P[i, 2]) * hypervolume_2d(P[:i + 1, :2], ref[:2])
    return total


def run(out, n_seeds, presets):
    res = {}
    for preset in presets:
        feed = EGYPT_BRACKISH_PRESETS[preset]
        res[preset] = []
        for seed in range(n_seeds):
            pa = LargeScaleDesignProblem(feed); _, Fa, na, _ = nsga2(pa, pop_size=40, n_gen=25, seed=seed)
            pc = LargeScaleDesignProblem(feed); _, Fc, nc, _ = nsga2(pc, pop_size=100, n_gen=10, seed=seed)
            pb = LargeScaleDesignProblem(feed); _, Fb, nb = mobo_chebyshev(pb, total_budget=na, n_weight_vectors=10, seed=seed)
            ref = np.vstack([Fa, Fb, Fc]).max(0) * 1.05
            row = {"seed": seed, "evals": [int(na), int(nc), int(nb)],
                   "front": [len(Fa), len(Fc), len(Fb)],
                   "hv2d": [hypervolume_2d(F, ref[:2]) for F in (Fa, Fc, Fb)],
                   "hv3d": [hv3d(F, ref) for F in (Fa, Fc, Fb)]}   # order: NSGA pop40, NSGA pop100, MOBO
            res[preset].append(row)
            json.dump(res, open(out, "w"), default=float)
            print(preset, seed, "3-D HV ratio NSGA40/MOBO %.3f  NSGA100/MOBO %.3f" % (row["hv3d"][0] / row["hv3d"][2], row["hv3d"][1] / row["hv3d"][2]), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
