
import warnings, json, numpy as np
warnings.filterwarnings("ignore")
from ai_multi_objective_design import LargeScaleDesignProblem, nsga2, mobo_chebyshev, hypervolume_2d
from desalination_plant_design import EGYPT_BRACKISH_PRESETS

def hv3d(P, ref):
    P = P[np.all(P < ref, axis=1)]
    P = P[np.argsort(P[:, 2])]
    total = 0.0
    for i in range(len(P)):
        z_next = P[i+1, 2] if i + 1 < len(P) else ref[2]
        total += (z_next - P[i, 2]) * hypervolume_2d(P[:i+1, :2], ref[:2])
    return total

feed = EGYPT_BRACKISH_PRESETS["el_moghra_aquifer"]
res = {}
d = np.load("fronts_seed42.npz"); Fa42, Fb42 = d["Fa"], d["Fb"]

# (1) NSGA-II population-size variants at (about) the same evaluation budget, seed 42, vs the saved MOBO front
variants = {"pop20_gen50": (20, 50), "pop40_gen25": (40, 25), "pop80_gen12": (80, 12), "pop100_gen10": (100, 10)}
fronts = {}
for name, (pop, gen) in variants.items():
    p = LargeScaleDesignProblem(feed); X, F, n, _ = nsga2(p, pop_size=pop, n_gen=gen, seed=42)
    fronts[name] = (F, n)
allF = np.vstack([Fb42] + [f for f, _ in fronts.values()])
ref = allF.max(0) * 1.05
hb = hv3d(Fb42, ref)
print("seed 42, common reference point; MOBO 3-D HV = %.3f (|front|=%d)" % (hb, len(Fb42)))
for name, (F, n) in fronts.items():
    h = hv3d(F, ref); print("  NSGA-II %-13s evals=%4d |front|=%3d  3-D HV=%.3f  ratio to MOBO = %.3f" % (name, n, len(F), h, h/hb))
    res["seed42_" + name] = {"evals": n, "front": len(F), "hv3d": h, "ratio_to_mobo": h/hb}
res["seed42_mobo"] = {"front": len(Fb42), "hv3d": hb}

# (2) two more seeds, original configuration (pop 40, 25 generations; MOBO with matched budget)
for seed in (0, 1):
    pa = LargeScaleDesignProblem(feed); Xa, Fa, na, _ = nsga2(pa, pop_size=40, n_gen=25, seed=seed)
    pb = LargeScaleDesignProblem(feed); Xb, Fb, nb = mobo_chebyshev(pb, total_budget=na, n_weight_vectors=10, seed=seed)
    r = np.maximum(Fa.max(0), Fb.max(0)) * 1.05
    ha, hbb = hv3d(Fa, r), hv3d(Fb, r)
    print("seed %d: |front| %d vs %d | 3-D HV NSGA-II %.3f MOBO %.3f ratio %.3f | 2-D HV %.5f vs %.5f" % (
        seed, len(Fa), len(Fb), ha, hbb, hbb/ha, hypervolume_2d(Fa, r[:2]), hypervolume_2d(Fb, r[:2])))
    res["seed%d" % seed] = {"front_nsga2": len(Fa), "front_mobo": len(Fb), "hv3d_nsga2": ha, "hv3d_mobo": hbb}
json.dump(res, open("analysis_b2.json", "w"), default=float)
