import warnings, numpy as np
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
pa = LargeScaleDesignProblem(feed); Xa, Fa, na, _ = nsga2(pa, pop_size=40, n_gen=25, seed=42)
pb = LargeScaleDesignProblem(feed); Xb, Fb, nb = mobo_chebyshev(pb, total_budget=na, n_weight_vectors=10, seed=42)
ref3 = np.maximum(Fa.max(0), Fb.max(0)) * 1.05
ref2 = ref3[:2]
print("evals:", na, nb, "| front sizes:", len(Fa), len(Fb))
print("2-D HV (LCOW x SEC): NSGA-II %.6f  MOBO %.6f" % (hypervolume_2d(Fa, ref2), hypervolume_2d(Fb, ref2)))
ha, hb = hv3d(Fa, ref3), hv3d(Fb, ref3)
print("3-D HV (LCOW x SEC x pond area): NSGA-II %.4f  MOBO %.4f  ratio MOBO/NSGA-II = %.4f" % (ha, hb, hb/ha))
print("objective ranges NSGA-II: LCOW %.3f-%.3f, SEC %.2f-%.2f, pond %.0f-%.0f ha" % (Fa[:,0].min(), Fa[:,0].max(), Fa[:,1].min(), Fa[:,1].max(), Fa[:,2].min(), Fa[:,2].max()))
print("objective ranges MOBO   : LCOW %.3f-%.3f, SEC %.2f-%.2f, pond %.0f-%.0f ha" % (Fb[:,0].min(), Fb[:,0].max(), Fb[:,1].min(), Fb[:,1].max(), Fb[:,2].min(), Fb[:,2].max()))
np.savez("fronts_seed42.npz", Fa=Fa, Fb=Fb, Xa=Xa, Xb=Xb)
