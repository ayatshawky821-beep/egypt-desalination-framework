"""
analysis_reoptimise_element_sec.py

Does the choice of membrane-train energy model change the design decision?
Re-runs the three-objective NSGA-II design (LCOW, SEC, pond area; recovery 0.50-0.90,
capacity 100,000-300,000 m3/d) for five Egyptian presets with
  (a) the framework's lumped SEC (as in the paper), and
  (b) the element-by-element SEC of analysis_element_level_ro.py,
everything else identical. Designs whose simulated pressure exceeds the 41-bar
element rating are penalised in (b).

Reports, per preset and seed: the cheapest design (recovery, LCOW, pond area) and the
smallest pond area reachable within 1 % of the cheapest LCOW.

Run:  python3 analysis_reoptimise_element_sec.py
"""
import json
import numpy as np
from desalination_plant_design import (EGYPT_BRACKISH_PRESETS, PlantCapacity, PlantFlows, EconomicAnalysis,
                                       BrineManagementSystem)
from ai_multi_objective_design import LargeScaleDesignProblem, nsga2
from analysis_element_level_ro import array, calibrate

PRESETS = ["el_moghra_aquifer", "nile_delta_shallow", "western_desert_nubian", "sinai_coastal",
           "agricultural_drainage_nile_delta"]
SEEDS = [42, 1, 2]
A0, B0 = calibrate()


class ElementSECProblem(LargeScaleDesignProblem):
    """Same problem, but SEC from the element-by-element simulation (cached on recovery)."""
    def __init__(self, feed, **kw):
        super().__init__(feed, **kw)
        self.cache = {}

    def sec(self, r):
        key = round(r, 4)
        if key not in self.cache:
            e = array(self.feed.tds_mg_l, key, self.feed.temp_c, A0, B0)
            self.cache[key] = (e["sec_kwh_m3"], e["within_41bar"])
        return self.cache[key]

    def evaluate(self, recovery, capacity_m3d):
        self.n_evals += 1
        sec, ok = self.sec(recovery)
        flows = PlantFlows.from_capacity(PlantCapacity(permeate_flow_m3d=capacity_m3d, recovery=recovery))
        lcow = EconomicAnalysis(capacity_m3d=capacity_m3d, water_type=self.feed.water_type,
                                specific_energy_kwh_m3=sec).run_full_analysis()["lcow"]["lcow_usd_m3"]
        pond = BrineManagementSystem(concentrate_flow_m3d=flows.concentrate_m3d, concentrate_tds_mg_l=0.0,
                                     feed_tds_mg_l=self.feed.tds_mg_l, is_coastal_site=False, is_arid_climate=True
                                     ).design_evaporation_ponds_alternative().design["evaporation_pond_option"]["required_pond_area_hectares"]
        if not ok:  # infeasible pressure: push out of the front
            return lcow + 10.0, sec + 10.0, pond + 1e5
        return lcow, sec, pond


def summarise(X, F):
    i = int(np.argmin(F[:, 0]))
    near = F[:, 0] <= F[i, 0] * 1.01
    j = int(np.argmin(np.where(near, F[:, 2], np.inf)))
    return {"cheapest": {"recovery": float(X[i, 0]), "capacity": float(X[i, 1]), "lcow": float(F[i, 0]),
                         "sec": float(F[i, 1]), "pond_ha": float(F[i, 2])},
            "min_pond_within_1pct": {"recovery": float(X[j, 0]), "lcow": float(F[j, 0]), "sec": float(F[j, 1]),
                                     "pond_ha": float(F[j, 2])},
            "recovery_range_on_front": [float(X[:, 0].min()), float(X[:, 0].max())],
            "front": {"X": X.tolist(), "F": F.tolist()}}


def main():
    out = {}
    for key in PRESETS:
        feed = EGYPT_BRACKISH_PRESETS[key]
        out[key] = {"lumped": [], "element": []}
        for seed in SEEDS:
            for label, P in (("lumped", LargeScaleDesignProblem), ("element", ElementSECProblem)):
                prob = P(feed)
                X, F, n, _ = nsga2(prob, pop_size=80, n_gen=12, seed=seed)
                out[key][label].append({"seed": seed, "n_evals": n, **summarise(X, F)})
            L, E = out[key]["lumped"][-1], out[key]["element"][-1]
            print(f"{key:<34} seed {seed:>2} | cheapest: lumped r={L['cheapest']['recovery']:.3f} "
                  f"pond {L['cheapest']['pond_ha']:7.0f} ha LCOW {L['cheapest']['lcow']:.3f} | element r={E['cheapest']['recovery']:.3f} "
                  f"pond {E['cheapest']['pond_ha']:7.0f} ha LCOW {E['cheapest']['lcow']:.3f} || min pond within 1%: "
                  f"lumped {L['min_pond_within_1pct']['pond_ha']:.0f} ha, element {E['min_pond_within_1pct']['pond_ha']:.0f} ha", flush=True)
    json.dump(out, open("analysis_reoptimise_element_sec.json", "w"))


if __name__ == "__main__":
    main()
