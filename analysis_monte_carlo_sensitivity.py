"""(v1.2.0) Global sensitivity and Monte Carlo analysis of the framework's placeholder inputs.

Outputs: LCOW (USD/m3) and evaporation-pond area (ha) for the El Moghra preset.
Ranges are the author's planning ranges (documented in RANGES), NOT measured
distributions; results show which assumptions matter, not a probability of cost.
Sobol indices: Saltelli sampling, Saltelli (2010) first-order and Jansen total-order estimators.
"""
import json, math, sys
import numpy as np
from desalination_plant_design import (EGYPT_BRACKISH_PRESETS, PlantCapacity, PlantFlows, default_recovery_for,
    ReverseOsmosisSystem, EconomicAnalysis, EconomicAssumptions)

PRESET = "el_moghra_aquifer"
# name: (low, high, source of range)
RANGES = {
    "net_evaporation_m_yr": (0.75, 2.8),   # Texas average 0.75 (Nicot & Gross 2009) .. 2.8 hot desert
    "pumping_head_m":       (30.0, 180.0), # planning range 30-150 m, preset up to 180 m
    "tariff_usd_kwh":       (0.03, 0.10),
    "opex_nonenergy_usd_m3":(0.20, 0.60),  # sensitivity band used in the framework
    "capex_usd_per_m3d":    (650.0, 1100.0),
    "capex_exponent":       (-0.20, 0.05),
    "discount_rate":        (0.06, 0.14),
    "availability":         (0.85, 0.95),
}
NAMES = list(RANGES)

def model(p, cap=20000.0, preset=PRESET):
    feed = EGYPT_BRACKISH_PRESETS[preset]
    rec = default_recovery_for(feed)
    fl = PlantFlows.from_capacity(PlantCapacity(permeate_flow_m3d=cap, recovery=rec))
    ro = ReverseOsmosisSystem(flow_m3d=fl.feed_to_membranes_m3d, permeate_m3d=fl.permeate_m3d,
        concentrate_m3d=fl.concentrate_m3d, feed_tds_mg_l=feed.tds_mg_l, water_type=feed.water_type,
        intake_m3d=fl.intake_m3d, pumping_head_m=p["pumping_head_m"]).design()
    lo, hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
    a = EconomicAssumptions(capex_per_m3d_brackish=p["capex_usd_per_m3d"], opex_nonenergy_brackish=p["opex_nonenergy_usd_m3"],
        electricity_tariff_usd_kwh=p["tariff_usd_kwh"], discount_rate=p["discount_rate"],
        availability_factor=p["availability"], economies_of_scale_exponent=p["capex_exponent"])
    e = EconomicAnalysis(capacity_m3d=cap, water_type=feed.water_type, specific_energy_kwh_m3=(lo+hi)/2,
        assumptions=a).run_full_analysis()
    pond_ha = fl.concentrate_m3d * 365 / p["net_evaporation_m_yr"] / 1e4
    return e["lcow"]["lcow_usd_m3"], pond_ha

BASE = {"net_evaporation_m_yr": 2.2, "pumping_head_m": EGYPT_BRACKISH_PRESETS[PRESET].pumping_head_m,
        "tariff_usd_kwh": 0.045, "opex_nonenergy_usd_m3": 0.40, "capex_usd_per_m3d": 870.0,
        "capex_exponent": -0.09, "discount_rate": 0.10, "availability": 0.90}

def to_params(U):
    lo = np.array([RANGES[n][0] for n in NAMES]); hi = np.array([RANGES[n][1] for n in NAMES])
    X = lo + U * (hi - lo)
    return [dict(zip(NAMES, row)) for row in X]

def run(cap, N=16384, seed=1):
    rng = np.random.default_rng(seed)
    k = len(NAMES)
    A, B = rng.random((N, k)), rng.random((N, k))
    def ev(U): 
        r = np.array([model(p, cap) for p in to_params(U)]); return r[:, 0], r[:, 1]
    fA, gA = ev(A); fB, gB = ev(B)
    res = {"capacity_m3d": cap, "N": N, "seed": seed}
    for out_name, yA0, yB0, idx in (("lcow", fA, fB, 0), ("pond_ha", gA, gB, 1)):
        # centre the output: the Saltelli first-order estimator is unbiased but very noisy
        # when the output mean is large relative to its spread
        mu0 = np.mean(np.concatenate([yA0, yB0])); yA, yB = yA0 - mu0, yB0 - mu0
        V = np.var(np.concatenate([yA, yB]), ddof=1)
        S1, ST = {}, {}
        for j, n in enumerate(NAMES):
            ABj = A.copy(); ABj[:, j] = B[:, j]
            yAB = np.array([model(p, cap)[idx] for p in to_params(ABj)]) - mu0
            S1[n] = float(np.mean(yB * (yAB - yA)) / V)
            ST[n] = float(0.5 * np.mean((yA - yAB) ** 2) / V)
        y = np.concatenate([yA0, yB0])
        res[out_name] = {"mean": float(y.mean()), "sd": float(y.std(ddof=1)),
            "p05": float(np.percentile(y, 5)), "p50": float(np.percentile(y, 50)), "p95": float(np.percentile(y, 95)),
            "min": float(y.min()), "max": float(y.max()), "S1": S1, "ST": ST}
    # one-at-a-time swings from the base case
    base = model(BASE, cap); oat = {}
    for n in NAMES:
        vals = []
        for v in RANGES[n][:2]:
            q = dict(BASE); q[n] = v; vals.append(model(q, cap))
        oat[n] = {"lcow_low": vals[0][0], "lcow_high": vals[1][0], "pond_low": vals[0][1], "pond_high": vals[1][1]}
    res["base"] = {"lcow": base[0], "pond_ha": base[1]}; res["oat"] = oat
    return res

if __name__ == "__main__":
    out = {"ranges": RANGES, "base_params": BASE, "runs": [run(20000.0), run(150000.0)]}
    json.dump(out, open("monte_carlo_sensitivity.json", "w"), indent=1)
    for r in out["runs"]:
        print("\ncapacity", r["capacity_m3d"], "base", r["base"])
        for o in ("lcow", "pond_ha"):
            d = r[o]; print(o, {k: round(d[k], 3) for k in ("mean", "sd", "p05", "p50", "p95", "min", "max")})
            print("  S1", {k: round(v, 3) for k, v in d["S1"].items()}); print("  ST", {k: round(v, 3) for k, v in d["ST"].items()})
