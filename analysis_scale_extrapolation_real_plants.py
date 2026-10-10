"""
analysis_scale_extrapolation_real_plants.py   (v1.2.0)

Out-of-sample scale test on REAL operating brackish-water RO plants.

Question: if the cost model is calibrated only on SMALL plants (<= 25,000 m3/d),
how well does it predict the LARGEST operating plants in the compiled dataset
(>= 40,000 m3/d: four Florida plants and Kay Bailey Hutchison, TX)?

This is the strongest scale test the open data allow. No operating brackish
plant above ~68,000 m3/d with published cost data was found, so the 100,000-600,000
m3/d range used in the AI experiments remains an extrapolation beyond real data.

Calibration (small plants only):
  * non-energy OPEX constant = median over small plants of
        OPEX_reported - SEC * tariff_reported
  * log-log OPEX(capacity) and CAPEX(capacity) power laws.
Prediction (large plants):
  * framework: c + SEC_framework * tariff   (Florida: tariff range 0.06-0.125)
  * baselines: small-plant median OPEX; small-plant OPEX power law extrapolated.
  * CAPEX: framework default benchmark, and the small-plant power law extrapolated.

Run:  python3 analysis_scale_extrapolation_real_plants.py
Writes analysis_scale_extrapolation.json
"""
import json
import numpy as np

from real_world_bwro_dataset import REAL_BWRO_PLANTS, REPORTED_TARIFF_USD_KWH
from real_data_calibration import sec_kwh_m3
from desalination_plant_design import EconomicAnalysis

SMALL_MAX, LARGE_MIN = 25_000, 40_000

small = [p for p in REAL_BWRO_PLANTS
         if p["capacity_m3d"] <= SMALL_MAX and p["name"] in REPORTED_TARIFF_USD_KWH
         and p["source"] == "pearson_tx"]
large = [p for p in REAL_BWRO_PLANTS
         if p["capacity_m3d"] >= LARGE_MIN and p["source"] in ("pearson_fl", "pearson_tx")]

# --- calibrate on small plants only ---------------------------------------
nonenergy = np.array([p["opex_usd_m3"] - sec_kwh_m3(p) * REPORTED_TARIFF_USD_KWH[p["name"]] for p in small])
c_med = float(np.median(nonenergy))
caps_s = np.array([p["capacity_m3d"] for p in small], float)
opex_s = np.array([p["opex_usd_m3"] for p in small], float)
n_op, lA = np.polyfit(np.log(caps_s), np.log(opex_s), 1)
cx = [p for p in small if p["capex_usd_m3d"]]
n_cx, lB = np.polyfit(np.log([p["capacity_m3d"] for p in cx]), np.log([p["capex_usd_m3d"] for p in cx]), 1)

rows = []
for p in large:
    act = p["opex_usd_m3"]
    sec = sec_kwh_m3(p)
    tariffs = [REPORTED_TARIFF_USD_KWH[p["name"]]] if p["name"] in REPORTED_TARIFF_USD_KWH else [0.06, 0.08, 0.10, 0.125]
    fw = [c_med + sec * t for t in tariffs]
    mid = fw[len(fw) // 2] if len(fw) > 1 else fw[0]       # 0.10 for FL, reported for TX
    mid = c_med + sec * (0.08 if len(tariffs) > 1 else tariffs[0])  # 0.08 USD/kWh for FL, as in real_data_calibration.py
    rows.append({
        "plant": p["name"], "capacity_m3d": p["capacity_m3d"], "reported_opex": act, "sec_framework": sec,
        "framework_opex": mid, "framework_opex_range": [min(fw), max(fw)],
        "naive_small_median": float(np.median(opex_s)),
        "powerlaw_extrapolated": float(np.exp(lA) * p["capacity_m3d"] ** n_op),
        "capex_reported": p["capex_usd_m3d"],
    })

def stats(key):
    e = np.array([r[key] - r["reported_opex"] for r in rows])
    a = np.array([r["reported_opex"] for r in rows])
    return {"mae": float(np.mean(np.abs(e))), "mape_pct": float(np.mean(np.abs(e) / a) * 100), "bias": float(np.mean(e))}

summary = {
    "n_small_train": len(small), "n_large_test": len(rows),
    "non_energy_opex_constant_usd_m3": c_med,
    "opex_powerlaw_exponent_small": float(n_op), "capex_powerlaw_exponent_small": float(n_cx),
    "opex_framework": stats("framework_opex"),
    "opex_naive_small_median": stats("naive_small_median"),
    "opex_powerlaw_extrapolated": stats("powerlaw_extrapolated"),
}

# --- CAPEX at the one large plant with CAPEX data (KBH) -------------------
kbh = next(p for p in large if "Hutchinson" in p["name"])
fw_capex = EconomicAnalysis(capacity_m3d=kbh["capacity_m3d"], water_type="brackish",
                            specific_energy_kwh_m3=sec_kwh_m3(kbh)).run_full_analysis()["capex"]["capex_per_m3d_usd"]
summary["capex_kbh"] = {
    "reported_usd_per_m3d": kbh["capex_usd_m3d"], "framework_usd_per_m3d": float(fw_capex),
    "framework_error_pct": float((fw_capex / kbh["capex_usd_m3d"] - 1) * 100),
    "small_plant_powerlaw_usd_per_m3d": float(np.exp(lB) * kbh["capacity_m3d"] ** n_cx),
    "small_plant_powerlaw_error_pct": float((np.exp(lB) * kbh["capacity_m3d"] ** n_cx / kbh["capex_usd_m3d"] - 1) * 100),
    "note": "single plant; capital cost in nominal USD of different years across sources; blended-capacity basis (Pearson et al. 2021)",
}

if __name__ == "__main__":
    print(f"Calibration on {len(small)} small plants (<= {SMALL_MAX:,} m3/d); test on {len(rows)} plants (>= {LARGE_MIN:,} m3/d)")
    print(f"non-energy OPEX constant = {c_med:.3f} USD/m3; OPEX exponent (small) = {n_op:.2f}; CAPEX exponent (small) = {n_cx:.2f}\n")
    for r in rows:
        print(f"{r['plant']:38s} {r['capacity_m3d']:>7,.0f}  reported {r['reported_opex']:.3f} | framework {r['framework_opex']:.3f} "
              f"({r['framework_opex_range'][0]:.3f}-{r['framework_opex_range'][1]:.3f}) | naive {r['naive_small_median']:.3f} | power-law {r['powerlaw_extrapolated']:.3f}")
    for k in ("opex_framework", "opex_naive_small_median", "opex_powerlaw_extrapolated"):
        s = summary[k]; print(f"{k:30s} MAE {s['mae']:.3f}  MAPE {s['mape_pct']:.0f}%  bias {s['bias']:+.3f}")
    print("\nCAPEX at Kay Bailey Hutchinson:", json.dumps(summary["capex_kbh"], indent=1))
    json.dump({"summary": summary, "rows": rows}, open("analysis_scale_extrapolation.json", "w"), indent=1)
    print("Saved analysis_scale_extrapolation.json")
