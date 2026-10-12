"""(v1.2.0) Check of the framework against one measured Egyptian pilot plant: El Alamein, 1,000 m3/d brackish RO.

Measured values are those REPORTED by Dawoud, Sallam, Abdelrahman & Emam, Sustainability 16(4), 1630 (2024),
https://doi.org/10.3390/su16041630 . The paper is internally inconsistent in places (capital cost, well depth),
so every inconsistent quantity is kept as a range and the model is run at both ends. One plant, one study:
this is a plausibility check at 1,000 m3/d, not a validation of the framework and not at large scale.
Nothing in the model is re-fitted to these data.

Outputs: el_alamein_validation.json
"""
import json
from desalination_plant_design import (PlantCapacity, PlantFlows, ReverseOsmosisSystem,
                                       EconomicAnalysis, EconomicAssumptions)

# ---- Reported by the source paper (values to be re-checked against the paper before submission) ----
MEASURED = {
    "capacity_m3d": 1000.0,
    "feed_tds_mg_l": (21150.0, 22400.0),          # at start / during the study year
    "feed_temp_c": 32.0,
    "recovery": (0.46, 0.35),                      # at start / after one year
    "sec_kwh_m3": (3.68, 4.20),                    # reported range; mean about 3.94
    "capex_usd_per_m3d": (760.0, 931.0),           # abstract/conclusion 760-850; body 931
    "opex_usd_m3": (0.55, 0.63),                   # abstract (EGP 15-17 per m3 in conclusions)
    "tariff_usd_kwh": 0.039,                       # tariff assumed by the authors
    "well_depth_m": (35.0, 76.0),                  # two values in the paper
}

def run(tds, rec, head, tariff=MEASURED["tariff_usd_kwh"]):
    cap = MEASURED["capacity_m3d"]
    fl = PlantFlows.from_capacity(PlantCapacity(permeate_flow_m3d=cap, recovery=rec))
    ro = ReverseOsmosisSystem(flow_m3d=fl.feed_to_membranes_m3d, permeate_m3d=fl.permeate_m3d,
                              concentrate_m3d=fl.concentrate_m3d, feed_tds_mg_l=tds, water_type="brackish",
                              intake_m3d=fl.intake_m3d, pumping_head_m=head).design()
    lo, hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
    mid = (lo + hi) / 2
    a = EconomicAssumptions(electricity_tariff_usd_kwh=tariff)
    e = EconomicAnalysis(capacity_m3d=cap, water_type="brackish", specific_energy_kwh_m3=mid,
                         assumptions=a).run_full_analysis()
    return {"tds": tds, "recovery": rec, "head_m": head,
            "feed_pressure_bar": ro["feed_pressure_bar_est"], "conc_tds_mg_l": ro["concentrate_tds_mg_l_est"],
            "sec_total_lo": lo, "sec_total_hi": hi, "sec_mid": round(mid, 2),
            "sec_lift": ro["specific_energy_lift_kwh_m3_est"],
            "capex_usd_per_m3d": round(e["capex"]["capex_per_m3d_usd"], 0),
            "opex_usd_m3": round(e["opex"]["opex_total_usd_m3"], 3) if "opex_total_usd_m3" in e["opex"] else None,
            "lcow_usd_m3": e["lcow"]["lcow_usd_m3"]}

if __name__ == "__main__":
    runs = []
    for tds, rec in ((21150.0, 0.46), (22400.0, 0.35)):
        for head in (35.0, 76.0):
            runs.append(run(tds, rec, head))
    secs = [r["sec_mid"] for r in runs]
    sec_mean_meas = 3.94  # mean reported by the source paper
    cx = runs[0]["capex_usd_per_m3d"]
    out = {"measured_as_reported": MEASURED, "runs": runs,
           "model_sec_mid_range": [min(secs), max(secs)],
           "model_sec_band_all_runs": [min(r["sec_total_lo"] for r in runs), max(r["sec_total_hi"] for r in runs)],
           "measured_sec_range": list(MEASURED["sec_kwh_m3"]),
           "mean_of_model_sec_mid": round(sum(secs) / len(secs), 2),
           "mean_sec_error_pct_vs_3.94": round(100 * (sum(secs) / len(secs) / sec_mean_meas - 1), 1),
           "capex_error_pct_vs_760": round(100 * (cx / 760 - 1), 0),
           "capex_error_pct_vs_931": round(100 * (cx / 931 - 1), 0),
           "opex_model_range": [min(r["opex_usd_m3"] for r in runs), max(r["opex_usd_m3"] for r in runs)]}
    json.dump(out, open("el_alamein_validation.json", "w"), indent=1)
    for r in runs: print(r)
    print("model SEC mid range", min(secs), max(secs), "measured", MEASURED["sec_kwh_m3"])
