import json, warnings, numpy as np
warnings.filterwarnings("ignore")
import openpyxl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from desalination_plant_design import (FeedWaterQuality, PlantCapacity, PlantFlows, ReverseOsmosisSystem,
    EconomicAnalysis, EGYPT_BRACKISH_PRESETS, default_recovery_for)
from external_validation_rosa_egypt import framework_sec, load_rosa_data
out = {}

# ---- 1. extrapolation experiment -------------------------------------------------
import egypt_ml_surrogate as small
import ai_deep_surrogate_large_scale as large
Xs, Ys = small.build_dataset()
m_small = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42).fit(Xs, Ys[:, 0])
Xl, yl = large.build_dataset()
Xtr, Xte, ytr, yte = train_test_split(Xl, yl, test_size=0.2, random_state=7)
m_large = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=7).fit(Xtr, ytr)
caps = np.linspace(100000, 600000, 51)
def phys(cap):
    c = PlantCapacity(permeate_flow_m3d=cap, recovery=0.80); f = PlantFlows.from_capacity(c)
    ro = ReverseOsmosisSystem(flow_m3d=f.feed_to_membranes_m3d, permeate_m3d=f.permeate_m3d, concentrate_m3d=f.concentrate_m3d,
                              feed_tds_mg_l=5000, water_type="brackish").design()
    lo, hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
    return EconomicAnalysis(capacity_m3d=cap, water_type="brackish", specific_energy_kwh_m3=(lo+hi)/2).run_full_analysis()["lcow"]["lcow_usd_m3"]
P = np.array([phys(c) for c in caps])
Xq = np.column_stack([np.full(len(caps), 5000.0), np.full(len(caps), 0.80), caps, np.zeros(len(caps))])
ps, pl = m_small.predict(Xq), m_large.predict(Xq)
es, el = (ps-P)/P*100, (pl-P)/P*100
out["extrap"] = {"caps": caps.tolist(), "physics": P.tolist(), "small": ps.tolist(), "large": pl.tolist()}
for c in (150000, 300000, 450000, 600000):
    i = int(np.argmin(abs(caps-c)))
    print(f"cap {c:>7}: physics {P[i]:.4f} | small-range surrogate {ps[i]:.4f} ({es[i]:+.1f}%) | large surrogate {pl[i]:.4f} ({el[i]:+.1f}%)")
beyond = caps > 150000
print("max |err| beyond 150k: small %.1f%%, large %.1f%%" % (abs(es[beyond]).max(), abs(el[beyond]).max()))
print("mean |err| 100-150k: small %.2f%%" % abs(es[~beyond]).mean())

# ---- 2. SEC comparisons ------------------------------------------------------------
sal, en = load_rosa_data("rosa_et_al_2025_source_data.xlsx")
print("\nRosa overall slope (kWh/m3 per ppt): %.4f" % np.polyfit(sal, en, 1)[0])
m = (sal >= 2) & (sal <= 10); print("Rosa slope 2-10 ppt: %.4f" % np.polyfit(sal[m], en[m], 1)[0])
tds = np.arange(2000, 10001, 500)
for rec in (0.75, 0.80, 0.85):
    fs = np.array([framework_sec(t, rec, 150000, "brackish") for t in tds])
    sl, ic = np.polyfit(tds/1000, fs, 1)
    r2 = np.corrcoef(tds/1000, fs)[0,1]**2
    print(f"framework slope 2-10 ppt at recovery {rec}: {sl:.4f} kWh/m3/ppt (R2 {r2:.3f})")
print("Stillwell & Webber printed partial slope for raw-water TDS: 8.3e-5 per mg/L = 0.083 per ppt")
rows = []
print("\npreset | ppt | recovery | Rosa | framework | diff%")
for key, feed in EGYPT_BRACKISH_PRESETS.items():
    ppt = feed.tds_mg_l/1000; rec = default_recovery_for(feed)
    fw = framework_sec(feed.tds_mg_l, rec, 150000, feed.water_type)
    inrange = sal.min() <= ppt <= sal.max()
    if inrange: ro_ = float(np.interp(ppt, sal, en))
    else:
        s = (en[1]-en[0])/(sal[1]-sal[0]); ro_ = float(en[0] - s*(sal[0]-ppt))
    rows.append({"preset": key, "ppt": ppt, "rec": rec, "rosa": ro_, "fw": fw, "diff": (fw-ro_)/ro_*100, "inrange": bool(inrange)})
    print(f"{key:<34}{ppt:6.2f} {rec:5.2f} {ro_:6.2f} {fw:6.2f} {(fw-ro_)/ro_*100:+6.1f}% {'' if inrange else '(extrap)'}")
out["presets"] = rows

# ---- 3. the 300,000 m3/d Nile Delta design study -------------------------------------
tds_p, cap_p, meas = 2705.0, 300000.0, 1.49
for rec in (0.75, 0.85):
    fw = framework_sec(tds_p, rec, cap_p, "brackish")
    print(f"\ndesign study: framework SEC at recovery {rec}: {fw:.3f} vs design-study {meas} ({(fw-meas)/meas*100:+.0f}%)")
rosa_p = float(np.interp(tds_p/1000, sal, en)); print(f"Rosa at 2.705 ppt: {rosa_p:.3f} ({(rosa_p-meas)/meas*100:+.0f}% vs design-study)")
fw75 = framework_sec(tds_p, 0.75, cap_p, "brackish")
e_meas = EconomicAnalysis(capacity_m3d=cap_p, water_type="brackish", specific_energy_kwh_m3=meas).run_full_analysis()
e_fw = EconomicAnalysis(capacity_m3d=cap_p, water_type="brackish", specific_energy_kwh_m3=fw75).run_full_analysis()
print("OPEX with design-study SEC: %.3f  | OPEX with framework's own SEC: %.3f | reported 0.487" % (e_meas["opex"]["opex_total_usd_m3"], e_fw["opex"]["opex_total_usd_m3"]))
print("CAPEX per m3/d: framework %.1f vs reported 396" % e_meas["capex"]["capex_per_m3d_usd"])
out["plant"] = {"fw_sec_075": fw75, "rosa": rosa_p, "meas": meas, "opex_meas": e_meas["opex"]["opex_total_usd_m3"],
                "opex_fw": e_fw["opex"]["opex_total_usd_m3"], "capex": e_meas["capex"]["capex_per_m3d_usd"]}
json.dump(out, open("analysis_a.json", "w"))
print("saved")
