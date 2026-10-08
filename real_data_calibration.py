"""
real_data_calibration.py
 
Calibrates and validates the framework's cost model against the compiled
real-world dataset (real_world_bwro_dataset.py; 18 plants).
 
DESIGN (chosen to avoid over-claiming on a small dataset):
  * TRAIN set (n = 11): plants whose source REPORTS its own electricity
    tariff (9 Texas plants + 2 Egyptian plants). For each, the energy cost
    is computed with its own tariff and specific energy, and the remainder
    of reported OPEX is the "non-energy OPEX" (chemicals, labor,
    maintenance, other). A single robust constant (the median) is fitted.
  * TEST set (n = 7): the Florida plants. They report no plant-specific
    tariff, so they are held out; predictions are made across a range of
    plausible tariffs (0.06-0.125 USD/kWh) and compared with reported OPEX.
  * Leave-one-out cross-validation (LOOCV) is reported for the fitted
    constant. Economies-of-scale exponents are re-fitted separately for
    OPEX and CAPEX by log-log regression, with R^2 reported honestly.
 
Run:  python3 real_data_calibration.py
Writes curve12_real_data_calibration.png and prints all statistics.
"""
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
 
from real_world_bwro_dataset import (
    REAL_BWRO_PLANTS, REPORTED_TARIFF_USD_KWH, MEASURED_SEC_KWH_M3,
    DEFAULT_RECOVERY, REPORTED_RECOVERY,
)
from desalination_plant_design import (
    PlantCapacity, PlantFlows, ReverseOsmosisSystem,
)
 
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
 
 
def sec_kwh_m3(plant) -> float:
    """Specific energy: measured value if the source reports one, otherwise
    the framework's own osmotic-pressure-based estimate (Eq. 11-16)."""
    if plant["name"] in MEASURED_SEC_KWH_M3:
        return MEASURED_SEC_KWH_M3[plant["name"]]
    recovery = REPORTED_RECOVERY.get(plant["name"], DEFAULT_RECOVERY)
    cap = PlantCapacity(permeate_flow_m3d=plant["capacity_m3d"], recovery=recovery)
    flows = PlantFlows.from_capacity(cap)
    ro = ReverseOsmosisSystem(
        flow_m3d=flows.feed_to_membranes_m3d, permeate_m3d=flows.permeate_m3d,
        concentrate_m3d=flows.concentrate_m3d, feed_tds_mg_l=plant["tds_mg_l"],
        water_type="brackish",
    ).design()
    lo, hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
    return (lo + hi) / 2
 
 
def split_train_test():
    train = [p for p in REAL_BWRO_PLANTS if p["name"] in REPORTED_TARIFF_USD_KWH]
    test = [p for p in REAL_BWRO_PLANTS if p["name"] not in REPORTED_TARIFF_USD_KWH]
    return train, test
 
 
def calibrate_nonenergy_opex():
    train, test = split_train_test()
    nonenergy = np.array([
        p["opex_usd_m3"] - sec_kwh_m3(p) * REPORTED_TARIFF_USD_KWH[p["name"]] for p in train
    ])
    c_median, c_mean, c_sd = float(np.median(nonenergy)), float(nonenergy.mean()), float(nonenergy.std(ddof=1))
    loocv_mae = float(np.mean([abs(np.median(np.delete(nonenergy, i)) - nonenergy[i])
                                for i in range(len(nonenergy))]))
 
    # Held-out Florida test across a plausible tariff range
    test_rows = {}
    act = np.array([p["opex_usd_m3"] for p in test])
    for tariff in (0.06, 0.08, 0.10, 0.125):
        pred = np.array([c_median + sec_kwh_m3(p) * tariff for p in test])
        test_rows[tariff] = {
            "mae": float(np.mean(np.abs(pred - act))),
            "mape": float(np.mean(np.abs(pred - act) / act) * 100),
            "bias": float(np.mean(pred - act)),
            "pred": pred,
        }
    # Baseline: the previous uncalibrated benchmark (0.12 USD/m3 non-energy)
    old_pred = np.array([0.12 + sec_kwh_m3(p) * 0.08 for p in test])
    old = {"mae": float(np.mean(np.abs(old_pred - act))),
           "mape": float(np.mean(np.abs(old_pred - act) / act) * 100),
           "bias": float(np.mean(old_pred - act))}
    return train, test, nonenergy, c_median, c_mean, c_sd, loocv_mae, test_rows, old, act
 
 
def fit_exponents():
    """Log-log fits. NOTE: regressing TOTAL cost/day on capacity gives an
    inflated R^2 by construction (capacity appears on both sides), so only
    the unit-cost fit is used for inference."""
    caps = np.array([p["capacity_m3d"] for p in REAL_BWRO_PLANTS], float)
    opex = np.array([p["opex_usd_m3"] for p in REAL_BWRO_PLANTS], float)
    n_op, lA = np.polyfit(np.log(caps), np.log(opex), 1)
    r2_op = r2_score(np.log(opex), lA + n_op * np.log(caps))
 
    cx = [p for p in REAL_BWRO_PLANTS if p["capex_usd_m3d"]]
    ccap = np.array([p["capacity_m3d"] for p in cx], float)
    cval = np.array([p["capex_usd_m3d"] for p in cx], float)
    n_cx, lB = np.polyfit(np.log(ccap), np.log(cval), 1)
    r2_cx = r2_score(np.log(cval), lB + n_cx * np.log(ccap))
    capex_at_20k = float(np.exp(lB) * 20000 ** n_cx)
    return dict(n_opex=n_op, r2_opex=r2_op, n_capex=n_cx, r2_capex=r2_cx,
                capex_at_20k=capex_at_20k, ccap=ccap, cval=cval, lB=lB)
 
 
def real_data_ml_loocv():
    """Can an ML model trained ONLY on the real plants predict OPEX better
    than a naive constant? Leave-one-out cross-validation of (a) a naive
    median predictor, (b) ridge regression on log(capacity) and log(TDS),
    and (c) a Random Forest on the same two features. With n = 18 and
    heterogeneous plants this is expected to show no skill; the experiment
    documents that quantitatively rather than assuming it."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneOut
 
    X = np.array([[np.log(p["capacity_m3d"]), np.log(p["tds_mg_l"])] for p in REAL_BWRO_PLANTS])
    y = np.array([p["opex_usd_m3"] for p in REAL_BWRO_PLANTS])
    preds = {"naive median": [], "ridge (log cap, log TDS)": [], "random forest": []}
    for tr, te in LeaveOneOut().split(X):
        preds["naive median"].append(np.median(y[tr]))
        preds["ridge (log cap, log TDS)"].append(Ridge(alpha=1.0).fit(X[tr], y[tr]).predict(X[te])[0])
        preds["random forest"].append(
            RandomForestRegressor(n_estimators=200, random_state=0).fit(X[tr], y[tr]).predict(X[te])[0])
    out = {}
    for k, v in preds.items():
        v = np.array(v)
        out[k] = {"mae": float(np.mean(np.abs(v - y))),
                  "mape": float(np.mean(np.abs(v - y) / y) * 100),
                  "r2": float(r2_score(y, v))}
    return out
 
 
def egyptian_plant_predictions():
    """Framework (recalibrated) predictions for the two Egyptian cases (one operating plant, one design study)."""
    from desalination_plant_design import EconomicAnalysis
    rows = {}
    for p in REAL_BWRO_PLANTS:
        if p["source"] not in ("dawoud_elalamein", "elsayed_adw"):
            continue
        sec = sec_kwh_m3(p)
        e = EconomicAnalysis(capacity_m3d=p["capacity_m3d"], water_type="brackish",
                             specific_energy_kwh_m3=sec).run_full_analysis()
        rows[p["name"]] = {"sec_used": sec, "capex": e["capex"]["capex_per_m3d_usd"],
                           "opex": e["opex"]["opex_total_usd_m3"], "lcow": e["lcow"]["lcow_usd_m3"]}
    return rows
 
 
def make_plot(train, test, nonenergy, c_median, test_rows, act, fits):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
 
    caps_tr = [p["capacity_m3d"] for p in train]
    ax[0].scatter(caps_tr, nonenergy, s=45, color="tab:blue", label="Non-energy OPEX (train plants)")
    ax[0].axhline(c_median, color="r", ls="--", label=f"Fitted constant = \\${c_median:.2f}/m³")
    ax[0].axhline(0.12, color="g", ls=":", label="Earlier uncalibrated benchmark = \\$0.12/m³")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Plant capacity (m³/day, log)")
    ax[0].set_ylabel("Non-energy OPEX (USD/m³)")
    ax[0].set_title("Non-energy OPEX shows no capacity trend\n(train set, n = 11)")
    ax[0].legend(fontsize=7.5)
 
    r = test_rows[0.08]
    ax[1].scatter(act, r["pred"], s=45, color="tab:purple")
    lo = np.array([test_rows[0.06]["pred"], test_rows[0.125]["pred"]]).min(axis=0)
    hi = np.array([test_rows[0.06]["pred"], test_rows[0.125]["pred"]]).max(axis=0)
    ax[1].errorbar(act, r["pred"], yerr=[r["pred"] - lo, hi - r["pred"]], fmt="none", ecolor="gray", alpha=0.7)
    lims = [0.3, 0.75]
    ax[1].plot(lims, lims, "k--", alpha=0.5, label="Perfect prediction")
    ax[1].set_xlim(lims); ax[1].set_ylim(lims)
    ax[1].set_xlabel("Reported OPEX (USD/m³)")
    ax[1].set_ylabel("Calibrated-model OPEX (USD/m³)")
    ax[1].set_title(f"Held-out Florida plants (n = 7)\nMAE \\${r['mae']:.2f}/m³, MAPE {r['mape']:.0f}% @ \\$0.08/kWh\n(bars: tariff \\$0.06–0.125/kWh)")
    ax[1].legend(fontsize=8)
 
    ax[2].scatter(fits["ccap"], fits["cval"], s=45, color="tab:orange", label="Real plants with CAPEX (n = 11)")
    xr = np.logspace(np.log10(fits["ccap"].min()), np.log10(fits["ccap"].max()), 100)
    ax[2].plot(xr, np.exp(fits["lB"]) * xr ** fits["n_capex"], "r-",
               label=f"Fit: n = {fits['n_capex']:.2f} (R² = {fits['r2_capex']:.2f})")
    ax[2].plot(xr, 550 * (xr / 20000) ** -0.30, "g--", label="Earlier assumption: \\$550 @ 20k, n = −0.30")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("Plant capacity (m³/day, log)")
    ax[2].set_ylabel("CAPEX (USD per m³/day)")
    ax[2].set_title("CAPEX vs. capacity (nominal USD)")
    ax[2].legend(fontsize=7.5)
 
    fig.tight_layout()
    fig.savefig("curve12_real_data_calibration.png")
    plt.close(fig)
    print("Saved curve12_real_data_calibration.png")
 
 
if __name__ == "__main__":
    (train, test, nonenergy, c_median, c_mean, c_sd, loocv_mae,
     test_rows, old, act) = calibrate_nonenergy_opex()
    fits = fit_exponents()
 
    print(f"TRAIN n={len(train)} (plants reporting a tariff), TEST n={len(test)} (Florida, held out)")
    print(f"Non-energy OPEX: median = {c_median:.3f}, mean = {c_mean:.3f}, sd = {c_sd:.3f} USD/m3")
    print(f"LOOCV MAE of the fitted constant: {loocv_mae:.3f} USD/m3")
    print("\nHeld-out Florida test (calibrated constant + framework SEC x assumed tariff):")
    for t, r in test_rows.items():
        print(f"  tariff {t:.3f}: MAE = {r['mae']:.3f}  MAPE = {r['mape']:.0f}%  bias = {r['bias']:+.3f}")
    print(f"Baseline (earlier $0.12 non-energy benchmark @0.08): MAE = {old['mae']:.3f}  "
          f"MAPE = {old['mape']:.0f}%  bias = {old['bias']:+.3f}")
    print(f"\nOPEX unit-cost scaling: n = {fits['n_opex']:.3f}  (R2 = {fits['r2_opex']:.3f})")
    print(f"CAPEX unit-cost scaling: n = {fits['n_capex']:.3f}  (R2 = {fits['r2_capex']:.3f}), "
          f"fitted value at 20,000 m3/d = ${fits['capex_at_20k']:.0f}/m3/day")
    make_plot(train, test, nonenergy, c_median, test_rows, act, fits)
 
    print("\nLeave-one-out ML on the 18 cases (target: OPEX USD/m3):")
    for k, v in real_data_ml_loocv().items():
        print(f"  {k:26s} MAE = {v['mae']:.3f}  MAPE = {v['mape']:.0f}%  R2 = {v['r2']:+.3f}")
    print("\nRecalibrated framework vs. the two Egyptian plants:")
    for k, v in egyptian_plant_predictions().items():
        print(f"  {k[:45]:45s} SEC {v['sec_used']:.2f}  CAPEX {v['capex']:.0f}  OPEX {v['opex']:.3f}  LCOW {v['lcow']:.3f}")
 