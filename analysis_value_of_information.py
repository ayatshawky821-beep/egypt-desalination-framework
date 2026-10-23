"""
analysis_value_of_information.py

Which missing data would most change the design decision?

Decision: the recovery r (0.50-0.90, step 0.01) of a 200,000 m3/d brackish plant with
evaporation-pond brine disposal. Cost of a decision = LCOW + annualised pond cost, per m3.
Uncertain inputs (independent uniform priors):
  capex_exp     capacity exponent of unit capital cost        -0.20 ... 0.00
  capex_mult    multiplier on the capital-cost benchmark       0.60 ... 1.40
  sec_model     weight on the lumped SEC (0 = element-by-element simulation, 1 = framework)  0 ... 1
  tariff        electricity tariff (USD/kWh)                   0.03 ... 0.10
  opex_ne       non-energy operating cost (USD/m3)             0.30 ... 0.50
  evap          net evaporation rate (m/yr)                    1.5 ... 3.0
  pond_cost     lined-pond capital cost (USD/m2)               5 ... 30
  discount      discount rate                                  0.08 ... 0.14
Outputs: distribution of the optimal recovery, first-order sensitivity index of the optimal
recovery, EVPI and per-parameter EVPPI (binning estimator), and a noise floor from a dummy input.

Run:  python3 analysis_value_of_information.py
"""
import json
import numpy as np
from desalination_plant_design import EGYPT_BRACKISH_PRESETS, EconomicAssumptions
from analysis_element_level_ro import array, calibrate
from external_validation_rosa_egypt import framework_sec

Q = 200000.0
R = np.round(np.arange(0.50, 0.9001, 0.01), 2)
N = 20000
BINS = 40
PRIORS = {"capex_exp": (-0.20, 0.0), "capex_mult": (0.6, 1.4), "sec_model": (0.0, 1.0), "tariff": (0.03, 0.10),
          "opex_ne": (0.30, 0.50), "evap": (1.5, 3.0), "pond_cost": (5.0, 30.0), "discount": (0.08, 0.14),
          "dummy": (0.0, 1.0)}
PRESETS = ["el_moghra_aquifer", "nile_delta_shallow", "western_desert_nubian", "sinai_coastal",
           "agricultural_drainage_nile_delta"]


def sec_curves(feed):
    A0, B0 = calibrate()
    el, ok = [], []
    for r in R:
        e = array(feed.tds_mg_l, float(r), feed.temp_c, A0, B0)
        el.append(e["sec_kwh_m3"]); ok.append(e["within_41bar"])
    fw = [framework_sec(feed.tds_mg_l, float(r), Q, "brackish") for r in R]
    return np.array(el), np.array(fw), np.array(ok)


def costs(theta, el, fw, ok, ponds=True):
    a = EconomicAssumptions()
    i, n = theta["discount"][:, None], a.plant_life_years
    crf = i * (1 + i) ** n / ((1 + i) ** n - 1)
    scale = np.clip((Q / a.reference_capacity_m3d) ** theta["capex_exp"][:, None], 0.5, 2.0)
    capex = a.capex_per_m3d_brackish * theta["capex_mult"][:, None] * scale          # USD per m3/d
    prod = 365 * a.availability_factor
    sec = (1 - theta["sec_model"][:, None]) * el[None, :] + theta["sec_model"][:, None] * fw[None, :]
    lcow = crf * capex / prod + sec * theta["tariff"][:, None] + theta["opex_ne"][:, None]
    area = Q * (1 - R[None, :]) / R[None, :] * 365 / theta["evap"][:, None]          # m2
    pond = crf * theta["pond_cost"][:, None] * area / (Q * prod)
    c = lcow + pond if ponds else lcow + 0 * pond
    c[:, ~ok] = np.inf
    return c


def evppi(c, x, d0):
    edges = np.quantile(x, np.linspace(0, 1, BINS + 1)); idx = np.clip(np.searchsorted(edges, x, "right") - 1, 0, BINS - 1)
    best = 0.0
    for b in range(BINS):
        m = idx == b
        best += m.mean() * np.min(c[m].mean(axis=0))
    return float(c[:, d0].mean() - best)


def first_order(y, x):
    edges = np.quantile(x, np.linspace(0, 1, BINS + 1)); idx = np.clip(np.searchsorted(edges, x, "right") - 1, 0, BINS - 1)
    means = np.array([y[idx == b].mean() for b in range(BINS)]); w = np.array([(idx == b).mean() for b in range(BINS)])
    return float(np.sum(w * (means - y.mean()) ** 2) / y.var())


def main():
    rng = np.random.default_rng(2026)
    theta = {k: rng.uniform(lo, hi, N) for k, (lo, hi) in PRIORS.items()}
    year = Q * 365 * EconomicAssumptions().availability_factor
    out = {"priors": PRIORS, "N": N, "bins": BINS, "capacity": Q, "results": {}}
    for key in PRESETS:
        feed = EGYPT_BRACKISH_PRESETS[key]
        el, fw, ok = sec_curves(feed)
        res = {}
        for label, ponds in (("lcow_only", False), ("lcow_plus_ponds", True)):
            c = costs(theta, el, fw, ok, ponds)
            d0 = int(np.argmin(c.mean(axis=0)))
            rstar = R[np.argmin(c, axis=1)]
            cmin = c.min(axis=1)
            evpi = float(c[:, d0].mean() - cmin.mean())
            r = {"r_fixed_best": float(R[d0]),
                 "rstar_quantiles": [float(v) for v in np.quantile(rstar, [0.05, 0.25, 0.5, 0.75, 0.95])],
                 "frac_rstar_at_upper": float((rstar >= R[ok].max() - 1e-9).mean()),
                 "rstar_hist": np.histogram(rstar, bins=np.arange(0.495, 0.906, 0.01))[0].tolist(),
                 "evpi_usd_m3": evpi, "evpi_musd_yr": evpi * year / 1e6,
                 "cost_quantiles": [float(v) for v in np.quantile(cmin, [0.05, 0.5, 0.95])],
                 "evppi_usd_m3": {}, "evppi_musd_yr": {}, "S1_rstar": {}, "S1_cost": {}}
            for k in PRIORS:
                v = evppi(c, theta[k], d0)
                r["evppi_usd_m3"][k] = v; r["evppi_musd_yr"][k] = v * year / 1e6
                r["S1_rstar"][k] = first_order(rstar, theta[k]) if rstar.var() > 0 else 0.0
                r["S1_cost"][k] = first_order(cmin, theta[k])
            res[label] = r
            print(f"\n{key} [{label}]: best fixed r {R[d0]:.2f}; r* 5-95% {r['rstar_quantiles'][0]:.2f}-{r['rstar_quantiles'][4]:.2f}"
                  f" (median {r['rstar_quantiles'][2]:.2f}); at upper bound {r['frac_rstar_at_upper']:.2f}; EVPI {evpi*1000:.2f} mUSD/m3"
                  f" = {r['evpi_musd_yr']:.2f} MUSD/yr; cost 5/50/95% {r['cost_quantiles']}")
            for k in sorted(PRIORS, key=lambda k: -r["S1_cost"][k]):
                print(f"  {k:<11} EVPPI {r['evppi_musd_yr'][k]:6.3f} MUSD/yr  S1(r*) {r['S1_rstar'][k]:.2f}  S1(cost) {r['S1_cost'][k]:.2f}")
        mid = {k: np.full(1, (v[0] + v[1]) / 2) for k, v in PRIORS.items()}
        for lab, w in (("lumped", 1.0), ("element", 0.0)):
            th = dict(mid); th["sec_model"] = np.full(1, w)
            r_lc = float(R[np.argmin(costs(th, el, fw, ok, False)[0])])
            area_ha = Q * (1 - r_lc) / r_lc * 365 / mid["evap"][0] / 1e4
            res[f"lcow_only_r_{lab}"] = r_lc; res[f"lcow_only_pond_ha_{lab}"] = float(area_ha)
        th = dict(mid); c_mid = costs(th, el, fw, ok, True)[0]; k_best = int(np.argmin(c_mid))
        c_np = costs(th, el, fw, ok, False)[0]
        res["pond_share_at_best_mid"] = float((c_mid[k_best] - c_np[k_best]) / c_mid[k_best])
        print(f"  mid priors, LCOW only: r* lumped {res['lcow_only_r_lumped']:.2f} ({res['lcow_only_pond_ha_lumped']:.0f} ha), "
              f"element {res['lcow_only_r_element']:.2f} ({res['lcow_only_pond_ha_element']:.0f} ha); pond share of cost at best design {res['pond_share_at_best_mid']:.2f}")
        out["results"][key] = res
    json.dump(out, open("analysis_value_of_information.json", "w"), indent=1)


if __name__ == "__main__":
    main()
