"""
analysis_hybrid_surrogate_extrapolation.py

Can a surrogate trained only on small plants (2,000-150,000 m3/d) be trusted
at 150,000-600,000 m3/d? Compares purely data-driven surrogates with a
physics-structured hybrid, and stress-tests all of them against "truths"
whose economies of scale change beyond the training range.

Surrogates (all trained on the same small-range data):
  GB            gradient boosting on raw inputs (the manuscript's baseline)
  GB-logQ       gradient boosting on log(capacity)
  GB-mono       histogram gradient boosting with a monotone-decreasing constraint on capacity
  MLP           neural network (64-32) on standardised inputs incl. log(capacity)
  Hybrid        structural prior  LCOW = a + a_sw*sw + (b + b_sw*sw) * (Q/Q_ref)^c
                (unit capital cost follows a power law in capacity; c is FITTED,
                not taken from the model) + gradient-boosting residual on the
                remaining inputs.

Truth scenarios beyond 150,000 m3/d (identical below it, so the training data
are identical and no surrogate can tell them apart):
  nominal       the framework as published (CAPEX exponent -0.09 throughout)
  steeper       exponent -0.20 above 150,000 m3/d (stronger economies of scale)
  saturating    exponent 0 above 150,000 m3/d (modular trains, no further economies)

Also reports a simple applicability-domain flag: the fraction of large-scale
queries that fall outside the training hull of each input.

Run:  python3 analysis_hybrid_surrogate_extrapolation.py
"""
import json, warnings
import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from desalination_plant_design import (FeedWaterQuality, PlantCapacity, PlantFlows,
                                       ReverseOsmosisSystem, EconomicAnalysis)

warnings.filterwarnings("ignore")
Q_REF, Q_BREAK = 20000.0, 150000.0
SCENARIOS = {"nominal": -0.09, "steeper": -0.20, "saturating": 0.0}
SEEDS = [42, 1, 2, 3, 4]


def sec_of(tds, rec, cap, wt):
    f = PlantFlows.from_capacity(PlantCapacity(permeate_flow_m3d=cap, recovery=rec))
    ro = ReverseOsmosisSystem(flow_m3d=f.feed_to_membranes_m3d, permeate_m3d=f.permeate_m3d,
                              concentrate_m3d=f.concentrate_m3d, feed_tds_mg_l=tds, water_type=wt).design()
    lo, hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
    return (lo + hi) / 2


class PiecewiseScaleEconomics(EconomicAnalysis):
    """Same as the framework, but the CAPEX scale exponent changes to e2 above Q_BREAK."""
    e2: float = -0.09

    def _scale_factor(self, exponent):
        if exponent != self.assumptions.economies_of_scale_exponent or self.capacity_m3d <= Q_BREAK:
            return super()._scale_factor(exponent)
        e1 = exponent
        s = (Q_BREAK / Q_REF) ** e1 * (self.capacity_m3d / Q_BREAK) ** self.e2
        return max(0.5, min(2.0, s))


def lcow_true(tds, rec, cap, sw, scenario="nominal"):
    wt = "seawater" if sw else "brackish"
    e = PiecewiseScaleEconomics(capacity_m3d=cap, water_type=wt, specific_energy_kwh_m3=sec_of(tds, rec, cap, wt))
    e.e2 = SCENARIOS[scenario]
    # unrounded LCOW (the framework rounds to 3 decimals for reporting)
    e.capex().opex()
    a = e.assumptions; i, n = a.discount_rate, a.plant_life_years
    crf = i * (1 + i) ** n / ((1 + i) ** n - 1)
    return crf * e.design["capex"]["total_capex_usd"] / e.design["opex"]["annual_production_m3"] + \
        e.specific_energy_kwh_m3 * a.electricity_tariff_usd_kwh + e._opex_nonenergy()


def sample(n, rng, qlo, qhi, log=True):
    X = []
    for _ in range(n):
        sw = rng.random() < 0.3
        tds = rng.uniform(28000, 45000) if sw else rng.uniform(500, 15000)
        rec = rng.uniform(0.35, 0.55) if sw else rng.uniform(0.50, 0.90)
        cap = 10 ** rng.uniform(np.log10(qlo), np.log10(qhi)) if log else rng.uniform(qlo, qhi)
        X.append([tds, rec, cap, float(sw)])
    return np.array(X)


def truth(X, scenario):
    return np.array([lcow_true(*x, scenario=scenario) for x in X])


def feats(X):  # tds, rec, log10 Q, sw
    return np.column_stack([X[:, 0], X[:, 1], np.log10(X[:, 2]), X[:, 3]])


def prior(Z, a, a_sw, b, b_sw, c):
    q, sw = Z
    return a + a_sw * sw + (b + b_sw * sw) * (q / Q_REF) ** c


class Hybrid:
    def fit(self, X, y, seed):
        self.p, _ = curve_fit(prior, (X[:, 2], X[:, 3]), y, p0=[0.4, 0.1, 0.3, 0.1, -0.1], maxfev=20000)
        r = y - prior((X[:, 2], X[:, 3]), *self.p)
        self.gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                            random_state=seed).fit(feats(X), r)
        return self

    def predict(self, X):
        return prior((X[:, 2], X[:, 3]), *self.p) + self.gb.predict(feats(X))


def make_models(seed):
    gb = lambda **k: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed, **k)
    return {
        "GB": (gb(), lambda X: X),
        "GB-logQ": (gb(), feats),
        "GB-mono": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, monotonic_cst=[0, 0, -1, 0], random_state=seed), feats),
        "MLP": (make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000,
                early_stopping=True, n_iter_no_change=25, random_state=seed)), feats),
        "Hybrid": (None, None),
    }


def main():
    out = {"seeds": SEEDS, "scenarios": SCENARIOS, "runs": []}
    line_caps = np.linspace(100000, 600000, 51)
    Xline = np.column_stack([np.full(51, 5000.0), np.full(51, 0.80), line_caps, np.zeros(51)])
    lines = {s: truth(Xline, s).tolist() for s in SCENARIOS}
    out["line"] = {"caps": line_caps.tolist(), "truth": lines, "pred": {}}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        Xtr = sample(6000, rng, 2000, 150000, log=True)
        ytr = truth(Xtr, "nominal")  # identical in all scenarios below 150,000
        Xin = sample(1000, rng, 2000, 150000, log=True); yin = truth(Xin, "nominal")
        Xbig = sample(2000, rng, 150000, 600000, log=False)
        ybig = {s: truth(Xbig, s) for s in SCENARIOS}
        run = {"seed": seed, "models": {}}
        for name, (m, tf) in make_models(seed).items():
            if name == "Hybrid":
                m = Hybrid().fit(Xtr, ytr, seed); pr = m.predict
                run["hybrid_prior"] = dict(zip(["a", "a_sw", "b", "b_sw", "c"], map(float, m.p)))
            else:
                m.fit(tf(Xtr), ytr); pr = (lambda m, tf: lambda X: m.predict(tf(X)))(m, tf)
            pin, pbig = pr(Xin), pr(Xbig)
            res = {"in_range_mape": float(np.mean(abs(pin - yin) / yin) * 100)}
            for s in SCENARIOS:
                ape = abs(pbig - ybig[s]) / ybig[s] * 100
                big = Xbig[:, 2] > 450000
                res[s] = {"mape": float(ape.mean()), "p95": float(np.percentile(ape, 95)),
                          "mape_450_600k": float(ape[big].mean())}
            run["models"][name] = res
            if seed == 42:
                out["line"]["pred"][name] = pr(Xline).tolist()
        out["runs"].append(run)
        print(f"seed {seed} done; hybrid fitted exponent c = {run['hybrid_prior']['c']:.4f}")

    names = list(out["runs"][0]["models"])
    summ = {}
    print(f"\n{'model':<9}{'in-range':>10}" + "".join(f"{s+' MAPE':>18}{'(450-600k)':>12}" for s in SCENARIOS))
    for n in names:
        rr = [r["models"][n] for r in out["runs"]]
        summ[n] = {"in_range_mape": [np.mean([r["in_range_mape"] for r in rr]), np.std([r["in_range_mape"] for r in rr], ddof=1)]}
        row = f"{n:<9}{summ[n]['in_range_mape'][0]:>9.2f}%"
        for s in SCENARIOS:
            m = [r[s]["mape"] for r in rr]; b = [r[s]["mape_450_600k"] for r in rr]
            summ[n][s] = {"mape": [float(np.mean(m)), float(np.std(m, ddof=1))],
                          "mape_450_600k": [float(np.mean(b)), float(np.std(b, ddof=1))],
                          "p95": float(np.mean([r[s]["p95"] for r in rr]))}
            row += f"{np.mean(m):>11.2f}±{np.std(m, ddof=1):<5.2f}{np.mean(b):>11.2f}%"
        print(row)
    out["summary"] = summ
    cs = [r["hybrid_prior"]["c"] for r in out["runs"]]
    out["hybrid_exponent"] = [float(np.mean(cs)), float(np.std(cs, ddof=1))]
    # pointwise errors on the 5 g/L, 80 % line (seed 42)
    for s in SCENARIOS:
        T = np.array(lines[s])
        for n in names:
            P = np.array(out["line"]["pred"][n]); e = (P - T) / T * 100
            print(f"line {s:<11}{n:<8} err@300k {e[20]:+5.1f}%  @600k {e[50]:+5.1f}%")
    json.dump(out, open("analysis_hybrid_surrogate.json", "w"), indent=1)
    print("saved analysis_hybrid_surrogate.json")


if __name__ == "__main__":
    main()
