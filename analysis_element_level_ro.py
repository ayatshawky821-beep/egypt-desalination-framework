"""
analysis_element_level_ro.py

Independent cross-check of the framework's lumped RO energy model against an
element-by-element spiral-wound RO simulation (the calculation that membrane
vendors' design programs perform), for the Egyptian brackish presets.

Element model (one 8-inch, 37 m2 element, solved in series along each vessel):
  water flux      Jw = A * TCF * FF * (P_avg - P_perm - (beta*pi(C_avg) - pi(C_p)))
  salt flux       Js = B * TCF * (beta*C_avg - C_p),   C_p = Js / Jw
  polarisation    beta = exp(0.7 * r_el)              (r_el = element recovery)
  pressure drop   dP(psi) = 0.01 q_avg(gpm)^1.7      (FilmTec design equations, Dow form 609-02057)
  osmotic         pi = 0.0008 bar per mg/L TDS        (same coefficient as the framework, so
                                                       differences come from process structure only)
  temperature     TCF = exp(2640 (1/298.15 - 1/T))
beta and TCF are the FilmTec design equations for 8-inch elements.
A and B are back-calculated from the FilmTec BW30-400 datasheet test point
(40 m3/d permeate, 99.5 % stabilised rejection, 2,000 mg/L NaCl, 15.5 bar,
25 C, 15 % recovery). Fouling factor FF = 0.85 (typical well-water design).

Array: 7 elements per vessel, average flux 22 LMH (as the framework),
for each recovery, the lowest-energy array among one stage (<= 55 %), two stages 2:1
(50-85 %) and three stages 4:2:1 (>= 70 %), preferring designs within the 41-bar rating;
interstage booster pumps where needed to give each stage its share of the
flux (as in vendor designs), no energy recovery, pump efficiency 0.80 (as the
framework). Feed pressure is solved so that the array meets the target
recovery. Results are intensive (independent of plant capacity).

Run:  python3 analysis_element_level_ro.py
"""
import json, math
import numpy as np
from scipy.optimize import brentq
from desalination_plant_design import EGYPT_BRACKISH_PRESETS
from external_validation_rosa_egypt import framework_sec

AREA, N_EL, FLUX, ETA = 37.0, 7, 22.0, 0.80
OSM = 0.0008            # bar per mg/L
QMAX, DPMAX = 19.0, 1.0  # m3/h, bar (datasheet)
FF = 0.85


def dp(q_avg):
    """FilmTec design equation 27b: dP (psi) = 0.01 q^1.7, q = average feed-concentrate flow (gpm).
    DPMAX acts as a multiplier (1.0 = vendor equation) for the sensitivity test."""
    return DPMAX * 0.01 * (q_avg * 4.4029) ** 1.7 / 14.504


def element(qf, cf, pf, A, B):
    """qf m3/h, cf mg/L, pf bar -> (qp, cp, qc, cc, pc, flux_lmh)."""
    def resid(qp):
        r = qp / qf
        qc = qf - qp
        cc_guess = cf / (1 - r)
        cavg = 0.5 * (cf + cc_guess)
        beta = math.exp(0.7 * r)
        jw = qp * 1000 / AREA                      # LMH
        cp = B * beta * cavg / (jw + B)            # from Js = B(beta*Cavg - Cp), Cp = Js/Jw
        pavg = pf - 0.5 * dp(0.5 * (qf + qc))
        ndp = pavg - (beta * OSM * cavg - OSM * cp)
        return jw - A * ndp
    hi = 0.5 * qf
    if resid(1e-6) > 0:  # no driving force
        return 0.0, 0.0, qf, cf, pf - dp(qf), 0.0
    qp = brentq(resid, 1e-6, hi) if resid(hi) > 0 else hi
    r = qp / qf; qc = qf - qp
    cavg = 0.5 * (cf + cf / (1 - r)); beta = math.exp(0.7 * r); jw = qp * 1000 / AREA
    cp = B * beta * cavg / (jw + B)
    cc = (qf * cf - qp * cp) / qc
    return qp, cp, qc, cc, pf - dp(0.5 * (qf + qc)), jw


def calibrate():
    qp = 40 / 24; qf = qp / 0.15; cf = 2000.0; pf = 15.5
    qc = qf - qp; cavg = 0.5 * (cf + cf / 0.85); beta = math.exp(0.7 * 0.15)
    jw = qp * 1000 / AREA; cp = cf * 0.005
    ndp = pf - 0.5 * dp(0.5 * (qf + qc)) - (beta * OSM * cavg - OSM * cp)
    A = jw / ndp
    B = jw * cp / (beta * cavg - cp)
    return A, B


def _array(cf, recovery, temp_c, A0, B0, boost=True, vessels=None):
    tcf = math.exp(2640 * (1 / 298.15 - 1 / (temp_c + 273.15)))
    A, B = A0 * tcf * FF, B0 * tcf
    n_el = sum(vessels) * N_EL
    qp_target = FLUX * AREA * n_el / 1000          # m3/h
    qf = qp_target / recovery

    shares = [nv / sum(vessels) for nv in vessels]

    def stage(q, c, p, nv):
        qv = q / nv; qp_s = salt = 0.0; fl = []
        for _ in range(N_EL):
            qpe, cpe, qv, c, p, jw = element(qv, c, p, A, B)
            qp_s += qpe * nv; salt += qpe * nv * cpe; fl.append(jw)
        return qp_s, salt, qv * nv, c, p, fl

    def run(pf):
        """Stage 1 at pf; with boost=True each later stage gets an interstage booster if needed for its share of the flux."""
        q, c, p = qf, cf, pf
        qp_tot = salt = 0.0; fluxes = []; work = qf * pf; boosts = []; pmax = pf
        for k, nv in enumerate(vessels):
            if k > 0 and boost:
                target = shares[k] * qp_target
                need = brentq(lambda P: stage(q, c, P, nv)[0] - target, 0.5, 120.0) if stage(q, c, 120.0, nv)[0] > target else 120.0
                b = max(0.0, need - p); boosts.append(b); work += q * b; p += b; pmax = max(pmax, p)
            qps, s_, q, c, p, fl = stage(q, c, p, nv)
            qp_tot += qps; salt += s_; fluxes += fl
        return qp_tot, salt / max(qp_tot, 1e-9), c, p, fluxes, work, boosts, pmax

    if boost:
        pf = brentq(lambda P: stage(qf, cf, P, vessels[0])[0] - shares[0] * qp_target, 2.0, 80.0)
        pf = brentq(lambda P: run(P)[0] - qp_target, 2.0, pf + 1e-6) if run(pf)[0] > qp_target + 1e-6 else pf
    else:
        pf = brentq(lambda P: run(P)[0] - qp_target, 2.0, 120.0)
    qp, cp, cc, pout, fl, work, boosts, pmax = run(pf)
    sec = work / (36 * ETA * qp)
    return {"stages": len(vessels), "feed_pressure_bar": pf, "sec_kwh_m3": sec, "permeate_tds": cp,
            "concentrate_tds": cc, "lead_flux_lmh": fl[0], "tail_flux_lmh": fl[-1],
            "boosters_bar": boosts, "conc_osmotic_bar": OSM * cc,
            "recovery_check": qp / qf, "max_pressure_bar": pmax,
            "within_41bar": bool(pmax <= 41.0), "lead_flux_ok": bool(fl[0] <= 35.0)}


def array(cf, recovery, temp_c, A0, B0, boost=True):
    """The designer's choice: for each recovery, the lowest-energy array among those normally used
    at that recovery -- one stage (<= 55 %), two stages 2:1 (50-85 %), three stages 4:2:1 (>= 70 %) --
    preferring designs within the 41-bar element rating."""
    cands = []
    if recovery <= 0.55 + 1e-9: cands.append([1])
    if 0.50 - 1e-9 <= recovery <= 0.85 + 1e-9: cands.append([2, 1])
    if recovery >= 0.70 - 1e-9: cands.append([4, 2, 1])
    res = [_array(cf, recovery, temp_c, A0, B0, boost, v) for v in cands]
    ok = [d for d in res if d["within_41bar"]] or res
    return min(ok, key=lambda d: d["sec_kwh_m3"])


def framework_pressure(tds, rec):
    # framework: P = pi(C_conc) + 6 + 1 bar, with its own rejection assumption
    cp = tds * (1 - 0.985) * 1.3
    cc = (tds - rec * cp) / (1 - rec)
    return OSM * cc + 7.0


def main():
    A0, B0 = calibrate()
    print(f"calibrated A = {A0:.3f} LMH/bar, B = {B0:.3f} LMH (BW30-400 datasheet test point)")
    feeds = {k: (v.tds_mg_l, v.temp_c) for k, v in EGYPT_BRACKISH_PRESETS.items() if v.water_type == "brackish"}
    feeds["design_study_ro_feed"] = (2892.0, 25.0)
    recs = [0.75, 0.80, 0.85, 0.90]
    rows = []
    print(f"{'feed':<32}{'TDS':>6}{'rec':>5}{'P_el':>7}{'P_fw':>7}{'SEC_el':>8}{'SEC_fw':>8}{'diff%':>7}{'leadJ':>7}{'Cp':>6}{'boost':>7}{'Pmax':>6}{'noB':>7}{'noBJ':>6}")
    for k, (tds, t) in feeds.items():
        for r in recs:
            e = array(tds, r, t, A0, B0)
            nb = array(tds, r, t, A0, B0, boost=False)
            fw = framework_sec(tds, r, 150000, "brackish")
            row = {"feed": k, "tds": tds, "temp_c": t, "recovery": r, **e, "fw_sec": fw,
                   "noboost_sec": nb["sec_kwh_m3"], "noboost_pressure_bar": nb["feed_pressure_bar"],
                   "noboost_lead_flux_lmh": nb["lead_flux_lmh"], "noboost_within_41bar": nb["within_41bar"],
                   "fw_pressure_bar": framework_pressure(tds, r), "diff_pct": (fw - e["sec_kwh_m3"]) / e["sec_kwh_m3"] * 100}
            rows.append(row)
            print(f"{k:<32}{tds:>6.0f}{r:>5.2f}{e['feed_pressure_bar']:>7.1f}{row['fw_pressure_bar']:>7.1f}"
                  f"{e['sec_kwh_m3']:>8.3f}{fw:>8.3f}{row['diff_pct']:>+7.1f}{e['lead_flux_lmh']:>7.1f}{e['permeate_tds']:>6.0f}{sum(e['boosters_bar']):>7.1f}{e['max_pressure_bar']:>6.0f}{nb['sec_kwh_m3']:>7.3f}{nb['lead_flux_lmh']:>6.0f}")
    # sensitivity at El Moghra 80 %
    sens = {}
    base = array(5000, 0.80, 27, A0, B0)["sec_kwh_m3"]
    global FF, DPMAX
    for name, ff, dpm in [("fouling 1.0", 1.0, 1.0), ("fouling 0.70", 0.70, 1.0), ("dP x0.5", 0.85, 0.5), ("dP x1.5", 0.85, 1.5)]:
        FF, DPMAX = ff, dpm
        sens[name] = (array(5000, 0.80, 27, A0, B0)["sec_kwh_m3"] - base) / base * 100
    FF, DPMAX = 0.85, 1.0
    print("sensitivity of element-model SEC (El Moghra, 80 %):", {k: f"{v:+.1f}%" for k, v in sens.items()})
    d = np.array([r["diff_pct"] for r in rows if r["feed"] != "design_study_ro_feed"])
    print(f"framework minus element model, presets: mean {d.mean():+.1f}%, range {d.min():+.1f} to {d.max():+.1f}%")
    json.dump({"A": A0, "B": B0, "rows": rows, "sensitivity_pct": sens}, open("analysis_element_level_ro.json", "w"), indent=1)


if __name__ == "__main__":
    main()


def slopes():
    """SEC-salinity slope at fixed recovery (0.80), element model vs framework, 1.8-8 g/L."""
    A0, B0 = calibrate()
    tds = np.arange(1800, 8001, 200)
    el = np.array([array(t, 0.80, 25, A0, B0)["sec_kwh_m3"] for t in tds])
    fw = np.array([framework_sec(t, 0.80, 150000, "brackish") for t in tds])
    return float(np.polyfit(tds / 1000, el, 1)[0]), float(np.polyfit(tds / 1000, fw, 1)[0])


if __name__ == "__main__":
    se, sf = slopes()
    print(f"SEC slope at 80 % recovery, 1.8-8 g/L: element model {se:.3f}, framework {sf:.3f} kWh/m3 per g/L")
    d = json.load(open("analysis_element_level_ro.json")); d["slope_element"] = se; d["slope_framework"] = sf
    json.dump(d, open("analysis_element_level_ro.json", "w"), indent=1)
