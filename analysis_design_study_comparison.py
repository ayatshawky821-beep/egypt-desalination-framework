"""
analysis_design_study_comparison.py (v1.1.1)

Re-derives, from the tables of El Sayed et al., Membranes 12, 923 (2022), the SEC, capital cost and
operating cost of their CONCEPTUAL 300,000 m3/day zero-liquid-discharge design study for Nile Delta
drainage water, and compares them with this framework. The study is a design benchmark (design-software
estimates), NOT an operating plant.

Run:  python3 analysis_design_study_comparison.py
"""
import numpy as np
from external_validation_rosa_egypt import framework_sec, load_rosa_data

feed, ro_permeate, product = 300000.0, 288000.0, 294000.0       # m3/day (RO1 270,000 + RO2 18,000; +6,000 thermal)
kw = {"pretreatment": 1525.0, "RO1 HP pumps": 12622.0, "RO1 booster": 626.0, "RO2 HP pumps": 2502.0, "thermal (TVC)": 913.0}
total_kw = sum(kw.values()); ro_kw = kw["RO1 HP pumps"] + kw["RO1 booster"] + kw["RO2 HP pumps"]
print(f"RO-train SEC: {ro_kw * 24 / ro_permeate:.3f} kWh/m3 of RO permeate")
print(f"whole-system electricity: {total_kw * 24 / product:.3f} kWh/m3 of product "
      f"(pretreatment {kw['pretreatment'] * 24 / product:.3f}, RO {ro_kw * 24 / product:.3f}, thermal {kw['thermal (TVC)'] * 24 / product:.3f})")
print(f"1.47e8 kWh/y over 300,000 m3/d x 365 x 0.9 = {1.47e8 / (feed * 365 * 0.9):.3f} kWh/m3 (the 1.49 in the validation set)")

capex_total, thermal, ponds = 116405.0, 15100.0, 15600.0         # thousand USD (paper Table 7; pretreatment total includes the pumping station)
print(f"capital cost: {capex_total * 1000 / product:.0f} USD per m3/d of product (whole ZLD system); "
      f"{(capex_total - thermal - ponds) * 1000 / ro_permeate:.0f} USD per m3/d of RO permeate without thermal unit and ponds")
print(f"operating cost: {47955e3 / (feed * 365 * 0.9):.3f} USD per m3 of feed; {47955e3 / (product * 365 * 0.9):.3f} per m3 of product")

sal, en = load_rosa_data()
print(f"\nRosa et al. curve at 2.892 g/L: {np.interp(2.892, sal, en):.2f} kWh/m3")
for r in (0.75, 0.85, 0.90):
    m = framework_sec(2892, r, 300000, "brackish")
    print(f"model membrane-train SEC at recovery {r:.2f}: {m:.2f}  ({(m - 1.3125) / 1.3125 * 100:+.0f}% vs RO train 1.31; "
          f"{(m - np.interp(2.892, sal, en)) / np.interp(2.892, sal, en) * 100:+.0f}% vs curve)")
