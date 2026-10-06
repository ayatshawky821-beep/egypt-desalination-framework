"""
external_validation_stillwell.py  (REVISED in v1.1.1)

Comparison with Stillwell & Webber (2016), Water 8, 601,
https://doi.org/10.3390/w8120601 (open access, CC BY 3.0): multiple linear
regression of SEC on 36 real plants (2,500-368,000 m3/day, R2 = 0.86).

IMPORTANT - what this script does and does not do
-------------------------------------------------
The paper prints its municipal-scale model as
    SEC = 260 - 0.13*YR + 8.3e-5*c_rw - 2.4e-3*c_pw
and states that coefficients are "rounded to two significant figures".
The year term (0.13 x ~2,000) is then uncertain by about +/-10 kWh/m3 from
rounding alone, so the ABSOLUTE level of the printed equation cannot be
used: it does not even reproduce the authors' own Carlsbad example
(printed equation -> ~0.1 kWh/m3; paper reports 3.5).

An earlier version of this file (v1.1.0) "corrected" the coefficients by
searching for rescalings that matched the Carlsbad value. That fit required
flipping the sign of the product-water term, which contradicts the paper's
abstract (higher product-water salinity -> LOWER SEC). It was wrong and has
been removed. No result in the manuscript depends on it.

What can legitimately be used is the partial SLOPE for raw-water TDS,
8.3e-5 kWh/m3 per mg/L (= 0.083 kWh/m3 per g/L), which is printed to a
usable precision. This script compares that slope with the slope of the
Rosa et al. (2025) curve and with the framework's own salinity sensitivity.

Run:  python3 external_validation_stillwell.py
"""
import numpy as np
from external_validation_rosa_egypt import framework_sec, load_rosa_data

PRINTED = {"b0": 260.0, "b_yr": -0.13, "b_crw": 8.3e-5, "b_cpw": -2.4e-3}


def printed_equation(yr, crw, cpw):
    return PRINTED["b0"] + PRINTED["b_yr"] * yr + PRINTED["b_crw"] * crw + PRINTED["b_cpw"] * cpw


if __name__ == "__main__":
    print("1) Does the printed equation reproduce the paper's own Carlsbad example?")
    carlsbad = printed_equation(2015, 35000, 350)
    print(f"   printed equation -> {carlsbad:.2f} kWh/m3 ; paper reports 3.5 (reported plant value 3.6)  => NOT reproduced")
    print(f"   rounding uncertainty of the year term alone: +/- {0.005 * 2015:.1f} kWh/m3  => absolute level unusable\n")

    print("2) Salinity sensitivity (kWh/m3 per g/L of feed salinity)")
    sal, en = load_rosa_data()
    m = (sal >= 2) & (sal <= 10)
    rosa_slope = np.polyfit(sal[m], en[m], 1)[0]
    sw_slope = PRINTED["b_crw"] * 1000.0
    print(f"   Rosa et al. curve, 2-10 g/L          : {rosa_slope:.3f}")
    print(f"   Stillwell & Webber partial slope     : {sw_slope:.3f}  (holding year and product-water TDS fixed)")
    tds = np.arange(2000, 10001, 500)
    for rec in (0.75, 0.80, 0.85):
        sec = np.array([framework_sec(t, rec, 150000, "brackish") for t in tds])
        slope = np.polyfit(tds / 1000.0, sec, 1)[0]
        print(f"   framework, recovery {rec:.2f}, 2-10 g/L  : {slope:.3f}")
    print("\n   The framework's sensitivity to salinity is about 2-4 times that of both external sources.")
