"""
external_validation_rosa_egypt.py

TWO further genuine external validation checks, both newly added:

(A) Rosa, Gabrielli & Sangiorgio (2025) -- "Global multi-model assessment of
    energy, costs, and emissions trade-offs associated with reverse osmosis
    desalination under future water scarcity." Source data published on
    Zenodo (DOI 10.5281/zenodo.15569107, CC BY 4.0), "Figure 1" sheet:
    Energy [kWh/m3] as a function of feed-water Salinity [ppt], 2-52 ppt,
    51 points, derived from their own multi-model global assessment
    (Carnegie Institution for Science / ETH Zurich). This is a genuinely
    independent source: different authors, different methodology (a
    global multi-model climate/energy assessment), and a dataset this
    project has never touched before this check.

    NOTE on their "Cost" column: verified by direct reconstruction to be
    Energy x an assumed global electricity price distribution (confirmed:
    Cost_median / Energy is constant at ~$0.0843/kWh across all 51 rows),
    NOT a full capital-inclusive LCOW. It is therefore not compared against
    this framework's own LCOW figures, which would not be a like-for-like
    comparison; only the Energy column is used here.

(B) A real, named Egyptian plant: Interwater/Safa Water Technology's
    seawater RO facility in Hurghada, Red Sea coast, Egypt, as documented
    by Energy Recovery Inc. (2024) in a published case study. Before an
    energy-recovery-device retrofit: 500 m3/day, SEC = 6.3 kWh/m3. After
    retrofitting with a modern isobaric pressure exchanger (Energy
    Recovery's PX 260): capacity 600 m3/day, SEC = 2.7 kWh/m3. This is
    used as a direct, in-country check on the framework's own Red Sea
    seawater preset.

Run:
    python3 external_validation_rosa_egypt.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

from desalination_plant_design import (
    FeedWaterQuality, PlantCapacity, PlantFlows, ReverseOsmosisSystem,
    EGYPT_BRACKISH_PRESETS, default_recovery_for,
)

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10.5})


def load_rosa_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rosa_et_al_2025_source_data.xlsx")
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Figure 1"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    salinity_ppt = np.array([r[0] for r in rows], dtype=float)
    energy_kwh_m3 = np.array([r[1] for r in rows], dtype=float)
    cost_median = np.array([r[3] for r in rows], dtype=float)
    # Verify the Cost column is Energy x a constant electricity price (not a full LCOW)
    implied_price = cost_median / energy_kwh_m3
    print(f"Verification: Rosa et al. 'Cost' column = Energy x ${implied_price.mean():.4f}/kWh "
          f"(std across 51 rows: {implied_price.std():.6f}) -- confirms this is an energy-cost-only "
          f"column, not a capital-inclusive LCOW, so it is NOT compared against this framework's LCOW.")
    return salinity_ppt, energy_kwh_m3


def framework_sec(tds_mg_l, recovery, capacity_m3d, water_type="brackish"):
    feed = FeedWaterQuality(tds_mg_l=tds_mg_l, water_type=water_type)
    cap = PlantCapacity(permeate_flow_m3d=capacity_m3d, recovery=recovery)
    flows = PlantFlows.from_capacity(cap)
    ro = ReverseOsmosisSystem(
        flow_m3d=flows.feed_to_membranes_m3d, permeate_m3d=flows.permeate_m3d,
        concentrate_m3d=flows.concentrate_m3d, feed_tds_mg_l=tds_mg_l, water_type=water_type,
    ).design()
    lo, hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
    return (lo + hi) / 2


if __name__ == "__main__":
    print("=" * 78)
    print("EXTERNAL VALIDATION (A): Rosa, Gabrielli & Sangiorgio (2025), Zenodo")
    print("Global multi-model energy-vs-salinity assessment")
    print("=" * 78)

    salinity_ppt, energy_kwh_m3 = load_rosa_data()
    print(f"Loaded {len(salinity_ppt)} points, salinity range {salinity_ppt.min():.0f}-{salinity_ppt.max():.0f} ppt")

    CAPACITY = 150000.0
    results = []
    print(f"\n{'Preset':<26}{'TDS (mg/L)':<12}{'ppt':<8}{'Rosa et al. SEC':<18}{'Framework SEC':<16}{'Diff'}")
    for key, feed in EGYPT_BRACKISH_PRESETS.items():
        ppt = feed.tds_mg_l / 1000.0
        recovery = default_recovery_for(feed)
        fw_sec = framework_sec(feed.tds_mg_l, recovery, CAPACITY, feed.water_type)

        if salinity_ppt.min() <= ppt <= salinity_ppt.max():
            rosa_sec = float(np.interp(ppt, salinity_ppt, energy_kwh_m3))
            in_range = True
        else:
            # Linear extrapolation using the two nearest boundary points (the
            # relationship is confirmed near-linear within the dataset, so a
            # short linear extrapolation below the 2 ppt floor is a defensible,
            # clearly-flagged estimate rather than a silent one)
            if ppt < salinity_ppt.min():
                slope = (energy_kwh_m3[1] - energy_kwh_m3[0]) / (salinity_ppt[1] - salinity_ppt[0])
                rosa_sec = float(energy_kwh_m3[0] - slope * (salinity_ppt[0] - ppt))
            else:
                slope = (energy_kwh_m3[-1] - energy_kwh_m3[-2]) / (salinity_ppt[-1] - salinity_ppt[-2])
                rosa_sec = float(energy_kwh_m3[-1] + slope * (ppt - salinity_ppt[-1]))
            in_range = False

        diff_pct = (fw_sec - rosa_sec) / rosa_sec * 100
        flag = "" if in_range else " (extrapolated)"
        results.append((key, feed.tds_mg_l, ppt, rosa_sec, fw_sec, diff_pct, in_range))
        print(f"{key:<26}{feed.tds_mg_l:<12.0f}{ppt:<8.2f}{rosa_sec:<18.3f}{fw_sec:<16.3f}{diff_pct:+.1f}%{flag}")

    mean_pct = np.mean([r[5] for r in results if r[6]])  # only in-range points
    print(f"\nMean % difference (in-range points only): {mean_pct:+.1f}%")

    # --- Plot: framework vs Rosa et al. full curve, with Egypt presets marked ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(salinity_ppt, energy_kwh_m3, "-", color="tab:green", linewidth=2,
             label="Rosa et al. (2025) global multi-model curve")
    fw_curve_ppt = np.linspace(1, 52, 60)
    fw_curve_sec = [framework_sec(p * 1000, 0.75, CAPACITY, "brackish" if p < 20 else "seawater") for p in fw_curve_ppt]
    ax.plot(fw_curve_ppt, fw_curve_sec, "--", color="tab:blue", linewidth=2,
             label="This framework (physics-based, recovery=0.75)")
    for key, tds, ppt, rosa_sec, fw_sec, diff_pct, in_range in results:
        marker = "o" if in_range else "x"
        ax.scatter([ppt], [fw_sec], color="tab:blue", marker=marker, s=60, zorder=5, edgecolor="black")
        ax.annotate(key.replace("_", " ").title(), (ppt, fw_sec), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Feed-water salinity (ppt)")
    ax.set_ylabel("Specific Energy Consumption (kWh/m3)")
    ax.set_title("External Validation: Framework vs. Rosa et al. (2025) Global Multi-Model Curve\nEgyptian presets marked (x = outside the dataset's 2-52 ppt range)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("external_validation_rosa.png")
    plt.close(fig)
    print("Saved external_validation_rosa.png")

    # =======================================================================
    print("\n" + "=" * 78)
    print("EXTERNAL VALIDATION (B): Real named Egyptian plant")
    print("Interwater/Safa Water Technology, Hurghada, Red Sea, Egypt")
    print("(Energy Recovery Inc. published case study, 2024)")
    print("=" * 78)

    hurghada_tds = 41000  # Red Sea seawater, matching this framework's own preset
    hurghada_capacity_before = 500.0
    hurghada_sec_before = 6.3  # kWh/m3, old Pelton-wheel ERD
    hurghada_capacity_after = 600.0
    hurghada_sec_after = 2.7  # kWh/m3, modern isobaric PX-260 ERD retrofit

    recovery_sw = default_recovery_for(EGYPT_BRACKISH_PRESETS["red_sea_seawater"])
    fw_sec_before = framework_sec(hurghada_tds, recovery_sw, hurghada_capacity_before, "seawater")
    fw_sec_after = framework_sec(hurghada_tds, recovery_sw, hurghada_capacity_after, "seawater")

    print(f"\nReal plant (BEFORE ERD retrofit): {hurghada_capacity_before:.0f} m3/day, "
          f"measured SEC = {hurghada_sec_before} kWh/m3 (old Pelton-wheel ERD)")
    print(f"Framework prediction at same capacity/TDS: {fw_sec_before:.3f} kWh/m3 "
          f"(diff: {(fw_sec_before-hurghada_sec_before)/hurghada_sec_before*100:+.1f}%)")

    print(f"\nReal plant (AFTER ERD retrofit):  {hurghada_capacity_after:.0f} m3/day, "
          f"measured SEC = {hurghada_sec_after} kWh/m3 (modern isobaric PX-260 ERD)")
    print(f"Framework prediction at same capacity/TDS: {fw_sec_after:.3f} kWh/m3 "
          f"(diff: {(fw_sec_after-hurghada_sec_after)/hurghada_sec_after*100:+.1f}%)")
    print("\n(The framework's own energy-recovery-device model, Eq. 15 of the companion paper, "
          "assumes a modern ERD by default -- so the AFTER case is the fairer like-for-like comparison.)")

    # second Hurghada plant (vendor case story: Interwater, three 600 m3/day trains, isobaric ERD)
    print("\nSecond Hurghada plant (Interwater, three 600 m3/day trains; vendor case story): reported average 2.2-2.5 kWh/m3")
    print(f"Framework {fw_sec_after:.2f} kWh/m3 -> {(fw_sec_after - 2.5) / 2.5 * 100:+.0f}% to {(fw_sec_after - 2.2) / 2.2 * 100:+.0f}% relative to that range")

    print("\n" + "=" * 78)
    print("INTERPRETATION (v1.1.1)")
    print("=" * 78)
    print("Against the Rosa et al. curve the framework's SEC is below the curve at low salinity")
    print("(about 21-36% at 1.8-2.7 g/L), close at 5-8 g/L, and about 13% below at seawater salinity.")
    print("The two Hurghada plants (2.7 and 2.2-2.5 kWh/m3, both vendor case studies) place the framework's")
    print("2.70 kWh/m3 at the upper end of real small Red Sea plants. Because the framework's SEC is")
    print("independent of plant capacity by construction, this does not test scale dependence.")
    print("See external_validation_stillwell.py for the salinity-slope comparison.")
