"""
egypt_parametric_study.py
 
Parametric sensitivity study built on desalination_plant_design.py, exploring
how key design outputs change as feed-water and operating parameters vary
across Egyptian brackish-groundwater conditions (El Moghra, Nile Delta,
Western Desert / Nubian Sandstone, Sinai) plus a Red Sea seawater case for
contrast.
 
Generates 6 curve sets, each saved as a PNG:
  1. RO/NF membrane area & specific energy vs. feed TDS
  2. Concentrate (brine) flow & TDS vs. system recovery
  3. Evaporation-pond area vs. plant capacity (arid inland disposal)
  4. Antiscalant dose & NF vs RO feed pressure vs. feed TDS
  5. Product water cost driver proxy: specific energy vs. recovery, by water type
  6. Site comparison bar chart: recovery, concentrate TDS, pond area across
     the four Egyptian aquifer presets + Red Sea seawater
 
Run:
    python3 egypt_parametric_study.py
Outputs PNGs into the current directory.
"""
 
import copy
import numpy as np
import matplotlib.pyplot as plt
 
from desalination_plant_design import (
    FeedWaterQuality,
    PlantCapacity,
    PlantFlows,
    ReverseOsmosisSystem,
    NanofiltrationSystem,
    BrineManagementSystem,
    EconomicAnalysis,
    EconomicAssumptions,
    EGYPT_BRACKISH_PRESETS,
    default_recovery_for,
)
 
plt.rcParams.update({
    "figure.figsize": (8, 5.5),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10.5,
})
 
BASE_CAPACITY_M3D = 20000.0
 
 
# --------------------------------------------------------------------------
# Helper: build flows + RO design for a given feed/recovery/capacity
# --------------------------------------------------------------------------
 
def ro_design_for(feed: FeedWaterQuality, recovery: float, capacity_m3d: float = BASE_CAPACITY_M3D):
    cap = PlantCapacity(permeate_flow_m3d=capacity_m3d, recovery=recovery)
    flows = PlantFlows.from_capacity(cap)
    # intake_m3d / pumping_head_m feed the supplementary groundwater-lift
    # energy term (Appendix A.3 / Section 4.7); this is what propagates
    # Plift into every curve and into economics_for() below, since all of
    # them read ro["specific_energy_consumption_kwh_m3_est"], which now
    # includes the lift contribution whenever feed.pumping_head_m is set.
    ro = ReverseOsmosisSystem(
        flow_m3d=flows.feed_to_membranes_m3d,
        permeate_m3d=flows.permeate_m3d,
        concentrate_m3d=flows.concentrate_m3d,
        feed_tds_mg_l=feed.tds_mg_l,
        water_type=feed.water_type,
        intake_m3d=flows.intake_m3d,
        pumping_head_m=feed.pumping_head_m,
    ).design()
    return flows, ro
 
 
def nf_design_for(feed: FeedWaterQuality, recovery: float, capacity_m3d: float = BASE_CAPACITY_M3D):
    cap = PlantCapacity(permeate_flow_m3d=capacity_m3d, recovery=recovery)
    flows = PlantFlows.from_capacity(cap)
    nf = NanofiltrationSystem(
        flow_m3d=flows.feed_to_membranes_m3d,
        feed_tds_mg_l=feed.tds_mg_l,
        feed_hardness_mg_l_caco3=feed.hardness_mg_l_caco3,
        recovery=recovery,
    ).design()
    return flows, nf
 
 
def base_egypt_feed(tds_override=None) -> FeedWaterQuality:
    f = copy.deepcopy(EGYPT_BRACKISH_PRESETS["el_moghra_aquifer"])
    if tds_override is not None:
        f.tds_mg_l = tds_override
    return f
 
 
# --------------------------------------------------------------------------
# CURVE 1: Membrane area & specific energy vs. feed TDS (RO), El Moghra-type water
# --------------------------------------------------------------------------
 
def curve_1_area_energy_vs_tds():
    tds_range = np.linspace(1000, 15000, 25)
    areas, sec_low, sec_high = [], [], []
 
    for tds in tds_range:
        feed = base_egypt_feed(tds_override=tds)
        recovery = default_recovery_for(feed)
        _, ro = ro_design_for(feed, recovery)
        areas.append(ro["total_membrane_area_m2"])
        lo, hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
        sec_low.append(lo)
        sec_high.append(hi)
 
    fig, ax1 = plt.subplots()
    ax1.plot(tds_range, areas, color="tab:blue", marker="o", markersize=3, label="RO membrane area")
    ax1.set_xlabel("Feed water TDS (mg/L)")
    ax1.set_ylabel("RO membrane area required (m²)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
 
    ax2 = ax1.twinx()
    ax2.fill_between(tds_range, sec_low, sec_high, color="tab:red", alpha=0.25,
                      label="Specific energy range")
    ax2.set_ylabel("Specific energy consumption (kWh/m³)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
 
    plt.title(f"RO Membrane Area & Specific Energy vs. Feed TDS\n"
              f"(Egypt brackish groundwater, plant capacity = {BASE_CAPACITY_M3D:,.0f} m³/day)")
    fig.tight_layout()
    fig.savefig("curve1_area_energy_vs_tds.png")
    plt.close(fig)
    print("Saved curve1_area_energy_vs_tds.png")
 
 
# --------------------------------------------------------------------------
# CURVE 2: Concentrate flow & TDS vs. system recovery
# --------------------------------------------------------------------------
 
def curve_2_concentrate_vs_recovery():
    recoveries = np.linspace(0.50, 0.90, 25)
    conc_flows, conc_tds = [], []
 
    feed = base_egypt_feed()  # El Moghra, TDS 5000 mg/L
    for r in recoveries:
        flows, ro = ro_design_for(feed, r)
        conc_flows.append(flows.concentrate_m3d)
        conc_tds.append(ro["concentrate_tds_mg_l_est"])
 
    fig, ax1 = plt.subplots()
    ax1.plot(recoveries * 100, conc_flows, color="tab:green", marker="o", markersize=3,
              label="Concentrate flow")
    ax1.set_xlabel("System recovery (%)")
    ax1.set_ylabel("Concentrate (brine) flow (m³/day)", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green")
 
    ax2 = ax1.twinx()
    ax2.plot(recoveries * 100, conc_tds, color="tab:purple", marker="s", markersize=3,
              label="Concentrate TDS")
    ax2.set_ylabel("Concentrate TDS (mg/L)", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")
 
    plt.title(f"Concentrate Flow & Salinity vs. System Recovery\n"
              f"(El Moghra Aquifer feed, {feed.tds_mg_l:,.0f} mg/L TDS, "
              f"{BASE_CAPACITY_M3D:,.0f} m³/day product)")
    fig.tight_layout()
    fig.savefig("curve2_concentrate_vs_recovery.png")
    plt.close(fig)
    print("Saved curve2_concentrate_vs_recovery.png")
 
 
# --------------------------------------------------------------------------
# CURVE 3: Evaporation pond area vs. plant capacity (arid inland disposal)
# --------------------------------------------------------------------------
 
def curve_3_pondarea_vs_capacity():
    capacities = np.linspace(2000, 100000, 25)
    pond_areas_ha = {}
 
    for site_name, feed in EGYPT_BRACKISH_PRESETS.items():
        if feed.water_type != "brackish":
            continue
        recovery = default_recovery_for(feed)
        areas = []
        for cap_m3d in capacities:
            flows, ro = ro_design_for(feed, recovery, capacity_m3d=cap_m3d)
            brine = BrineManagementSystem(
                concentrate_flow_m3d=flows.concentrate_m3d,
                concentrate_tds_mg_l=ro["concentrate_tds_mg_l_est"],
                feed_tds_mg_l=feed.tds_mg_l,
                is_coastal_site=False,
                is_arid_climate=True,
            ).design_evaporation_ponds_alternative().design
            areas.append(brine["evaporation_pond_option"]["required_pond_area_hectares"])
        pond_areas_ha[site_name] = areas
 
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "^", "s", "D"]
    fig, ax = plt.subplots()
    for i, (site_name, areas) in enumerate(pond_areas_ha.items()):
        ax.plot(capacities, areas, marker=markers[i % 4], markersize=4,
                 linestyle=line_styles[i % 4], label=site_name.replace("_", " ").title())
 
    ax.set_xlabel("Plant product-water capacity (m³/day)")
    ax.set_ylabel("Required evaporation pond area (hectares)")
    ax.set_title("Evaporation Pond Land Area vs. Plant Capacity\n"
                  "(Arid inland brine disposal, Egyptian brackish aquifer presets)")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig("curve3_pondarea_vs_capacity.png")
    plt.close(fig)
    print("Saved curve3_pondarea_vs_capacity.png")
 
 
# --------------------------------------------------------------------------
# CURVE 4: Feed pressure & antiscalant dose vs. feed TDS — NF vs RO
# --------------------------------------------------------------------------
 
def curve_4_pressure_dose_vs_tds():
    tds_range = np.linspace(1000, 10000, 20)
    ro_press_mid, nf_press_mid, antiscalant = [], [], []
 
    for tds in tds_range:
        feed = base_egypt_feed(tds_override=tds)
        recovery = default_recovery_for(feed)
        _, ro = ro_design_for(feed, recovery)
        _, nf = nf_design_for(feed, 0.85)
 
        ro_lo, ro_hi = [float(x) for x in ro["feed_pressure_bar_est"].split(" ")[0].split("-")]
        nf_lo, nf_hi = [float(x) for x in nf["feed_pressure_bar_est"].split(" ")[0].split("-")]
        ro_press_mid.append((ro_lo + ro_hi) / 2)
        nf_press_mid.append((nf_lo + nf_hi) / 2)
        antiscalant.append(3.5)  # constant dose per design rule (brackish default)
 
    fig, ax1 = plt.subplots()
    ax1.plot(tds_range, ro_press_mid, color="tab:orange", marker="o", markersize=3, label="RO feed pressure")
    ax1.plot(tds_range, nf_press_mid, color="tab:cyan", marker="s", markersize=3, label="NF feed pressure")
    ax1.set_xlabel("Feed water TDS (mg/L)")
    ax1.set_ylabel("Estimated feed pressure (bar)")
    ax1.legend(loc="upper left")
 
    plt.title("Estimated Membrane Feed Pressure vs. Feed TDS: NF vs. Brackish RO\n"
              "(El Moghra-type Egyptian brackish groundwater)")
    fig.tight_layout()
    fig.savefig("curve4_pressure_vs_tds_NF_RO.png")
    plt.close(fig)
    print("Saved curve4_pressure_vs_tds_NF_RO.png")
 
 
# --------------------------------------------------------------------------
# CURVE 5: Specific energy vs recovery, brackish vs seawater
# --------------------------------------------------------------------------
 
def curve_5_energy_vs_recovery_by_type():
    fig, ax = plt.subplots()
 
    recoveries_bw = np.linspace(0.60, 0.90, 20)
    recoveries_sw = np.linspace(0.35, 0.55, 20)
 
    bw_feed = base_egypt_feed()  # brackish, 5000 mg/L
    sw_feed = EGYPT_BRACKISH_PRESETS["red_sea_seawater"]
 
    def energy_series(feed, recoveries):
        mids = []
        for r in recoveries:
            _, ro = ro_design_for(feed, r)
            lo, hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
            mids.append((lo + hi) / 2)
        return mids
 
    bw_energy = energy_series(bw_feed, recoveries_bw)
    sw_energy = energy_series(sw_feed, recoveries_sw)
 
    ax.plot(recoveries_bw * 100, bw_energy, color="tab:brown", marker="o", markersize=3,
             label=f"Brackish RO (El Moghra, {bw_feed.tds_mg_l:,.0f} mg/L)")
    ax.plot(recoveries_sw * 100, sw_energy, color="tab:blue", marker="s", markersize=3,
             label=f"Seawater RO (Red Sea, {sw_feed.tds_mg_l:,.0f} mg/L)")
 
    ax.set_xlabel("System recovery (%)")
    ax.set_ylabel("Estimated specific energy consumption (kWh/m³)")
    ax.set_title("Specific Energy vs. Recovery: Brackish RO vs. Seawater RO\n"
                 "(Illustrates why Egypt's inland brackish sites are far less energy-intensive)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("curve5_energy_vs_recovery_brackish_vs_seawater.png")
    plt.close(fig)
    print("Saved curve5_energy_vs_recovery_brackish_vs_seawater.png")
 
 
# --------------------------------------------------------------------------
# CURVE 6: Site comparison across Egyptian aquifer presets
# --------------------------------------------------------------------------
 
def curve_6_site_comparison():
    sites = list(EGYPT_BRACKISH_PRESETS.keys())
    labels = [s.replace("_", " ").title() for s in sites]
 
    recoveries, conc_tds_vals, pond_areas = [], [], []
    for s in sites:
        feed = EGYPT_BRACKISH_PRESETS[s]
        recovery = default_recovery_for(feed)
        flows, ro = ro_design_for(feed, recovery)
        recoveries.append(recovery * 100)
        conc_tds_vals.append(ro["concentrate_tds_mg_l_est"])
        is_coastal = "coastal" in s or feed.water_type == "seawater"
        brine = BrineManagementSystem(
            concentrate_flow_m3d=flows.concentrate_m3d,
            concentrate_tds_mg_l=ro["concentrate_tds_mg_l_est"],
            feed_tds_mg_l=feed.tds_mg_l,
            is_coastal_site=is_coastal,
            is_arid_climate=not is_coastal,
        ).design_evaporation_ponds_alternative().design
        pond_areas.append(brine["evaporation_pond_option"]["required_pond_area_hectares"])
 
    x = np.arange(len(sites))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
 
    axes[0].bar(x, recoveries, color="tab:green")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Recommended recovery (%)")
    axes[0].set_title("System Recovery")
 
    axes[1].bar(x, conc_tds_vals, color="tab:purple")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Concentrate TDS (mg/L)")
    axes[1].set_title("Concentrate Salinity")
 
    axes[2].bar(x, pond_areas, color="tab:orange")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[2].set_ylabel("Evaporation pond area (ha)")
    axes[2].set_title(f"Pond Area @ {BASE_CAPACITY_M3D:,.0f} m³/day")
 
    fig.suptitle("Comparison Across Egyptian Feed-Water Presets", fontsize=13)
    fig.tight_layout()
    fig.savefig("curve6_site_comparison.png")
    plt.close(fig)
    print("Saved curve6_site_comparison.png")
 
 
# --------------------------------------------------------------------------
# Helper: run economics for a given feed/recovery/capacity
# --------------------------------------------------------------------------
 
def economics_for(feed: FeedWaterQuality, recovery: float, capacity_m3d: float = BASE_CAPACITY_M3D):
    flows, ro = ro_design_for(feed, recovery, capacity_m3d)
    sec_lo, sec_hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
    econ = EconomicAnalysis(
        capacity_m3d=capacity_m3d,
        water_type=feed.water_type,
        specific_energy_kwh_m3=(sec_lo + sec_hi) / 2,
    ).run_full_analysis()
    return flows, ro, econ
 
 
# --------------------------------------------------------------------------
# CURVE 7: LCOW vs. plant capacity (economies of scale), all 5 presets
# --------------------------------------------------------------------------
 
def curve_7_lcow_vs_capacity():
    capacities = np.linspace(2000, 100000, 25)
    fig, ax = plt.subplots()
    line_styles = ["-", "--", "-.", ":", "-"]
    markers = ["o", "^", "s", "D", "v"]
 
    for i, (site_name, feed) in enumerate(EGYPT_BRACKISH_PRESETS.items()):
        recovery = default_recovery_for(feed)
        lcows = []
        for cap_m3d in capacities:
            _, _, econ = economics_for(feed, recovery, cap_m3d)
            lcows.append(econ["lcow"]["lcow_usd_m3"])
        ax.plot(capacities, lcows, marker=markers[i % 5], markersize=3,
                 linestyle=line_styles[i % 5], label=site_name.replace("_", " ").title())
 
    ax.set_xlabel("Plant product-water capacity (m³/day)")
    ax.set_ylabel("Levelized cost of water, LCOW (USD/m³)")
    ax.set_title("LCOW vs. Plant Capacity: Economies of Scale\n"
                 "(All five feed-water presets; CAPEX/OPEX benchmarks in Appendix A)")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig("curve7_lcow_vs_capacity.png")
    plt.close(fig)
    print("Saved curve7_lcow_vs_capacity.png")
 
 
# --------------------------------------------------------------------------
# CURVE 8: Avoided-cost payback period vs. system recovery
# --------------------------------------------------------------------------
 
def curve_8_payback_vs_recovery():
    recoveries = np.linspace(0.50, 0.90, 25)
    feed = base_egypt_feed()  # El Moghra, 5,000 mg/L TDS
    paybacks, lcows = [], []
 
    for r in recoveries:
        _, _, econ = economics_for(feed, r)
        capex = econ["capex"]["total_capex_usd"]
        net_cf = econ["roi_payback"]["avoided_cost_scenario"]["annual_net_cashflow_usd"]
        paybacks.append(capex / net_cf if net_cf > 0 else np.nan)
        lcows.append(econ["lcow"]["lcow_usd_m3"])
 
    fig, ax1 = plt.subplots()
    ax1.plot(recoveries * 100, paybacks, color="tab:blue", marker="o", markersize=3,
              label="Avoided-cost payback")
    ax1.set_xlabel("System recovery (%)")
    ax1.set_ylabel("Simple payback period (years)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
 
    ax2 = ax1.twinx()
    ax2.plot(recoveries * 100, lcows, color="tab:red", marker="s", markersize=3,
              label="LCOW")
    ax2.set_ylabel("LCOW (USD/m³)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
 
    plt.title("Payback Period & LCOW vs. System Recovery\n"
              "(El Moghra Aquifer feed, illustrative avoided-cost benefit of $2.00/m³)")
    fig.tight_layout()
    fig.savefig("curve8_payback_vs_recovery.png")
    plt.close(fig)
    print("Saved curve8_payback_vs_recovery.png")
 
 
# --------------------------------------------------------------------------
# CURVE 9: LCOW vs. feed TDS, brackish vs seawater
# --------------------------------------------------------------------------
 
def curve_9_lcow_vs_tds():
    tds_range_bw = np.linspace(1000, 15000, 20)
    lcows_bw = []
    for tds in tds_range_bw:
        feed = base_egypt_feed(tds_override=tds)
        recovery = default_recovery_for(feed)
        _, _, econ = economics_for(feed, recovery)
        lcows_bw.append(econ["lcow"]["lcow_usd_m3"])
 
    sw_feed = EGYPT_BRACKISH_PRESETS["red_sea_seawater"]
    _, _, econ_sw = economics_for(sw_feed, default_recovery_for(sw_feed))
    lcow_sw = econ_sw["lcow"]["lcow_usd_m3"]
 
    fig, ax = plt.subplots()
    ax.plot(tds_range_bw, lcows_bw, color="tab:brown", marker="o", markersize=3,
             label="Brackish RO (El Moghra-type, varying TDS)")
    ax.axhline(lcow_sw, color="tab:blue", linestyle="--",
               label=f"Seawater RO (Red Sea, {sw_feed.tds_mg_l:,.0f} mg/L) = ${lcow_sw:.2f}/m³")
 
    ax.set_xlabel("Feed water TDS (mg/L)")
    ax.set_ylabel("Levelized cost of water, LCOW (USD/m³)")
    ax.set_title("LCOW vs. Feed TDS: Brackish RO Trend vs. Seawater RO Reference\n"
                 f"(Plant capacity = {BASE_CAPACITY_M3D:,.0f} m³/day)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("curve9_lcow_vs_tds.png")
    plt.close(fig)
    print("Saved curve9_lcow_vs_tds.png")
 
 
# --------------------------------------------------------------------------
if __name__ == "__main__":
    curve_1_area_energy_vs_tds()
    curve_2_concentrate_vs_recovery()
    curve_3_pondarea_vs_capacity()
    curve_4_pressure_dose_vs_tds()
    curve_5_energy_vs_recovery_by_type()
    curve_6_site_comparison()
    curve_7_lcow_vs_capacity()
    curve_8_payback_vs_recovery()
    curve_9_lcow_vs_tds()
    print("\nAll parametric curves generated.")
 