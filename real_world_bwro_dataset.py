"""
real_world_bwro_dataset.py
 
A compiled, citable dataset of REAL, published brackish-water RO (and one
agricultural-drainage-water RO/ZLD) plant capacities, feed TDS, CAPEX, and
OPEX figures, drawn from two peer-reviewed sources:
 
  - Pearson, J.L., Michael, P.R., Ghaffour, N., & Missimer, T.M. (2021).
    Economics and Energy Consumption of Brackish Water Reverse Osmosis
    Desalination: Innovations and Impacts of Feedwater Quality. Membranes,
    11(8), 616. (Tables 2 and 3: 7 Florida BWRO plants, 9 Texas BWRO plants)
  - Dawoud, M.A., Sallam, G.R., Abdelrahman, M.A., & Emam, M. (2024). The
    Performance and Feasibility of Solar-Powered Desalination for Brackish
    Groundwater in Egypt. Sustainability, 16(4), 1630. (El Alamein plant)
  - El Sayed, M.M., Abulnour, A.M.G., Tewfik, S.R., Sorour, M.H., &
    Shaalan, H.F. (2022). Reverse Osmosis Membrane Zero Liquid Discharge for
    Agriculture Drainage Water Desalination. Membranes, 12(10), 923.
    (Nile Delta agricultural drainage water ZLD plant)
 
WHY THIS FILE EXISTS: a synthetic dataset generated purely by sampling this
project's own physics-based model (as in egypt_ml_surrogate.py) can only
ever validate that the ML surrogate learned the physics-based model
correctly -- it cannot tell you whether the physics-based model itself is
accurate. This file provides an independent, real-world anchor: ~18 actual
operating or fully-engineered plants, with real capacities, salinities, and
costs, against which the physics-based model's own predictions can be
checked and (for the economies-of-scale exponent) recalibrated.
 
HONESTY NOTE: 18 data points is far too few to train a multi-feature ML
model (e.g., a Random Forest) from scratch with any statistical confidence.
This dataset is therefore used for two more appropriate purposes: (1) an
out-of-sample accuracy check of the physics-based model's OPEX/CAPEX
predictions, and (2) an empirical re-fit of the single economies-of-scale
exponent (Eq. 20), which — unlike a full multi-feature model — a dataset
this size CAN support.
"""
 
# Each record: (plant_name, capacity_m3d, feed_tds_mg_l, capex_usd_per_m3d or
# None, opex_usd_per_m3, source_key)
# source_key: "pearson_fl" | "pearson_tx" | "dawoud_elalamein" | "elsayed_adw"
 
REAL_BWRO_PLANTS = [
    # --- Pearson et al. (2021) Table 2: Southwest Florida BWRO plants (OPEX only) ---
    {"name": "Cape Coral North (FL)", "capacity_m3d": 45420, "tds_mg_l": 2452,
     "capex_usd_m3d": None, "opex_usd_m3": 0.50, "source": "pearson_fl"},
    {"name": "Cape Coral Southwest (FL)", "capacity_m3d": 68130, "tds_mg_l": 2132,
     "capex_usd_m3d": None, "opex_usd_m3": 0.36, "source": "pearson_fl"},
    {"name": "Island Water Association (FL)", "capacity_m3d": 22617, "tds_mg_l": 2800,
     "capex_usd_m3d": None, "opex_usd_m3": 0.64, "source": "pearson_fl"},
    {"name": "Lee County Green Meadows IX&RO (FL)", "capacity_m3d": 60560, "tds_mg_l": 2913,
     "capex_usd_m3d": None, "opex_usd_m3": 0.425, "source": "pearson_fl"},
    {"name": "Lee County North (FL)", "capacity_m3d": 43906, "tds_mg_l": 2700,  # TDS not reported; using regional average
     "capex_usd_m3d": None, "opex_usd_m3": 0.475, "source": "pearson_fl"},
    {"name": "Lee County Pinewoods RO&NF (FL)", "capacity_m3d": 20060, "tds_mg_l": 3848,
     "capex_usd_m3d": None, "opex_usd_m3": 0.44, "source": "pearson_fl"},
    {"name": "Marco Island South (FL)", "capacity_m3d": 22710, "tds_mg_l": 2700,  # TDS not reported; using regional average
     "capex_usd_m3d": None, "opex_usd_m3": 0.43, "source": "pearson_fl"},
 
    # --- Pearson et al. (2021) Table 3: Texas BWRO plants (CAPEX + OPEX, blended capacity basis) ---
    {"name": "NAWSC Doolittle (TX)", "capacity_m3d": 11364, "tds_mg_l": 2750,
     "capex_usd_m3d": 704, "opex_usd_m3": 0.29, "source": "pearson_tx"},
    {"name": "NAWSC Owassa (TX)", "capacity_m3d": 5682, "tds_mg_l": 2750,
     "capex_usd_m3d": 1030, "opex_usd_m3": 0.35, "source": "pearson_tx"},
    {"name": "Fort Hancock WCID (TX)", "capacity_m3d": 1894, "tds_mg_l": 2200,
     "capex_usd_m3d": 1782, "opex_usd_m3": 0.86, "source": "pearson_tx"},
    {"name": "Roscoe (TX)", "capacity_m3d": 1364, "tds_mg_l": 3800,
     "capex_usd_m3d": 714, "opex_usd_m3": 0.23, "source": "pearson_tx"},
    {"name": "Kay Bailey Hutchinson (TX)", "capacity_m3d": 56818, "tds_mg_l": 2500,
     "capex_usd_m3d": 1602, "opex_usd_m3": 0.40, "source": "pearson_tx"},
    {"name": "North Cameron Regional A (TX)", "capacity_m3d": 3788, "tds_mg_l": 3500,
     "capex_usd_m3d": 1848, "opex_usd_m3": 0.63, "source": "pearson_tx"},
    {"name": "North Cameron Regional B (TX)", "capacity_m3d": 7576, "tds_mg_l": 3500,
     "capex_usd_m3d": 1056, "opex_usd_m3": 0.47, "source": "pearson_tx"},
    {"name": "Southmost (TX)", "capacity_m3d": 22727, "tds_mg_l": 3500,
     "capex_usd_m3d": 1012, "opex_usd_m3": 0.54, "source": "pearson_tx"},
    {"name": "NAWSC Lasara (TX)", "capacity_m3d": 3788, "tds_mg_l": 2750,
     "capex_usd_m3d": 528, "opex_usd_m3": 0.63, "source": "pearson_tx"},
 
    # --- Egypt data points (this paper's Section 6 validation) ---
    {"name": "El Alamein pilot (Egypt, brackish)", "capacity_m3d": 1000, "tds_mg_l": 21775,
     "capex_usd_m3d": 890, "opex_usd_m3": 0.59, "source": "dawoud_elalamein"},
    {"name": "Nile Delta ADW ZLD (Egypt, agricultural drainage)", "capacity_m3d": 300000, "tds_mg_l": 2705,
     "capex_usd_m3d": 396, "opex_usd_m3": 0.487, "source": "elsayed_adw"},
]
 
SOURCE_CITATIONS = {
    "pearson_fl": "Pearson et al. (2021), Membranes 11(8):616, Table 2",
    "pearson_tx": "Pearson et al. (2021), Membranes 11(8):616, Table 3 (recalculated from Arroyo & Shirazi, 2012)",
    "dawoud_elalamein": "Dawoud et al. (2024), Sustainability 16(4):1630",
    "elsayed_adw": "El Sayed et al. (2022), Membranes 12(10):923",
}
 
 
 
# --------------------------------------------------------------------------
# Auxiliary data used for calibration (Section 6.4 of the paper)
# --------------------------------------------------------------------------
 
# Electricity tariff (USD/kWh) actually REPORTED for each plant's source.
# Pearson et al. (2021) Table 3 power-cost column (Texas); Dawoud et al.
# (2024) solar tariff (El Alamein); El Sayed et al. (2022) design tariff.
# Florida plants (Pearson Table 2) do not report a plant-specific tariff and
# are therefore used as a HELD-OUT TEST SET, not for fitting.
REPORTED_TARIFF_USD_KWH = {
    "NAWSC Doolittle (TX)": 0.069, "NAWSC Owassa (TX)": 0.059,
    "Fort Hancock WCID (TX)": 0.082, "Roscoe (TX)": 0.070,
    "Kay Bailey Hutchinson (TX)": 0.0835, "North Cameron Regional A (TX)": 0.080,
    "North Cameron Regional B (TX)": 0.080, "Southmost (TX)": 0.0749,
    "NAWSC Lasara (TX)": 0.072,
    "El Alamein pilot (Egypt, brackish)": 0.039,
    "Nile Delta ADW ZLD (Egypt, agricultural drainage)": 0.070,
}
 
# Measured / derived specific energy (kWh/m3) where the source reports it.
# El Alamein: mean of measured 3.68-4.20 (Dawoud et al. 2024).
# ADW ZLD: 1.47e8 kWh/yr over 0.9 x 300,000 x 365 m3/yr (El Sayed et al. 2022, Table 8).
MEASURED_SEC_KWH_M3 = {
    "El Alamein pilot (Egypt, brackish)": 3.94,
    "Nile Delta ADW ZLD (Egypt, agricultural drainage)": 1.49,
}
 
# Recovery used when running the framework's own SEC estimate for a plant
# whose source does not report recovery (uniform mid-range assumption).
DEFAULT_RECOVERY = 0.75
REPORTED_RECOVERY = {"El Alamein pilot (Egypt, brackish)": 0.40}
 
 
def summary_stats():
    import numpy as np
    caps = np.array([p["capacity_m3d"] for p in REAL_BWRO_PLANTS])
    opex = np.array([p["opex_usd_m3"] for p in REAL_BWRO_PLANTS])
    tds = np.array([p["tds_mg_l"] for p in REAL_BWRO_PLANTS])
    print(f"n = {len(REAL_BWRO_PLANTS)} real plants")
    print(f"Capacity range: {caps.min():,.0f} - {caps.max():,.0f} m3/d")
    print(f"TDS range: {tds.min():,.0f} - {tds.max():,.0f} mg/L")
    print(f"OPEX range: ${opex.min():.2f} - ${opex.max():.2f}/m3")
 
 
if __name__ == "__main__":
    summary_stats()
 