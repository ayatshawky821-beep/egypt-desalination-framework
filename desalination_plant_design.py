"""
desalination_plant_design.py
 
Engineering sizing/design tool for an RO/NF desalination plant's
FILTRATION TRAIN (MF / UF / NF / RO), POST-TREATMENT, and
BRINE (CONCENTRATE) MANAGEMENT system.
 
Preliminary/conceptual design calculator (not detailed hydraulic simulation),
using standard rules of thumb (AWWA, Metcalf & Eddy, Voutchkov
"Desalination Engineering", Crittenden "MWH's Water Treatment").
 
Includes built-in feed-water presets for brackish groundwater sites in
Egypt (El Moghra Aquifer, Nile Delta shallow aquifer, Western Desert /
Nubian Sandstone Aquifer, Sinai coastal aquifer) alongside a generic
open-ocean seawater case, so the same code can design either a seawater
RO plant or an inland Egyptian brackish-water RO/NF plant.
 
Run directly for a worked example (Egypt brackish groundwater plant):
    python3 desalination_plant_design.py
"""
 
from dataclasses import dataclass, field
from typing import Dict, Optional
import math
import json
 
 
# --------------------------------------------------------------------------
# 1. INPUT DATA MODELS
# --------------------------------------------------------------------------
 
@dataclass
class FeedWaterQuality:
    """Raw feed water characteristics (defaults = open-intake seawater)."""
    source: str = "Open ocean intake"
    water_type: str = "seawater"          # "seawater" | "brackish" | "agricultural_drainage"
    tds_mg_l: float = 35000.0
    turbidity_ntu: float = 10.0
    sdi15: float = 6.5                    # Silt Density Index of raw water
    temp_c: float = 22.0
    tss_mg_l: float = 15.0
    algae_present: bool = True
    ph: float = 8.1
    hardness_mg_l_caco3: float = 6500.0
    alkalinity_mg_l_caco3: float = 120.0
    boron_mg_l: float = 4.6
    silica_mg_l: float = 5.0
    iron_mg_l: float = 0.1
    # Agricultural-drainage-specific water quality (0 for non-ADW sources)
    cod_mg_l: float = 0.0                 # chemical oxygen demand (organic loading)
    bod_mg_l: float = 0.0                 # biochemical oxygen demand
    nitrate_mg_l: float = 0.0             # NO3 as N or as ion, per source data
    manganese_mg_l: float = 0.0
    # Assumed total dynamic pumping head for raw-water abstraction (m):
    # well static/dynamic water level + drawdown + discharge-piping friction
    # for groundwater sources, or a short lift for a surface/canal-fed
    # intake (agricultural drainage) or shallow seawater intake. This is a
    # PLACEHOLDER pending site data (see paper Appendix A.3 / Section 4.7)
    # and is intentionally None (no lift term applied) unless set below.
    pumping_head_m: Optional[float] = None
 
 
# --- Egypt brackish-groundwater site presets (from published hydrogeochemical
#     surveys of Egypt's aquifer systems: El Moghra, Nile Delta, Nubian
#     Sandstone / Western Desert, Sinai). Representative, not site-specific;
#     always confirm against a real site water analysis before final design.
EGYPT_BRACKISH_PRESETS: Dict[str, FeedWaterQuality] = {
    "el_moghra_aquifer": FeedWaterQuality(
        source="El Moghra Aquifer, west of Nile Delta (Quaternary sand aquifer)",
        water_type="brackish",
        tds_mg_l=5000, turbidity_ntu=3, sdi15=3.5, temp_c=27,
        tss_mg_l=5, algae_present=False, ph=7.6,
        hardness_mg_l_caco3=1400, alkalinity_mg_l_caco3=260,
        boron_mg_l=0.6, silica_mg_l=25, iron_mg_l=0.8,
        pumping_head_m=120,  # mid-upper end of the 30-150 m planning range (deeper preset)
    ),
    "nile_delta_shallow": FeedWaterQuality(
        source="Nile Delta shallow Quaternary aquifer (agricultural fringe)",
        water_type="brackish",
        tds_mg_l=2200, turbidity_ntu=2, sdi15=3.0, temp_c=25,
        tss_mg_l=4, algae_present=False, ph=7.4,
        hardness_mg_l_caco3=900, alkalinity_mg_l_caco3=310,
        boron_mg_l=0.4, silica_mg_l=18, iron_mg_l=0.3,
        pumping_head_m=40,  # shallow fringe wells, low end of the 30-150 m range
    ),
    "western_desert_nubian": FeedWaterQuality(
        source="Nubian Sandstone Aquifer, Western Desert (deep fossil water)",
        water_type="brackish",
        tds_mg_l=1800, turbidity_ntu=1, sdi15=2.5, temp_c=32,
        tss_mg_l=2, algae_present=False, ph=7.2,
        hardness_mg_l_caco3=450, alkalinity_mg_l_caco3=180,
        boron_mg_l=0.3, silica_mg_l=35, iron_mg_l=0.2,
        # Deep fossil Nubian Sandstone wells commonly exceed this paper's
        # 30-150 m planning range by a wide margin; 180 m is a conservative
        # illustrative midpoint, NOT a site-specific value — confirm against
        # an actual well-completion report before use.
        pumping_head_m=180,
    ),
    "sinai_coastal": FeedWaterQuality(
        source="Sinai coastal aquifer (seawater-intrusion influenced)",
        water_type="brackish",
        tds_mg_l=8000, turbidity_ntu=4, sdi15=4.0, temp_c=26,
        tss_mg_l=6, algae_present=False, ph=7.7,
        hardness_mg_l_caco3=2200, alkalinity_mg_l_caco3=220,
        boron_mg_l=1.2, silica_mg_l=20, iron_mg_l=0.4,
        pumping_head_m=50,  # shallow coastal wells, low-mid end of the 30-150 m range
    ),
    "agricultural_drainage_nile_delta": FeedWaterQuality(
        # Real composition from a major Nile Delta drain (Bahr El-Bakar /
        # El-Omoum / Bahr Hadous type), as characterized and treated at
        # 300,000 m3/d scale in El Sayed et al. (2022), Membranes 12(10):923.
        source="Major Nile Delta agricultural drain (e.g. Bahr El-Bakar, El-Omoum, Bahr Hadous)",
        water_type="agricultural_drainage",
        tds_mg_l=2705, turbidity_ntu=100, sdi15=5.5, temp_c=24,
        tss_mg_l=80, algae_present=True, ph=7.7,
        hardness_mg_l_caco3=849,   # from Ca 110 + Mg 140 mg/L (real reported ions)
        alkalinity_mg_l_caco3=303, # from HCO3 370 mg/L reported
        boron_mg_l=0.3, silica_mg_l=10, iron_mg_l=2.0, manganese_mg_l=1.0,
        cod_mg_l=20, bod_mg_l=10, nitrate_mg_l=40,
        # Surface drainage water pumped from a drain/canal pump station, not
        # a deep well — well outside the 30-150 m groundwater range; this is
        # a small illustrative lift, not sourced from the aquifer citations.
        pumping_head_m=15,
    ),
    "red_sea_seawater": FeedWaterQuality(
        source="Red Sea open intake (e.g. Hurghada / Safaga coastal plants)",
        water_type="seawater",
        tds_mg_l=41000, turbidity_ntu=6, sdi15=5.0, temp_c=29,
        tss_mg_l=10, algae_present=True, ph=8.2,
        hardness_mg_l_caco3=7200, alkalinity_mg_l_caco3=130,
        boron_mg_l=5.0, silica_mg_l=3, iron_mg_l=0.05,
        # Shallow open-ocean/beach-well intake lift — not a deep aquifer
        # well; kept small and separate from the brackish groundwater
        # pumping-head discussion in Appendix A.3.
        pumping_head_m=8,
    ),
}
 
 
@dataclass
class PlantCapacity:
    """Plant production target."""
    permeate_flow_m3d: float = 50000.0   # finished (product) water, m3/day
    recovery: float = 0.45               # fraction: seawater RO ~0.40-0.50,
                                          # brackish RO ~0.70-0.85
    design_margin: float = 0.10          # capacity margin for equipment sizing
 
 
def default_recovery_for(feed: FeedWaterQuality) -> float:
    """Rule-of-thumb recovery target by water type/salinity (brackish RO/NF
    can push much higher recovery than seawater RO before scaling limits)."""
    if feed.water_type == "seawater":
        return 0.45
    if feed.tds_mg_l <= 3000:
        return 0.85
    if feed.tds_mg_l <= 6000:
        return 0.80
    return 0.75
 
 
# --------------------------------------------------------------------------
# 2. FLOW BALANCE
# --------------------------------------------------------------------------
 
@dataclass
class PlantFlows:
    permeate_m3d: float
    feed_to_membranes_m3d: float
    concentrate_m3d: float
    intake_m3d: float
    pretreatment_loss_frac: float = 0.03
 
    @classmethod
    def from_capacity(cls, cap: PlantCapacity) -> "PlantFlows":
        feed_to_membranes = cap.permeate_flow_m3d / cap.recovery
        concentrate = feed_to_membranes - cap.permeate_flow_m3d
        loss_frac = 0.03
        intake = feed_to_membranes / (1 - loss_frac)
        return cls(
            permeate_m3d=cap.permeate_flow_m3d,
            feed_to_membranes_m3d=feed_to_membranes,
            concentrate_m3d=concentrate,
            intake_m3d=intake,
            pretreatment_loss_frac=loss_frac,
        )
 
 
# --------------------------------------------------------------------------
# 3. FILTRATION TRAIN SELECTOR (chooses which membrane stages are needed)
# --------------------------------------------------------------------------
 
@dataclass
class FiltrationTrainSelector:
    """Decides which combination of MF / UF / NF / RO is appropriate for the
    given feed water, and whether conventional media filtration can
    substitute for MF/UF."""
    feed: FeedWaterQuality
 
    def recommend(self) -> Dict:
        # --- Particulate pretreatment barrier: MF, UF, or conventional DMF
        if self.feed.sdi15 > 5 or self.feed.algae_present or self.feed.turbidity_ntu > 10:
            barrier = "UF (ultrafiltration) — robust barrier for high/variable turbidity, algae, biofouling risk"
        elif self.feed.turbidity_ntu > 3:
            barrier = "MF (microfiltration) or UF — moderate turbidity, low biofouling risk"
        else:
            barrier = "Conventional dual-media filtration (DMF) sufficient — low turbidity, low SDI groundwater"
 
        # --- Desalting stage: NF alone, NF+RO, or RO alone
        if self.feed.water_type == "brackish" and self.feed.hardness_mg_l_caco3 > 800 and self.feed.tds_mg_l < 4000:
            desalting = ("NF (nanofiltration) as primary desalting/softening stage — "
                         "removes hardness/silica/organics and >90% of divalent+monovalent salts "
                         "at lower pressure/energy than RO; consider NF alone if TDS target is met, "
                         "or NF-then-RO polishing if lower product TDS is required")
        elif self.feed.water_type == "brackish":
            desalting = "Brackish water RO (BWRO), with NF as optional hardness/silica pretreatment stage"
        else:
            desalting = "Seawater RO (SWRO), two-pass if boron or product TDS limits require it"
 
        return {
            "particulate_barrier_recommendation": barrier,
            "desalting_stage_recommendation": desalting,
            "basis": {
                "sdi15": self.feed.sdi15,
                "turbidity_ntu": self.feed.turbidity_ntu,
                "algae_present": self.feed.algae_present,
                "water_type": self.feed.water_type,
                "tds_mg_l": self.feed.tds_mg_l,
                "hardness_mg_l_caco3": self.feed.hardness_mg_l_caco3,
            },
        }
 
 
# --------------------------------------------------------------------------
# 4. PRETREATMENT TRAIN (intake, chemical conditioning, clarification, DMF)
# --------------------------------------------------------------------------
 
@dataclass
class PretreatmentTrain:
    feed: FeedWaterQuality
    flows: PlantFlows
    design: Dict = field(default_factory=dict)
 
    def design_intake(self):
        approach_vel = 0.15  # m/s
        q_m3s = self.flows.intake_m3d / 86400
        screen_area_m2 = q_m3s / approach_vel
        if self.feed.water_type == "brackish":
            intake_type = "Water supply wells (vertical or horizontal), well screens + gravel pack"
        elif self.feed.turbidity_ntu < 5:
            intake_type = "Subsurface (beach well) intake"
        else:
            intake_type = "Open intake with traveling band screens"
        self.design["intake"] = {
            "type": intake_type,
            "coarse_screen_spacing_mm": 50,
            "fine_screen_spacing_mm": 2,
            "traveling_screen_area_m2": round(screen_area_m2, 1) if self.feed.water_type != "brackish" else None,
            "approach_velocity_m_s": approach_vel if self.feed.water_type != "brackish" else None,
            "intake_flow_m3d": round(self.flows.intake_m3d, 0),
        }
        return self
 
    def design_chemical_conditioning(self):
        q = self.flows.intake_m3d
        ferric_dose = 2.0 if self.feed.turbidity_ntu <= 10 else 4.0
        chlorine_dose = 2.0 if self.feed.algae_present else (0.5 if self.feed.iron_mg_l > 0.3 else 0.0)
        sbs_dose = chlorine_dose * 3.3 if chlorine_dose else 0.0
        antiscalant_dose = 2.5 if self.feed.water_type == "seawater" else 3.5  # brackish often higher LSI/silica risk
        conditioning = {
            "coagulant": {
                "chemical": "Ferric chloride (FeCl3)",
                "dose_mg_l": ferric_dose,
                "consumption_kg_d": round(ferric_dose * q / 1000, 1),
                "applicability": "Only if turbidity/TSS or Fe/Mn precipitation requires it",
            },
            "pre_oxidation": {
                "chemical": "Sodium hypochlorite" if chlorine_dose else "Not required",
                "dose_mg_l": chlorine_dose,
                "consumption_kg_d": round(chlorine_dose * q / 1000, 1),
            },
            "dechlorination": {
                "chemical": "Sodium bisulfite (SBS)" if sbs_dose else "Not required",
                "dose_mg_l": round(sbs_dose, 2),
                "consumption_kg_d": round(sbs_dose * q / 1000, 1),
                "location": "Upstream of cartridge filters / membrane feed",
            },
            "antiscalant": {
                "dose_mg_l": antiscalant_dose,
                "consumption_kg_d": round(antiscalant_dose * self.flows.feed_to_membranes_m3d / 1000, 1),
                "purpose": "CaCO3/CaSO4/BaSO4/SiO2 scale inhibition on NF/RO membranes "
                           "(silica scaling is a key risk in Egyptian desert brackish groundwater)",
            },
        }
        if self.feed.iron_mg_l > 0.3:
            conditioning["iron_manganese_removal"] = {
                "process": "Aeration + greensand or catalytic media filtration ahead of membranes",
                "reason": f"Feed iron {self.feed.iron_mg_l} mg/L exceeds membrane fouling threshold (~0.05-0.1 mg/L)",
            }
        if self.feed.water_type == "agricultural_drainage" or self.feed.hardness_mg_l_caco3 > 700:
            # Lime/caustic softening: agricultural drainage water typically carries much higher
            # hardness, organic loading (COD/BOD), and microbiological contamination (algae,
            # coliforms) than natural brackish groundwater at comparable TDS, because it is a
            # mixture of irrigation return flows, shallow groundwater seepage, and some municipal
            #/agro-industrial effluent. NaOH dose is scaled from the real full-scale design of
            # El Sayed et al. (2022): 400 mg/L NaOH achieving ~90% Ca and ~67% Mg removal by
            # precipitation for a feed of comparable hardness (~849 mg/L as CaCO3).
            reference_hardness = 849.0
            reference_naoh_dose = 400.0
            naoh_dose = reference_naoh_dose * (self.feed.hardness_mg_l_caco3 / reference_hardness)
            conditioning["lime_caustic_softening"] = {
                "process": "Chemical softening (NaOH-induced Ca/Mg precipitation) + clarification, "
                           "ahead of dual-media filtration",
                "naoh_dose_mg_l": round(naoh_dose, 0),
                "naoh_consumption_kg_d": round(naoh_dose * q / 1000, 1),
                "expected_hardness_removal_pct": "85-90% Ca, 60-70% Mg (precipitated as CaCO3/Mg(OH)2)",
                "purpose": "Protects downstream NF/RO from severe carbonate/hydroxide scaling and "
                           "substantially raises achievable system recovery",
                "basis": "Calibrated against El Sayed et al. (2022) full-scale (300,000 m3/d) "
                         "agricultural drainage water ZLD design",
                "sludge_note": "CaCO3/Mg(OH)2 sludge is dewatered and may be sold as a by-product "
                               "(as in the reference design) rather than landfilled",
            }
        if self.feed.cod_mg_l > 5 or self.feed.bod_mg_l > 5:
            conditioning["biological_load_note"] = {
                "observed_cod_mg_l": self.feed.cod_mg_l,
                "observed_bod_mg_l": self.feed.bod_mg_l,
                "note": "Organic loading from agricultural drainage (residual fertilizer, crop "
                        "residue, some municipal/agro-industrial effluent) increases biofouling "
                        "risk; chlorination (typical dose 8 mg/L, per real ADW plant design) "
                        "ahead of softening is standard practice, in addition to the RO/NF "
                        "antiscalant dose above",
            }
        self.design["chemical_conditioning"] = conditioning
        return self
 
    def design_clarification(self):
        needs_dmf_only = self.feed.turbidity_ntu <= 10 and self.feed.tss_mg_l <= 20
        q = self.flows.intake_m3d
        if needs_dmf_only:
            stage = "Not required — feed quality allows direct filtration"
        else:
            stage = "Dissolved air flotation (DAF) or lamella clarifier ahead of filtration"
        self.design["clarification"] = {
            "required": not needs_dmf_only,
            "process": stage,
            "design_surface_loading_m_h": 8 if not needs_dmf_only else None,
            "basin_area_m2": round(q / 24 / 8, 1) if not needs_dmf_only else None,
        }
        return self
 
    def design_media_filtration(self):
        q = self.flows.intake_m3d
        filtration_rate = 10.0
        total_area_m2 = (q / 24) / filtration_rate
        n_filters = max(2, math.ceil(total_area_m2 / 40))
        self.design["media_filtration"] = {
            "type": "Dual-media (anthracite/sand) pressure filters",
            "filtration_rate_m_h": filtration_rate,
            "total_filter_area_m2": round(total_area_m2, 1),
            "number_of_filters": n_filters,
            "area_per_filter_m2": round(total_area_m2 / n_filters, 1),
            "backwash_frequency": "Every 24-48 h or at 0.5-0.7 bar headloss",
            "target_effluent_sdi15": "< 4",
        }
        return self
 
    def run_full_design(self):
        (self.design_intake()
             .design_chemical_conditioning()
             .design_clarification()
             .design_media_filtration())
        return self.design
 
 
# --------------------------------------------------------------------------
# 5. MEMBRANE FILTRATION STAGES: MF / UF / NF / RO
# --------------------------------------------------------------------------
 
@dataclass
class MicrofiltrationSystem:
    """MF: 0.1-10 micron, removes suspended solids/bacteria/some protozoa.
    Lower fouling resistance than UF; used where biofouling risk is modest."""
    flow_m3d: float
 
    def design(self) -> Dict:
        flux_lmh = 80.0  # typical MF design flux, L/m2-h
        area_m2 = (self.flow_m3d * 1000) / (flux_lmh * 24)
        return {
            "membrane_type": "Microfiltration (MF), hollow-fiber or spiral, 0.1-10 micron",
            "design_flux_lmh": flux_lmh,
            "membrane_area_m2": round(area_m2, 0),
            "recovery_pct": 95,
            "target_effluent_sdi15": "< 3",
            "removes": ["Turbidity/TSS", "Bacteria", "Some protozoa (Giardia/Crypto)"],
            "does_not_remove": ["Viruses", "Dissolved salts", "Most NOM/color"],
            "cip_frequency": "Weekly to biweekly; daily air/water backwash",
        }
 
 
@dataclass
class UltrafiltrationSystem:
    """UF: 0.01-0.1 micron, tighter than MF — removes viruses and most
    colloids/NOM; the standard robust pretreatment barrier for SWRO."""
    flow_m3d: float
 
    def design(self) -> Dict:
        flux_lmh = 60.0
        area_m2 = (self.flow_m3d * 1000) / (flux_lmh * 24)
        return {
            "membrane_type": "Ultrafiltration (UF), hollow-fiber, outside-in, 0.01-0.04 micron",
            "design_flux_lmh": flux_lmh,
            "membrane_area_m2": round(area_m2, 0),
            "recovery_pct": 92,
            "target_effluent_sdi15": "< 2.5",
            "removes": ["Turbidity/TSS", "Bacteria", "Viruses (>4-log)", "Most colloids/algae"],
            "does_not_remove": ["Dissolved salts", "Most dissolved organics/color"],
            "cip_frequency": "Weekly (chemically enhanced backwash daily)",
        }
 
 
@dataclass
class NanofiltrationSystem:
    """NF: 'loose RO' — rejects >90% of divalent ions (Ca, Mg, SO4, silica)
    and NOM/color, but passes a meaningful fraction of monovalent ions
    (Na, Cl). Operates at ~40-60% of RO pressure. Common as: (a) softening
    pretreatment ahead of RO for scaling control, or (b) a stand-alone
    desalting stage for moderately brackish water where a lower-TDS
    (not necessarily potable-low) product is acceptable."""
    flow_m3d: float
    feed_tds_mg_l: float
    feed_hardness_mg_l_caco3: float
    recovery: float = 0.85
 
    def design(self) -> Dict:
        flux_lmh = 20.0  # NF typical design flux, L/m2-h (brackish groundwater)
        permeate_m3d = self.flow_m3d * self.recovery
        area_m2 = (permeate_m3d * 1000) / (flux_lmh * 24)
        elements = math.ceil(area_m2 / 37)   # 8" element ~37 m2
        vessels = math.ceil(elements / 6)    # 6 elements/vessel typical NF array
        divalent_rejection = 0.97
        monovalent_rejection = 0.50
        # crude blended salt-passage estimate
        blended_rejection = 0.5 * divalent_rejection + 0.5 * monovalent_rejection
        permeate_tds = self.feed_tds_mg_l * (1 - blended_rejection)
        hardness_removed_pct = 96
 
        # NF osmotic-pressure-driven feed pressure: only ~60% of the salt
        # concentration acts osmotically across an NF membrane (partial
        # monovalent-ion passage lowers the effective differential vs RO),
        # and NF needs a lower net driving pressure for its typical flux.
        concentrate_tds_est = self.feed_tds_mg_l / max(1 - self.recovery, 0.05)
        nf_osmotic_bar = 0.6 * osmotic_pressure_bar(concentrate_tds_est)
        ndp_bar, friction_bar = 3.0, 1.0
        feed_pressure_bar = nf_osmotic_bar + ndp_bar + friction_bar
        pressure_lo, pressure_hi = feed_pressure_bar * 0.85, feed_pressure_bar * 1.15
 
        return {
            "membrane_type": "Nanofiltration (NF), spiral wound, ~200-400 Da MWCO",
            "design_flux_lmh": flux_lmh,
            "feed_flow_m3d": round(self.flow_m3d, 0),
            "recovery_pct": round(self.recovery * 100, 0),
            "permeate_flow_m3d_est": round(permeate_m3d, 0),
            "membrane_area_m2": round(area_m2, 0),
            "elements_required": elements,
            "pressure_vessels_6_element": vessels,
            "feed_pressure_bar_est": f"{pressure_lo:.0f}-{pressure_hi:.0f} bar (much lower than RO)",
            "divalent_ion_rejection_pct_est": round(divalent_rejection * 100, 0),
            "monovalent_ion_rejection_pct_est": round(monovalent_rejection * 100, 0),
            "hardness_removal_pct_est": hardness_removed_pct,
            "permeate_tds_mg_l_est": round(permeate_tds, 0),
            "typical_role": "Softening/silica-removal pretreatment to RO, OR stand-alone desalting "
                            "for brackish sources where product TDS of a few hundred to ~1000 mg/L is acceptable",
            "advantage_vs_ro": "30-50% lower operating pressure/energy; retains beneficial minerals "
                               "(partial monovalent passage) — attractive for Egyptian brackish groundwater "
                               "with high hardness/silica but moderate TDS",
        }
 
 
def osmotic_pressure_bar(tds_mg_l: float) -> float:
    """Rough van 't Hoff-type estimate: ~0.8 bar of osmotic pressure per
    1000 mg/L TDS for a typical NaCl-dominated natural water. Adequate for
    conceptual sizing/trend studies, not for final membrane system design."""
    return 0.0008 * tds_mg_l


WELL_PUMP_EFFICIENCY = 0.65  # combined well-pump/motor efficiency (see paper Appendix A.3);
                              # lower than the 0.80 high-pressure-pump efficiency used for
                              # the membrane train, reflecting well pumps' typically lower
                              # efficiency, especially at partial load or as static water
                              # levels decline over a well field's operating life.


def groundwater_lift_sec_kwh_m3(
    intake_m3d: float,
    permeate_m3d: float,
    pumping_head_m: Optional[float],
    well_pump_efficiency: float = WELL_PUMP_EFFICIENCY,
) -> float:
    """Specific energy consumption (kWh per m3 of PERMEATE) attributable to
    lifting raw water from the source to the surface, per the supplementary
    Plift relationship introduced in the paper's Appendix A.3:

        Plift (kW) = rho * g * Q_intake(m3/h) * H_pump(m) / (3.6e6 * eta_well)

    Returns 0.0 if pumping_head_m is None (e.g. a gravity-fed or
    unspecified source), so this term is opt-in per feed-water preset
    rather than silently assumed everywhere.
    """
    if pumping_head_m is None or pumping_head_m <= 0:
        return 0.0
    rho, g = 1000.0, 9.81
    intake_m3h = intake_m3d / 24.0
    plift_kw = rho * g * intake_m3h * pumping_head_m / (3.6e6 * well_pump_efficiency)
    return (plift_kw * 24.0) / permeate_m3d
 
 
@dataclass
class ReverseOsmosisSystem:
    """RO: tight (<1 nm) desalting membrane, rejects >99% of dissolved
    salts. Used as SWRO for seawater or BWRO for brackish water (BWRO runs
    at much lower pressure and higher recovery than SWRO). Feed pressure
    and specific energy are derived from feed/concentrate osmotic pressure
    (a function of TDS and recovery), not a fixed lookup, so both respond
    continuously to changes in feed salinity and recovery target."""
    flow_m3d: float
    permeate_m3d: float
    concentrate_m3d: float
    feed_tds_mg_l: float
    water_type: str = "seawater"
    # Raw-water intake flow (m3/day, post-pretreatment-loss-margin) and the
    # assumed total dynamic pumping head (m) for that intake, used to add a
    # groundwater/source-water lift energy term (see groundwater_lift_sec_kwh_m3
    # and paper Appendix A.3 / Section 4.7). Both optional and default to no
    # lift term, preserving prior behavior if not supplied by the caller.
    intake_m3d: Optional[float] = None
    pumping_head_m: Optional[float] = None
 
    def design(self) -> Dict:
        rejection = 0.996 if self.water_type == "seawater" else 0.985
        permeate_tds = self.feed_tds_mg_l * (1 - rejection) * 1.3
        concentrate_tds = (
            (self.flow_m3d * self.feed_tds_mg_l - self.permeate_m3d * permeate_tds)
            / self.concentrate_m3d
        )
        avg_flux = 15.0 if self.water_type == "seawater" else 22.0  # LMH
        membrane_area_m2 = (self.permeate_m3d * 1000) / (avg_flux * 24)
        elements = math.ceil(membrane_area_m2 / 37)
        vessels = math.ceil(elements / 7)
        recovery = self.permeate_m3d / self.flow_m3d
 
        # --- Pressure required: overcome osmotic pressure at the concentrate
        # (worst-case) end of the vessel, plus a net driving pressure (NDP)
        # for flux, plus friction losses.
        ndp_bar = 10.0 if self.water_type == "seawater" else 6.0
        friction_bar = 2.0 if self.water_type == "seawater" else 1.0
        conc_osmotic_bar = osmotic_pressure_bar(concentrate_tds)
        feed_pressure_bar = conc_osmotic_bar + ndp_bar + friction_bar
        pressure_lo, pressure_hi = feed_pressure_bar * 0.9, feed_pressure_bar * 1.1
 
        if self.water_type == "seawater":
            ero = "Isobaric pressure exchanger, >95% efficiency"
            erd_recovered_fraction = 0.95  # fraction of concentrate-stream hydraulic energy recovered
        else:
            ero = "Not usually justified below ~20 bar feed pressure"
            erd_recovered_fraction = 0.0
 
        # --- Specific energy: hydraulic power = 0.02778 * Q(m3/h) * P(bar) [kW],
        # net of energy recovery device credit on the concentrate stream fraction.
        pump_efficiency = 0.80
        feed_m3h = self.flow_m3d / 24
        conc_fraction = 1 - recovery
        power_kw_no_erd = 0.02778 * feed_m3h * feed_pressure_bar / pump_efficiency
        power_kw = power_kw_no_erd * (1 - erd_recovered_fraction * conc_fraction)
        sec_mid = (power_kw * 24) / self.permeate_m3d  # kWh per m3 permeate (membrane train only)
        sec_lo, sec_hi = sec_mid * 0.9, sec_mid * 1.15

        # --- Groundwater/source-water lift energy (supplementary term, paper
        # Appendix A.3 / Section 4.7). Falls back to 0 if intake_m3d or
        # pumping_head_m were not supplied, so this is purely additive and
        # never changes behavior for callers that don't opt in.
        intake_for_lift = self.intake_m3d if self.intake_m3d is not None else feed_m3h * 24
        sec_lift = groundwater_lift_sec_kwh_m3(
            intake_m3d=intake_for_lift,
            permeate_m3d=self.permeate_m3d,
            pumping_head_m=self.pumping_head_m,
        )
        sec_total_lo, sec_total_hi = sec_lo + sec_lift, sec_hi + sec_lift

        return {
            "membrane_type": f"{'Seawater' if self.water_type=='seawater' else 'Brackish water'} RO (SWRO/BWRO), "
                              "8-inch spiral wound, ~100-400 Da MWCO",
            "array_configuration": "Two-stage (concentrate staged) array for higher system recovery",
            "average_design_flux_lmh": avg_flux,
            "total_membrane_area_m2": round(membrane_area_m2, 0),
            "elements_required": elements,
            "pressure_vessels_7_element": vessels,
            "feed_pressure_bar_est": f"{pressure_lo:.0f}-{pressure_hi:.0f} bar",
            "energy_recovery_device": ero,
            "specific_energy_membrane_train_kwh_m3_est": f"{sec_lo:.2f}-{sec_hi:.2f}",
            "specific_energy_lift_kwh_m3_est": round(sec_lift, 3),
            "pumping_head_m_assumed": self.pumping_head_m,
            "specific_energy_consumption_kwh_m3_est": f"{sec_total_lo:.2f}-{sec_total_hi:.2f}",
            "permeate_tds_mg_l_est": round(permeate_tds, 1),
            "concentrate_tds_mg_l_est": round(concentrate_tds, 0),
            "concentrate_osmotic_pressure_bar_est": round(conc_osmotic_bar, 1),
            "recovery_pct": round(recovery * 100, 1),
        }
 
 
@dataclass
class TwoStageROConcentrateStaging:
    """Concentrate (reject) staging: a second RO stage treats the first
    stage's concentrate to recover additional permeate, raising overall
    system recovery well beyond what a single RO stage can achieve at
    acceptable concentrate-side osmotic pressure. Pattern and default
    stage recoveries (90% / 60%) are taken directly from the full-scale
    (300,000 m3/d) agricultural drainage water design of El Sayed et al.
    (2022), Membranes 12(10):923, rather than assumed from first
    principles, since this is a real, built design pattern rather than a
    generic rule of thumb."""
    feed_flow_m3d: float
    feed_tds_mg_l: float
    stage1_recovery: float = 0.90
    stage2_recovery: float = 0.60
 
    def design(self) -> Dict:
        stage1_permeate = self.feed_flow_m3d * self.stage1_recovery
        stage1_concentrate = self.feed_flow_m3d - stage1_permeate
 
        stage2_feed = stage1_concentrate
        stage2_permeate = stage2_feed * self.stage2_recovery
        stage2_concentrate = stage2_feed - stage2_permeate
 
        total_permeate = stage1_permeate + stage2_permeate
        overall_recovery = total_permeate / self.feed_flow_m3d
 
        # Concentrate TDS rises steeply across both stages; approximate via
        # simple mass balance assuming ~98.5% rejection at each stage.
        rejection = 0.985
        stage1_permeate_tds = self.feed_tds_mg_l * (1 - rejection)
        stage1_conc_tds = ((self.feed_flow_m3d * self.feed_tds_mg_l
                             - stage1_permeate * stage1_permeate_tds) / stage1_concentrate)
        stage2_permeate_tds = stage1_conc_tds * (1 - rejection)
        stage2_conc_tds = ((stage2_feed * stage1_conc_tds
                             - stage2_permeate * stage2_permeate_tds) / stage2_concentrate)
 
        return {
            "design_basis": "El Sayed et al. (2022), Membranes 12(10):923 (full-scale ADW ZLD plant)",
            "stage1": {
                "feed_m3d": round(self.feed_flow_m3d, 0),
                "recovery_pct": self.stage1_recovery * 100,
                "permeate_m3d": round(stage1_permeate, 0),
                "concentrate_m3d": round(stage1_concentrate, 0),
                "concentrate_tds_mg_l_est": round(stage1_conc_tds, 0),
            },
            "stage2_concentrate_polisher": {
                "feed_m3d": round(stage2_feed, 0),
                "recovery_pct": self.stage2_recovery * 100,
                "permeate_m3d": round(stage2_permeate, 0),
                "concentrate_m3d": round(stage2_concentrate, 0),
                "concentrate_tds_mg_l_est": round(stage2_conc_tds, 0),
            },
            "overall_system": {
                "total_permeate_m3d": round(total_permeate, 0),
                "overall_recovery_pct": round(overall_recovery * 100, 1),
                "final_concentrate_m3d": round(stage2_concentrate, 0),
                "final_concentrate_tds_mg_l_est": round(stage2_conc_tds, 0),
            },
            "note": "Stage 2 operates at markedly higher feed pressure than Stage 1 despite its "
                    "much lower flow, because it treats Stage 1's already-concentrated reject "
                    "(28.7 bar vs. 57.0 bar in the reference design) — a real, non-obvious "
                    "engineering consequence of concentrate staging worth flagging to a first-time user.",
        }
 
 
@dataclass
class FiltrationTrainDesign:
    """Assembles the recommended MF/UF/NF/RO combination for the given feed
    and flow balance, running only the stages that are actually recommended,
    while still reporting sizing for all four technologies for comparison."""
    feed: FeedWaterQuality
    flows: PlantFlows
    design: Dict = field(default_factory=dict)
 
    def run_full_design(self) -> Dict:
        selector = FiltrationTrainSelector(self.feed).recommend()
        self.design["technology_selection"] = selector
 
        # Always compute MF and UF sizing for direct comparison (particulate barrier options)
        self.design["microfiltration_MF"] = MicrofiltrationSystem(
            self.flows.intake_m3d
        ).design()
        self.design["ultrafiltration_UF"] = UltrafiltrationSystem(
            self.flows.intake_m3d
        ).design()
 
        # NF: always computed for comparison; sized on the flow feeding the desalting stage
        nf_recovery = 0.85
        self.design["nanofiltration_NF"] = NanofiltrationSystem(
            flow_m3d=self.flows.feed_to_membranes_m3d,
            feed_tds_mg_l=self.feed.tds_mg_l,
            feed_hardness_mg_l_caco3=self.feed.hardness_mg_l_caco3,
            recovery=nf_recovery,
        ).design()
 
        # RO: primary desalting stage sized on full feed/permeate/concentrate flows.
        # intake_m3d and pumping_head_m feed the supplementary groundwater-lift
        # energy term (Appendix A.3 / Section 4.7); pumping_head_m is None for
        # presets that haven't had a lift assumption set, which keeps this a
        # strictly additive, opt-in change.
        self.design["reverse_osmosis_RO"] = ReverseOsmosisSystem(
            flow_m3d=self.flows.feed_to_membranes_m3d,
            permeate_m3d=self.flows.permeate_m3d,
            concentrate_m3d=self.flows.concentrate_m3d,
            feed_tds_mg_l=self.feed.tds_mg_l,
            water_type=self.feed.water_type,
            intake_m3d=self.flows.intake_m3d,
            pumping_head_m=self.feed.pumping_head_m,
        ).design()
 
        # Cartridge filtration — final 5-micron safety barrier before high-pressure pumps
        q_m3h = self.flows.feed_to_membranes_m3d / 24
        design_flux_m3_h_per_10in = 1.5
        n_cartridges = math.ceil(q_m3h / design_flux_m3_h_per_10in)
        self.design["cartridge_filtration"] = {
            "rating_micron": 5,
            "type": "Melt-blown polypropylene, 10-inch equivalents",
            "cartridges_required": n_cartridges,
            "replacement_frequency": "1-3 months (monitor dP, replace at 1.0-1.5 bar)",
        }
        return self.design
 
 
# --------------------------------------------------------------------------
# 6. POST-TREATMENT
# --------------------------------------------------------------------------
 
@dataclass
class PostTreatmentTrain:
    permeate_flow_m3d: float
    permeate_tds_mg_l: float
    design: Dict = field(default_factory=dict)
 
    def design_remineralization(self):
        q = self.permeate_flow_m3d
        target_hardness = 80
        co2_dose = 25
        lime_dose = 35
        self.design["remineralization"] = {
            "process": "CO2 + lime (Ca(OH)2) dosing, or CO2 + limestone/calcite bed contactor",
            "co2_dose_mg_l": co2_dose,
            "co2_consumption_kg_d": round(co2_dose * q / 1000, 1),
            "lime_dose_mg_l": lime_dose,
            "lime_consumption_kg_d": round(lime_dose * q / 1000, 1),
            "target_hardness_mg_l_caco3": target_hardness,
            "target_alkalinity_mg_l_caco3": 65,
        }
        return self
 
    def design_ph_stabilization(self):
        self.design["ph_stabilization"] = {
            "target_ph": "8.0 - 8.3",
            "dosing_point": "After remineralization, before clearwell",
            "corrosion_index_target": "LSI +0.3 to +0.8 (mildly scale-forming, protective film)",
            "trim_chemical": "NaOH (caustic soda) if fine pH trim needed",
        }
        return self
 
    def design_disinfection(self):
        q = self.permeate_flow_m3d
        cl2_dose = 2.0
        self.design["disinfection"] = {
            "primary": "UV disinfection (if pathogen log-credit required) or none "
                       "(RO/NF already provide strong virus/bacteria removal)",
            "secondary_residual": "Sodium hypochlorite for distribution residual",
            "cl2_dose_mg_l": cl2_dose,
            "cl2_consumption_kg_d": round(cl2_dose * q / 1000, 1),
            "target_residual_mg_l": "0.2-0.5 at point of entry",
        }
        return self
 
    def design_fluoridation_and_polish(self):
        self.design["polishing"] = {
            "fluoridation": "Optional, per Egyptian MoHP drinking water standards",
            "boron_removal_note": "If boron > 0.5 mg/L target (seawater/Sinai sites), add second-pass "
                                   "boron-selective RO stage at elevated pH (9.5-10.5)",
            "corrosion_control": "Confirm compatibility with distribution pipe materials (LSI/AI checks)",
        }
        return self
 
    def run_full_design(self):
        (self.design_remineralization()
             .design_ph_stabilization()
             .design_disinfection()
             .design_fluoridation_and_polish())
        return self.design
 
 
# --------------------------------------------------------------------------
# 7. BRINE (CONCENTRATE) MANAGEMENT
# --------------------------------------------------------------------------
 
@dataclass
class BrineManagementSystem:
    concentrate_flow_m3d: float
    concentrate_tds_mg_l: float
    feed_tds_mg_l: float
    is_coastal_site: bool = True
    is_arid_climate: bool = False
    design: Dict = field(default_factory=dict)
 
    def select_disposal_method(self):
        if self.is_coastal_site:
            method = "Ocean outfall with multiport diffuser (Mediterranean/Red Sea coastal sites)"
        elif self.is_arid_climate and self.concentrate_flow_m3d < 20000:
            method = "Lined solar evaporation ponds — favorable given Egypt's high net evaporation " \
                     "(desert/inland sites: El Moghra, Nile Delta fringe, Western Desert, Sinai interior)"
        elif self.concentrate_flow_m3d < 2000:
            method = "Deep well injection into confined saline aquifer (where geology permits, subject to " \
                      "Ministry of Water Resources and Irrigation approval)"
        else:
            method = "Zero Liquid Discharge (ZLD): brine concentrator + crystallizer"
        self.design["disposal_method_selection"] = {
            "recommended_primary_method": method,
            "site_type": "Coastal" if self.is_coastal_site else "Inland/desert",
            "climate": "Arid (high net evaporation)" if self.is_arid_climate else "Not specified as arid",
            "concentrate_flow_m3d": round(self.concentrate_flow_m3d, 0),
            "concentrate_tds_mg_l": round(self.concentrate_tds_mg_l, 0),
        }
        return self
 
    def design_dilution_and_outfall(self):
        target_delta_tds = 2000
        dilution_water_required = self.concentrate_flow_m3d * (
            (self.concentrate_tds_mg_l - self.feed_tds_mg_l - target_delta_tds) / target_delta_tds
        )
        dilution_water_required = max(dilution_water_required, 0)
        n_ports = max(8, math.ceil(self.concentrate_flow_m3d / 86400 / 0.05))
        self.design["outfall_diffuser"] = {
            "applicability": "Coastal sites only",
            "estimated_dilution_water_m3d": round(dilution_water_required, 0),
            "diffuser_type": "Multiport rosette diffuser, submerged, perpendicular to prevailing current",
            "estimated_diffuser_ports": n_ports,
            "port_exit_velocity_m_s": 3.0,
            "mixing_zone_target": "Background + 10% salinity within regulatory mixing zone boundary",
        }
        return self
 
    def design_zld_alternative(self):
        q = self.concentrate_flow_m3d
        brine_concentrator_recovery = 0.95
        concentrator_product = q * brine_concentrator_recovery
        crystallizer_feed = q * (1 - brine_concentrator_recovery)
        solids_production_kg_d = crystallizer_feed * (self.concentrate_tds_mg_l / 1_000_000) * 1000 * 0.98
        self.design["zld_option"] = {
            "stage_1_brine_concentrator": {
                "type": "Mechanical vapor compression (MVC) brine concentrator",
                "feed_m3d": round(q, 0),
                "recovery_pct": brine_concentrator_recovery * 100,
                "distillate_product_m3d": round(concentrator_product, 0),
                "concentrated_brine_to_crystallizer_m3d": round(crystallizer_feed, 0),
            },
            "stage_2_crystallizer": {
                "type": "Forced-circulation vacuum crystallizer",
                "feed_m3d": round(crystallizer_feed, 1),
                "estimated_dry_solids_kg_d": round(solids_production_kg_d, 0),
                "solids_disposal": "Hauled to licensed landfill or beneficial reuse (industrial salt)",
            },
            "energy_note": "ZLD is highly energy-intensive: 40-60 kWh/m3 of concentrate treated "
                           "vs. 0.5-4 kWh/m3 for the membrane train itself — evaluate vs. evaporation ponds "
                           "given Egypt's high solar/evaporation potential",
        }
        return self
 
    def design_evaporation_ponds_alternative(self):
        q = self.concentrate_flow_m3d
        # Egypt desert net evaporation is high; ~2.0-2.5 m/yr typical for inland/desert sites
        net_evap_m_yr = 2.2
        pond_area_m2 = (q * 365) / net_evap_m_yr
        self.design["evaporation_pond_option"] = {
            "applicability": "Well suited to Egypt's arid inland sites (low land cost, high solar/net evaporation)",
            "assumed_net_evaporation_m_yr": net_evap_m_yr,
            "required_pond_area_m2": round(pond_area_m2, 0),
            "required_pond_area_hectares": round(pond_area_m2 / 10000, 1),
            "liner": "Double HDPE liner with leak detection layer (groundwater protection)",
            "residual_solids_management": "Periodic mechanical removal; possible beneficial reuse (salt/gypsum)",
        }
        return self
 
    def design_monitoring_and_compliance(self):
        self.design["monitoring_and_compliance"] = {
            "permit_authority": "Egyptian Environmental Affairs Agency (EEAA) and Ministry of Water "
                                 "Resources and Irrigation (MWRI) / Holding Company for Water and Wastewater",
            "parameters_monitored": ["Salinity/TDS", "Temperature", "Residual chlorine/oxidants",
                                      "Heavy metals (from antiscalant/coagulant carryover)", "pH", "Dissolved oxygen"],
            "environmental_impact_study": "Required: baseline survey, dispersion modeling (coastal) or "
                                           "soil/groundwater impact assessment (inland ponds/injection)",
        }
        return self
 
    def run_full_design(self):
        (self.select_disposal_method()
             .design_dilution_and_outfall()
             .design_zld_alternative()
             .design_evaporation_ponds_alternative()
             .design_monitoring_and_compliance())
        return self.design
 
 
# --------------------------------------------------------------------------
# 8. ECONOMIC / FINANCIAL ANALYSIS (CAPEX, OPEX, LCOW, PAYBACK, NPV)
# --------------------------------------------------------------------------
#
# Cost benchmarks used below. REVISION NOTE: an earlier version of this module
# used brackish CAPEX = $550/m3/day and non-energy OPEX = $0.12/m3, with a
# single capacity exponent of -0.30 applied to both. Calibration against the
# compiled real-world dataset (real_world_bwro_dataset.py; see
# real_data_calibration.py and Section 6.4 of the paper) showed those
# values were not supported by data:
#   - Non-energy OPEX (chemicals, labor, maintenance), computed from 11
#     plants that report their own tariff, has median $0.40/m3 and shows NO
#     capacity trend (fitted unit-OPEX exponent -0.02, R^2 0.02). The
#     earlier $0.12 was ~3x too low. Applied to 7 held-out Florida plants
#     it gives MAE $0.06/m3 (MAPE 13%) -- but this is no better than a
#     naive constant predictor (MAE $0.067); i.e. the level is right, the
#     plant-to-plant variation is unexplained.
#   - Brackish CAPEX fitted on 11 plants: ~$870/m3/day at 20,000 m3/d with
#     a weak capacity exponent (-0.09, R^2 0.09), nominal (not
#     inflation-adjusted) USD.
#   - Seawater CAPEX $1,300/m3/day comes from Egypt's own 2020-2024 SWRO
#     tenders (real programme data). No SWRO OPEX data were in the
#     calibration set: the seawater non-energy OPEX is therefore set equal
#     to the calibrated brackish value as a neutral assumption and MUST be
#     treated as uncalibrated (the paper reports a 0.20-0.60 sensitivity).
#   - Egypt industrial electricity tariff ~USD 0.045/kWh (EGP ~2.33/kWh).
#
# These are order-of-magnitude planning benchmarks, not a substitute for a
# vendor quotation or detailed bill of quantities.
 
@dataclass
class EconomicAssumptions:
    """Cost and financial assumptions; defaults reflect Egypt, 2025-2026,
    with brackish values calibrated to real plant data (see header note)."""
    capex_per_m3d_seawater: float = 1300.0     # USD per m3/day (Egypt SWRO tenders)
    capex_per_m3d_brackish: float = 870.0      # USD per m3/day at reference capacity (fitted, 11 real plants)
    opex_nonenergy_seawater: float = 0.40      # USD/m3 -- UNCALIBRATED (set = brackish; sensitivity 0.20-0.60)
    opex_nonenergy_brackish: float = 0.40      # USD/m3 -- median of 11 real plants
    electricity_tariff_usd_kwh: float = 0.045  # Egypt industrial tariff, ~EGP 2.33/kWh @ ~52 EGP/USD
    discount_rate: float = 0.10                # annual, typical for Egyptian infrastructure PPPs
    plant_life_years: int = 20
    availability_factor: float = 0.90          # fraction of year at full production
    municipal_tariff_usd_m3: float = 0.05      # heavily subsidized Egyptian potable tariff (illustrative)
    avoided_cost_usd_m3: float = 2.00          # illustrative value of trucked/imported water avoided
    reference_capacity_m3d: float = 20000.0    # capacity at which the CAPEX benchmarks are quoted
    economies_of_scale_exponent: float = -0.09 # CAPEX unit cost ~ (capacity/reference)^exponent (fitted, R^2 0.09)
    opex_scale_exponent: float = 0.0           # non-energy OPEX: no capacity trend found in real data
 
 
@dataclass
class EconomicAnalysis:
    capacity_m3d: float
    water_type: str                # "seawater" | "brackish"
    specific_energy_kwh_m3: float  # from ReverseOsmosisSystem/NanofiltrationSystem output
    assumptions: EconomicAssumptions = field(default_factory=EconomicAssumptions)
    design: Dict = field(default_factory=dict)
 
    def _capex_per_m3d(self) -> float:
        base = (self.assumptions.capex_per_m3d_seawater if self.water_type == "seawater"
                else self.assumptions.capex_per_m3d_brackish)
        return base * self._scale_factor(self.assumptions.economies_of_scale_exponent)
 
    def _opex_nonenergy(self) -> float:
        base = (self.assumptions.opex_nonenergy_seawater if self.water_type == "seawater"
                else self.assumptions.opex_nonenergy_brackish)
        return base * self._scale_factor(self.assumptions.opex_scale_exponent)
 
    def _scale_factor(self, exponent: float) -> float:
        """Sub-linear cost-vs-capacity scaling (Eq. 20), clipped to 0.5-2.0x
        to avoid unrealistic extrapolation far outside the calibration range.
        CAPEX and non-energy OPEX use separate exponents because the real
        data support only a weak CAPEX trend and no OPEX trend."""
        a = self.assumptions
        ratio = self.capacity_m3d / a.reference_capacity_m3d
        return max(0.5, min(2.0, ratio ** exponent))
 
    def capex(self) -> Dict:
        capex_per_m3d = self._capex_per_m3d()
        total_capex = capex_per_m3d * self.capacity_m3d
        self.design["capex"] = {
            "capex_per_m3d_usd": capex_per_m3d,
            "total_capex_usd": round(total_capex, 0),
            "basis": "Egypt SWRO tender benchmark" if self.water_type == "seawater"
                     else "Pearson et al. (2021) BWRO 'normal range'",
        }
        return self
 
    def opex(self) -> Dict:
        a = self.assumptions
        opex_energy = self.specific_energy_kwh_m3 * a.electricity_tariff_usd_kwh
        opex_nonenergy = self._opex_nonenergy()
        opex_total_per_m3 = opex_energy + opex_nonenergy
        annual_production_m3 = self.capacity_m3d * 365 * a.availability_factor
        annual_opex_usd = opex_total_per_m3 * annual_production_m3
        self.design["opex"] = {
            "opex_energy_usd_m3": round(opex_energy, 3),
            "opex_nonenergy_usd_m3": opex_nonenergy,
            "opex_total_usd_m3": round(opex_total_per_m3, 3),
            "annual_production_m3": round(annual_production_m3, 0),
            "annual_opex_usd": round(annual_opex_usd, 0),
        }
        return self
 
    def lcow(self) -> Dict:
        """Levelized Cost of Water: annualized CAPEX (via capital recovery
        factor) plus OPEX, per m3 of water produced (Eq. 20-22)."""
        a = self.assumptions
        i, n = a.discount_rate, a.plant_life_years
        crf = (i * (1 + i) ** n) / ((1 + i) ** n - 1)
        total_capex = self.design["capex"]["total_capex_usd"]
        annualized_capex = crf * total_capex
        annual_production_m3 = self.design["opex"]["annual_production_m3"]
        capex_component_usd_m3 = annualized_capex / annual_production_m3
        lcow = capex_component_usd_m3 + self.design["opex"]["opex_total_usd_m3"]
        self.design["lcow"] = {
            "capital_recovery_factor": round(crf, 4),
            "annualized_capex_usd_yr": round(annualized_capex, 0),
            "capex_component_usd_m3": round(capex_component_usd_m3, 3),
            "opex_component_usd_m3": self.design["opex"]["opex_total_usd_m3"],
            "lcow_usd_m3": round(lcow, 3),
        }
        return self
 
    def roi_and_payback(self) -> Dict:
        """Two illustrative financial scenarios (Eq. 23-25):
        (a) tariff-recovery, using Egypt's heavily subsidized municipal
            potable tariff as revenue; and
        (b) avoided-cost, using the value of trucked/imported water displaced
            in an unserved area as the benefit, net of OPEX.
        Both are simple (undiscounted) payback plus a discounted NPV over
        the plant life for the avoided-cost scenario."""
        a = self.assumptions
        total_capex = self.design["capex"]["total_capex_usd"]
        annual_production_m3 = self.design["opex"]["annual_production_m3"]
        annual_opex_usd = self.design["opex"]["annual_opex_usd"]
 
        # (a) Tariff-recovery scenario
        annual_revenue_tariff = annual_production_m3 * a.municipal_tariff_usd_m3
        annual_net_tariff = annual_revenue_tariff - annual_opex_usd
        payback_tariff_years = (total_capex / annual_net_tariff) if annual_net_tariff > 0 else float("inf")
 
        # (b) Avoided-cost scenario
        annual_benefit_avoided = annual_production_m3 * a.avoided_cost_usd_m3
        annual_net_avoided = annual_benefit_avoided - annual_opex_usd
        payback_avoided_years = (total_capex / annual_net_avoided) if annual_net_avoided > 0 else float("inf")
 
        npv_avoided = -total_capex
        for t in range(1, a.plant_life_years + 1):
            npv_avoided += annual_net_avoided / ((1 + a.discount_rate) ** t)
 
        self.design["roi_payback"] = {
            "tariff_recovery_scenario": {
                "annual_revenue_usd": round(annual_revenue_tariff, 0),
                "annual_net_cashflow_usd": round(annual_net_tariff, 0),
                "simple_payback_years": (round(payback_tariff_years, 1)
                                          if payback_tariff_years != float("inf") else "never (OPEX exceeds tariff revenue)"),
                "note": "Uses Egypt's heavily subsidized municipal tariff; illustrates why public "
                        "capital/PPP financing, not tariff revenue alone, funds most Egyptian plants.",
            },
            "avoided_cost_scenario": {
                "annual_benefit_usd": round(annual_benefit_avoided, 0),
                "annual_net_cashflow_usd": round(annual_net_avoided, 0),
                "simple_payback_years": (round(payback_avoided_years, 1)
                                          if payback_avoided_years != float("inf") else "never"),
                "npv_over_plant_life_usd": round(npv_avoided, 0),
                "note": "Uses an illustrative avoided cost of trucked/imported water in an unserved "
                        "area; representative of remote Sinai/Red Sea/Matrouh sites, not a specific project.",
            },
        }
        return self
 
    def run_full_analysis(self) -> Dict:
        self.capex().opex().lcow().roi_and_payback()
        return self.design
 
 
# --------------------------------------------------------------------------
# 9. MASTER PLANT DESIGN + REPORT
# --------------------------------------------------------------------------
 
@dataclass
class DesalinationPlantDesign:
    feed: FeedWaterQuality = field(default_factory=FeedWaterQuality)
    capacity: PlantCapacity = field(default_factory=PlantCapacity)
    is_coastal_site: bool = True
    is_arid_climate: bool = False
    economics: EconomicAssumptions = field(default_factory=EconomicAssumptions)
 
    def build(self) -> Dict:
        flows = PlantFlows.from_capacity(self.capacity)
        pretreatment = PretreatmentTrain(self.feed, flows).run_full_design()
        filtration = FiltrationTrainDesign(self.feed, flows).run_full_design()
        ro_summary = filtration["reverse_osmosis_RO"]
        post_treatment = PostTreatmentTrain(
            flows.permeate_m3d, ro_summary["permeate_tds_mg_l_est"]
        ).run_full_design()
        brine = BrineManagementSystem(
            concentrate_flow_m3d=flows.concentrate_m3d,
            concentrate_tds_mg_l=ro_summary["concentrate_tds_mg_l_est"],
            feed_tds_mg_l=self.feed.tds_mg_l,
            is_coastal_site=self.is_coastal_site,
            is_arid_climate=self.is_arid_climate,
        ).run_full_design()
        sec_lo, sec_hi = [float(x) for x in ro_summary["specific_energy_consumption_kwh_m3_est"].split("-")]
        economics = EconomicAnalysis(
            capacity_m3d=self.capacity.permeate_flow_m3d,
            water_type=self.feed.water_type,
            specific_energy_kwh_m3=(sec_lo + sec_hi) / 2,
            assumptions=self.economics,
        ).run_full_analysis()
 
        return {
            "plant_capacity_permeate_m3d": self.capacity.permeate_flow_m3d,
            "flow_balance": flows.__dict__,
            "feed_water_quality": self.feed.__dict__,
            "pretreatment_train": pretreatment,
            "filtration_train_MF_UF_NF_RO": filtration,
            "post_treatment_train": post_treatment,
            "brine_management_system": brine,
            "economic_analysis": economics,
        }
 
    def print_report(self):
        result = self.build()
 
        def section(title):
            print("\n" + "=" * 78)
            print(title)
            print("=" * 78)
 
        def dump(d, indent=2):
            print(json.dumps(d, indent=indent, default=str))
 
        section("PLANT OVERVIEW")
        print(f"Permeate (product water) capacity : {result['plant_capacity_permeate_m3d']:,.0f} m3/day")
        dump(result["flow_balance"])
 
        section("1. FEED WATER QUALITY")
        dump(result["feed_water_quality"])
 
        section("2. PRETREATMENT TRAIN")
        dump(result["pretreatment_train"])
 
        section("3. FILTRATION TRAIN — MF / UF / NF / RO COMPARISON & SELECTION")
        dump(result["filtration_train_MF_UF_NF_RO"])
 
        section("4. POST-TREATMENT TRAIN")
        dump(result["post_treatment_train"])
 
        section("5. BRINE / CONCENTRATE MANAGEMENT SYSTEM")
        dump(result["brine_management_system"])
 
        section("6. ECONOMIC / FINANCIAL ANALYSIS (CAPEX, OPEX, LCOW, ROI)")
        dump(result["economic_analysis"])
 
        return result
 
 
# --------------------------------------------------------------------------
# 10. EXAMPLE RUN — Egypt brackish groundwater plant (El Moghra Aquifer)
# --------------------------------------------------------------------------
 
if __name__ == "__main__":
    feed = EGYPT_BRACKISH_PRESETS["el_moghra_aquifer"]
    capacity = PlantCapacity(
        permeate_flow_m3d=20000,
        recovery=default_recovery_for(feed),
    )
 
    plant = DesalinationPlantDesign(
        feed=feed,
        capacity=capacity,
        is_coastal_site=False,
        is_arid_climate=True,
    )
    design_result = plant.print_report()
 
    with open("desalination_design_output.json", "w") as f:
        json.dump(design_result, f, indent=2, default=str)
    print("\nFull design data written to desalination_design_output.json")
 