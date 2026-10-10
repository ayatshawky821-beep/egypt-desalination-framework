"""Verification tests for the desalination design framework (run: pytest -q tests)."""
import math, os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from desalination_plant_design import (FeedWaterQuality, PlantCapacity, PlantFlows, default_recovery_for,
    osmotic_pressure_bar, groundwater_lift_sec_kwh_m3, ReverseOsmosisSystem, EconomicAnalysis,
    EconomicAssumptions, DesalinationPlantDesign, EGYPT_BRACKISH_PRESETS, BrineManagementSystem)

def ro(cap=20000., tds=5000., rec=0.80, wt="brackish", head=None):
    f = PlantFlows.from_capacity(PlantCapacity(permeate_flow_m3d=cap, recovery=rec))
    return f, ReverseOsmosisSystem(flow_m3d=f.feed_to_membranes_m3d, permeate_m3d=f.permeate_m3d,
        concentrate_m3d=f.concentrate_m3d, feed_tds_mg_l=tds, water_type=wt, intake_m3d=f.intake_m3d, pumping_head_m=head).design()

def sec_mid(d):
    lo, hi = [float(x) for x in d["specific_energy_consumption_kwh_m3_est"].split("-")]; return (lo + hi) / 2

# ---- mass balance ----
@pytest.mark.parametrize("cap,rec", [(1000, .45), (20000, .8), (150000, .85), (500000, .75)])
def test_flow_balance(cap, rec):
    f = PlantFlows.from_capacity(PlantCapacity(permeate_flow_m3d=cap, recovery=rec))
    assert f.permeate_m3d + f.concentrate_m3d == pytest.approx(f.feed_to_membranes_m3d)
    assert f.permeate_m3d / f.feed_to_membranes_m3d == pytest.approx(rec)
    assert f.intake_m3d * (1 - f.pretreatment_loss_frac) == pytest.approx(f.feed_to_membranes_m3d)

@pytest.mark.parametrize("tds,rec,wt", [(2200, .85, "brackish"), (5000, .8, "brackish"), (35000, .45, "seawater")])
def test_salt_balance(tds, rec, wt):
    f, d = ro(tds=tds, rec=rec, wt=wt)
    salt_in = f.feed_to_membranes_m3d * tds
    salt_out = f.permeate_m3d * d["permeate_tds_mg_l_est"] + f.concentrate_m3d * d["concentrate_tds_mg_l_est"]
    assert salt_out == pytest.approx(salt_in, rel=1e-3)

# ---- known values ----
def test_osmotic_pressure():
    assert osmotic_pressure_bar(10000) == pytest.approx(8.0)

def test_recovery_rule():
    mk = lambda t, w="brackish": FeedWaterQuality(water_type=w, tds_mg_l=t)
    assert default_recovery_for(mk(35000, "seawater")) == 0.45
    assert default_recovery_for(mk(3000)) == 0.85 and default_recovery_for(mk(3001)) == 0.80
    assert default_recovery_for(mk(6000)) == 0.80 and default_recovery_for(mk(6001)) == 0.75

def test_lift_closed_form():
    # 1 m3/d lifted 100 m at eta=0.65 -> rho*g*H/3.6e6/eta = 0.419 kWh per m3 of intake
    v = groundwater_lift_sec_kwh_m3(1.0, 1.0, 100.0)
    assert v == pytest.approx(1000 * 9.81 * 100 / 3.6e6 / 0.65, rel=1e-9)
    assert groundwater_lift_sec_kwh_m3(1.0, 1.0, None) == 0.0

def test_crf_and_lcow_hand_calc():
    a = EconomicAssumptions()
    e = EconomicAnalysis(capacity_m3d=20000., water_type="brackish", specific_energy_kwh_m3=1.0, assumptions=a).run_full_analysis()
    crf = 0.10 * 1.10 ** 20 / (1.10 ** 20 - 1)
    assert e["lcow"]["capital_recovery_factor"] == pytest.approx(crf, abs=1e-4)
    capex = 870 * 20000
    prod = 20000 * 365 * 0.9
    assert e["lcow"]["lcow_usd_m3"] == pytest.approx(crf * capex / prod + 0.045 + 0.40, abs=2e-3)

def test_scale_factor_clip():
    a = EconomicAssumptions(economies_of_scale_exponent=-0.5)
    small = EconomicAnalysis(capacity_m3d=1.0, water_type="brackish", specific_energy_kwh_m3=1, assumptions=a)
    assert small._scale_factor(-0.5) == 2.0
    big = EconomicAnalysis(capacity_m3d=1e9, water_type="brackish", specific_energy_kwh_m3=1, assumptions=a)
    assert big._scale_factor(-0.5) == 0.5

# ---- monotonicity ----
def test_sec_increases_with_salinity():
    s = [sec_mid(ro(tds=t)[1]) for t in (1000, 3000, 5000, 8000, 12000)]
    assert all(b > a for a, b in zip(s, s[1:]))

def test_sec_increases_with_recovery_brackish():
    s = [sec_mid(ro(rec=r)[1]) for r in (0.6, 0.7, 0.8, 0.85)]
    assert all(b > a for a, b in zip(s, s[1:]))

def test_sec_increases_with_head():
    s = [sec_mid(ro(head=h)[1]) for h in (0, 30, 90, 180)]
    assert all(b > a for a, b in zip(s, s[1:]))

def test_lcow_increases_with_tariff_and_opex():
    base = lambda **k: EconomicAnalysis(capacity_m3d=20000., water_type="brackish", specific_energy_kwh_m3=1.5,
        assumptions=EconomicAssumptions(**k)).run_full_analysis()["lcow"]["lcow_usd_m3"]
    assert base(electricity_tariff_usd_kwh=0.08) > base(electricity_tariff_usd_kwh=0.04)
    assert base(opex_nonenergy_brackish=0.6) > base(opex_nonenergy_brackish=0.3)
    assert base(discount_rate=0.14) > base(discount_rate=0.06)

def test_pond_area_inverse_to_evaporation():
    mk = lambda ev: None
    b = BrineManagementSystem(concentrate_flow_m3d=5000., concentrate_tds_mg_l=25000., feed_tds_mg_l=5000.,
        is_coastal_site=False, is_arid_climate=True).run_full_design()
    area = b["evaporation_pond_option"]["required_pond_area_m2"]
    assert area == pytest.approx(5000 * 365 / 2.2, abs=1)

def test_seawater_uses_erd_brackish_does_not():
    _, sw = ro(tds=35000, rec=.45, wt="seawater"); _, bw = ro(tds=5000)
    assert "Isobaric" in sw["energy_recovery_device"] and "Not usually" in bw["energy_recovery_device"]
    assert 2.0 < sec_mid(sw) < 6.0

# ---- end-to-end ----
@pytest.mark.parametrize("key", list(EGYPT_BRACKISH_PRESETS))
def test_full_build_runs_and_is_sane(key):
    feed = EGYPT_BRACKISH_PRESETS[key]
    d = DesalinationPlantDesign(feed=feed, capacity=PlantCapacity(20000., default_recovery_for(feed)),
        is_coastal_site=feed.water_type == "seawater").build()
    l = d["economic_analysis"]["lcow"]["lcow_usd_m3"]
    assert 0.3 < l < 3.0

def test_el_moghra_reference_value():
    f = EGYPT_BRACKISH_PRESETS["el_moghra_aquifer"]
    d = DesalinationPlantDesign(feed=f, capacity=PlantCapacity(20000., 0.80), is_coastal_site=False, is_arid_climate=True).build()
    assert d["economic_analysis"]["lcow"]["lcow_usd_m3"] == pytest.approx(0.794, abs=0.001)
