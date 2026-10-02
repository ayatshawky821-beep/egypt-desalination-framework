# MF/UF/NF/RO Desalination Design Framework — Egypt Case Study

Companion source code for:

> Najjaa, A.S., Ettouney, R.S., El-Rifai, M.A., Tewfik, S., & Elsayed, M.M.
> *MF/UF/NF/RO Desalination Design Framework: A Case Study on Brackish
> Groundwater and Agricultural Drainage Water Desalination in Egypt.*
> (Manuscript in preparation / submission.)

A conceptual/preliminary design calculator for brackish-groundwater and
seawater reverse osmosis (RO) and nanofiltration (NF) desalination plants,
built around five Egyptian feed-water presets (El Moghra Aquifer, Nile
Delta shallow aquifer, Western Desert/Nubian Sandstone Aquifer, Sinai
coastal aquifer, and agricultural drainage water) alongside a generic
open-ocean seawater case. It covers the full treatment train — pretreatment,
MF/UF/NF/RO filtration (including two-stage concentrate staging), post-
treatment, brine/concentrate management, and a capital/operating cost and
return-on-investment layer — plus a parametric sensitivity study, a
real-world 18-plant validation/calibration dataset, and a machine-learning
surrogate model for rapid design-space screening.

## Files

| File | Purpose |
|---|---|
| `desalination_plant_design.py` | Core engineering module: feed-water presets, filtration train sizing (pretreatment through brine management), and the economic analysis (CAPEX/OPEX/LCOW/ROI) classes. Includes the groundwater/source-water lift energy term (`groundwater_lift_sec_kwh_m3`) and per-preset assumed pumping heads. Run directly for a worked example: `python3 desalination_plant_design.py` |
| `egypt_parametric_study.py` | Generates the paper's nine parametric sensitivity figures (SEC vs. TDS, concentrate vs. recovery, evaporation-pond area, NF vs. RO pressure, brackish vs. seawater energy, cross-site comparison, LCOW vs. capacity, payback vs. recovery, LCOW vs. TDS). Run directly to regenerate all figures: `python3 egypt_parametric_study.py` |
| `real_world_bwro_dataset.py` | Curated dataset of 18 real/planned brackish-water RO plants (capacity, TDS, reported or estimated specific energy, CAPEX/OPEX where available) used for validation. |
| `real_data_calibration.py` | Fits the framework's empirical CAPEX-scaling exponent and compares predicted vs. measured OPEX/SEC against the real-plant dataset. |
| `egypt_ml_surrogate.py` | Trains a random-forest surrogate model on synthetic design-space sweeps to screen capacity/TDS/recovery combinations without running the full sizing calculation. |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Worked example: size a 20,000 m3/day El Moghra brackish RO plant
python3 desalination_plant_design.py

# Regenerate all nine parametric-study figures (curve1...curve9.png)
python3 egypt_parametric_study.py

# Compare framework predictions against the real-plant dataset
python3 real_data_calibration.py

# Train and evaluate the ML surrogate model
python3 egypt_ml_surrogate.py
```

## Scope and limitations

This is a **planning-level conceptual design tool**, not a substitute for
detailed hydraulic simulation (e.g., membrane-vendor projection software),
a site-specific pumping test, or a certified water analysis. Several model
inputs are explicitly documented placeholders pending site-specific data —
most notably the assumed groundwater pumping heads (30-150 m range, varying
by preset) and the 2.2 m/year net evaporation rate used for evaporation-
pond sizing. See the accompanying manuscript's Limitations section and
Appendix A for the full governing-equations documentation.

## License

MIT License — see `LICENSE`.

## Citation

See `CITATION.cff`. If you use this code, please cite the accompanying
manuscript (details above) and, where convenient, this software release
via its Zenodo DOI.

## Generative AI disclosure

Large language model assistance (Claude, Anthropic) was used in drafting
portions of this codebase and its documentation, under the direction and
review of the human author(s), who remain solely responsible for its
correctness.
