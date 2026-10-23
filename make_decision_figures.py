"""Figures for the re-optimisation (fig_reopt.png) and value-of-information (fig_voi.png) analyses."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
B, O, G, P, K, S, Y = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#222222", "#56B4E9", "#E69F00"
def tag(ax, s): ax.text(-0.13, 1.05, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
NAMES = {"el_moghra_aquifer": "El\nMoghra", "nile_delta_shallow": "Nile\nDelta", "western_desert_nubian": "Western\nDesert",
         "sinai_coastal": "Sinai", "agricultural_drainage_nile_delta": "Drainage"}

# ---------------- re-optimisation
Ro = json.load(open("analysis_reoptimise_element_sec.json"))
fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.9))
for lab, c, m, name in (("lumped", O, "o", "Lumped energy model"), ("element", B, "s", "Element-by-element energy")):
    run = Ro["el_moghra_aquifer"][lab][0]
    F = np.array(run["front"]["F"])
    ax[0].scatter(F[:, 2], F[:, 0], s=10, color=c, marker=m, alpha=0.75, lw=0, label=name)
    ch = run["cheapest"]
    ax[0].scatter([ch["pond_ha"]], [ch["lcow"]], s=70, facecolor="none", edgecolor=c, lw=1.4, zorder=4)
    if lab == "lumped":
        ax[0].annotate(f"cheapest: {ch['recovery'] * 100:.0f}% recovery", (ch["pond_ha"], ch["lcow"]), xytext=(1900, 0.703),
                       textcoords="data", fontsize=6.3, color=c, arrowprops=dict(arrowstyle="-", color=c, lw=0.6))
    else:
        ax[0].annotate(f"cheapest: {ch['recovery'] * 100:.0f}% recovery", (ch["pond_ha"], ch["lcow"]), xytext=(8, 6),
                       textcoords="offset points", fontsize=6.3, color=c)
ax[0].set_xlabel("Evaporation-pond area (ha)"); ax[0].set_ylabel("LCOW (USD m$^{-3}$)")
ax[0].legend(frameon=False, loc="upper right"); tag(ax[0], "a")
keys = list(NAMES); x = np.arange(len(keys)); w = 0.38
for i, (lab, c, name) in enumerate((("lumped", O, "Lumped energy model"), ("element", B, "Element-by-element energy"))):
    vals = [np.mean([s["cheapest"]["pond_ha"] for s in Ro[k][lab]]) for k in keys]
    recs = [np.mean([s["cheapest"]["recovery"] for s in Ro[k][lab]]) for k in keys]
    bars = ax[1].bar(x + (i - 0.5) * w, vals, w * 0.92, color=c, label=name)
    for xb, v, r in zip(x + (i - 0.5) * w, vals, recs):
        ax[1].text(xb, v + 40, f"{r * 100:.0f}%", ha="center", fontsize=5.8, color="0.25")
ax[1].set_xticks(x); ax[1].set_xticklabels([NAMES[k] for k in keys], fontsize=6.8)
ax[1].set_ylabel("Pond area of cheapest design (ha)"); ax[1].set_ylim(0, 4000); ax[1].legend(frameon=False, fontsize=6.3, loc="upper right"); tag(ax[1], "b")
fig.tight_layout(); fig.savefig("figs/fig_reopt.png"); plt.close(fig)

# ---------------- value of information
V = json.load(open("analysis_value_of_information.json"))
r = V["results"]["el_moghra_aquifer"]
fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.9))
centres = np.round(np.arange(0.50, 0.901, 0.01), 2) * 100
for lab, c, name in (("lcow_only", O, "Cost = LCOW only"), ("lcow_plus_ponds", B, "Cost = LCOW + pond cost")):
    h = np.array(r[lab]["rstar_hist"], float); h /= h.sum()
    ax[0].bar(centres, h, 0.9, color=c, alpha=0.85, label=name)
ax[0].set_xlabel("Cost-optimal recovery (%)"); ax[0].set_ylabel("Fraction of 20,000 samples")
ax[0].set_xlim(55, 92); ax[0].legend(frameon=False, loc="upper left"); tag(ax[0], "a")
labels = {"capex_mult": "Capital-cost level", "opex_ne": "Non-energy O&M cost", "pond_cost": "Pond cost per m$^2$",
          "discount": "Discount rate", "capex_exp": "Capacity exponent", "tariff": "Electricity tariff",
          "evap": "Net evaporation", "sec_model": "Energy model", "dummy": "Dummy (noise floor)"}
s1 = r["lcow_plus_ponds"]["S1_cost"]
order = sorted(s1, key=lambda k: s1[k])
cols = [("0.6" if k == "dummy" else B) for k in order]
ax[1].barh(range(len(order)), [s1[k] for k in order], color=cols, height=0.7)
ax[1].set_yticks(range(len(order))); ax[1].set_yticklabels([labels[k] for k in order], fontsize=6.8)
ax[1].set_xlabel("First-order sensitivity index of cost of water"); ax[1].set_xlim(0, 0.32); tag(ax[1], "b")
fig.tight_layout(); fig.savefig("figs/fig_voi.png"); plt.close(fig)
print("saved figs/fig_reopt.png, figs/fig_voi.png")
