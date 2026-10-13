"""Figure for the element-by-element RO cross-check (reads analysis_element_level_ro.json)."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
B, O, G, P, K, S, Y = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#222222", "#56B4E9", "#E69F00"
def tag(ax, s): ax.text(-0.13, 1.05, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")

D = json.load(open("analysis_element_level_ro.json"))
rows = D["rows"]
feeds = [("el_moghra_aquifer", "El Moghra, 5 g L$^{-1}$", B), ("nile_delta_shallow", "Nile Delta, 2.2 g L$^{-1}$", O)]
fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.9))
for key, lab, c in feeds:
    rr = [r for r in rows if r["feed"] == key]
    rec = [r["recovery"] * 100 for r in rr]
    ax[0].plot(rec, [r["fw_sec"] for r in rr], "-o", color=c, lw=1.6, ms=3.5, label=f"{lab}: framework")
    ax[0].plot(rec, [r["sec_kwh_m3"] for r in rr], "--s", color=c, lw=1.3, ms=3.5, label=f"{lab}: element model")
    ax[0].plot(rec, [r["noboost_sec"] for r in rr], ":", color=c, lw=1.0, label=f"{lab}: element, no booster")
ax[0].set_xlabel("Recovery (%)"); ax[0].set_ylabel("SEC, membrane train (kWh m$^{-3}$)")
ax[0].set_xticks([75, 80, 85, 90]); ax[0].set_ylim(0, 2.75); ax[0].legend(frameon=False, loc="upper left", fontsize=5.4); tag(ax[0], "a")

names = {"el_moghra_aquifer": ("El Moghra", B), "nile_delta_shallow": ("Nile Delta", O), "western_desert_nubian": ("Western Desert", G),
         "sinai_coastal": ("Sinai", P), "design_study_ro_feed": ("Drainage (design-study feed)", Y)}
for key, (lab, c) in names.items():
    rr = [r for r in rows if r["feed"] == key]
    ok = [r for r in rr if r["within_41bar"]]; bad = [r for r in rr if not r["within_41bar"]]
    ax[1].scatter([r["sec_kwh_m3"] for r in ok], [r["fw_sec"] for r in ok], s=22, color=c, edgecolor="k", lw=0.3, label=lab, zorder=3)
    if bad:
        ax[1].scatter([r["sec_kwh_m3"] for r in bad], [r["fw_sec"] for r in bad], s=22, facecolor="white", edgecolor=c, lw=1.1, zorder=3)
lim = [0.3, 2.9]
ax[1].plot(lim, lim, color="0.4", lw=0.8); ax[1].text(2.15, 2.3, "1:1", color="0.4", fontsize=6.5)
ax[1].plot(lim, [1.5 * x for x in lim], color="0.6", lw=0.6, ls="--"); ax[1].text(1.55, 2.5, "+50%", color="0.5", fontsize=6.5)
ax[1].set_xlim(0.3, 2.4); ax[1].set_ylim(0.3, 2.9)
ax[1].set_xlabel("SEC, element model (kWh m$^{-3}$)"); ax[1].set_ylabel("SEC, framework (kWh m$^{-3}$)")
ax[1].legend(frameon=False, fontsize=5.8, loc="lower right"); tag(ax[1], "b")
fig.tight_layout(); fig.savefig("figs/fig_element_ro.png"); plt.close(fig)
print("saved figs/fig_element_ro.png")
