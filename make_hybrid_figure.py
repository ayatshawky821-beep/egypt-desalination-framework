"""Figure for the hybrid-surrogate extrapolation analysis (reads analysis_hybrid_surrogate.json)."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
O, G, S, K = "#D55E00", "#009E73", "#56B4E9", "#222222"
def tag(ax, s): ax.text(-0.13, 1.05, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")

H = json.load(open("analysis_hybrid_surrogate.json"))
L = H["line"]; caps = np.array(L["caps"]) / 1000
fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.9))
ax[0].axvspan(100, 150, color="0.9", lw=0)
ax[0].text(125, 0.7375, "training\nrange", ha="center", va="top", fontsize=6, color="0.35")
ax[0].plot(caps, L["truth"]["nominal"], color=K, lw=2.2, label="Physics-based model")
ax[0].plot(caps, L["truth"]["steeper"], color=K, lw=0.9, ls=":", label="Steeper scaling above 150k")
ax[0].plot(caps, L["truth"]["saturating"], color=K, lw=0.9, ls="-.", label="No scaling above 150k")
ax[0].plot(caps, L["pred"]["GB"], "--", color=O, lw=1.5, label="Gradient boosting")
ax[0].plot(caps, L["pred"]["MLP"], color=S, lw=1.3, label="Neural network")
ax[0].plot(caps, L["pred"]["Hybrid"], color=G, lw=1.5, label="Hybrid (scaling law + GB)")
ax[0].set_xlabel("Plant capacity (10$^3$ m$^3$ d$^{-1}$)"); ax[0].set_ylabel("LCOW (USD m$^{-3}$)")
ax[0].set_ylim(0.635, 0.74); ax[0].legend(frameon=False, loc="lower left", fontsize=5.8, ncol=1); tag(ax[0], "a")

scen = ["nominal", "steeper", "saturating"]; labels = ["As modelled\n(exponent −0.09)", "Steeper\n(−0.20 above 150k)", "No economies\n(0 above 150k)"]
models = [("GB", O, "Gradient boosting"), ("MLP", S, "Neural network"), ("Hybrid", G, "Hybrid")]
x = np.arange(3); w = 0.26
for i, (m, c, lab) in enumerate(models):
    v = [H["summary"][m][s]["mape"][0] for s in scen]; e = [H["summary"][m][s]["mape"][1] for s in scen]
    ax[1].bar(x + (i - 1) * w, v, w * 0.92, color=c, yerr=e, capsize=2, error_kw={"lw": 0.8}, label=lab)
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=6.5)
ax[1].set_ylabel("Mean abs. error, 150k–600k m$^3$ d$^{-1}$ (%)"); ax[1].legend(frameon=False, fontsize=6.5); tag(ax[1], "b")
fig.tight_layout(); fig.savefig("figs/fig_hybrid.png"); plt.close(fig)
print("saved figs/fig_hybrid.png")
