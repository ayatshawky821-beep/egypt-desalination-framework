import os, json, warnings, numpy as np
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ai_multi_plant_rl import train_reinforce, run_episode, SITE_KEYS

os.makedirs('figs', exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
B, O, G, P, K = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#222222"
def tag(ax, s): ax.text(-0.13, 1.05, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")

A = json.load(open("analysis_a.json")); B2 = json.load(open("analysis_b2.json"))
d = np.load("fronts_seed42.npz"); Fa, Fb = d["Fa"], d["Fb"]
mo = json.load(open("nsga2_mobo_results.json")); rl = json.load(open("rl_multiseed_results.json"))

# ---------------- Figure 1 ----------------
fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
ax[0].scatter(Fb[:, 0], Fb[:, 1], s=7, c=O, alpha=0.45, marker="s", lw=0, label=f"MOBO (n = {len(Fb)})")
ax[0].scatter(Fa[:, 0], Fa[:, 1], s=16, c=B, edgecolor="k", lw=0.3, label=f"NSGA-II, population 40 (n = {len(Fa)})")
ax[0].set_xlabel("LCOW (USD m$^{-3}$)"); ax[0].set_ylabel("SEC (kWh m$^{-3}$)"); ax[0].legend(frameon=False, loc="upper left"); tag(ax[0], "a")
labs = ["pop 20", "pop 40", "pop 80", "pop 100", "seed 0", "seed 1", "seed 42"]
vals = [B2["seed42_pop20_gen50"]["ratio_to_mobo"], B2["seed42_pop40_gen25"]["ratio_to_mobo"], B2["seed42_pop80_gen12"]["ratio_to_mobo"],
        B2["seed42_pop100_gen10"]["ratio_to_mobo"], B2["seed0"]["hv3d_nsga2"]/B2["seed0"]["hv3d_mobo"],
        B2["seed1"]["hv3d_nsga2"]/B2["seed1"]["hv3d_mobo"], B2["seed42_pop40_gen25"]["ratio_to_mobo"]]
x = [0, 1, 2, 3, 5, 6, 7]; cols = [B]*4 + [P]*3
ax[1].bar(x, vals, color=cols, width=0.75)
for xi, v in zip(x, vals): ax[1].text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.5)
ax[1].axhline(1.0, color=O, lw=1.2); ax[1].text(3.55, 1.0025, "MOBO = 1", color=O, ha="center", fontsize=6.5)
ax[1].set_xticks(x); ax[1].set_xticklabels(labs, rotation=35, ha="right"); ax[1].set_ylim(0.90, 1.035)
ax[1].set_ylabel("NSGA-II / MOBO hypervolume\n(3 objectives)"); tag(ax[1], "b")
ax[1].text(1.5, 1.018, "NSGA-II population size\n(seed 42, ~1,040 evaluations)", ha="center", va="center", fontsize=6.3, color=B); ax[1].text(6, 1.018, "population 40,\nthree seeds", ha="center", va="center", fontsize=6.3, color=P)
fig.tight_layout(); fig.savefig("figs/fig1.png"); plt.close(fig)

# ---------------- Figure 2 ----------------
E = A["extrap"]; caps = np.array(E["caps"]) / 1000; ph, sm, lg = map(np.array, (E["physics"], E["small"], E["large"]))
fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.9))
ax[0].axvspan(100, 150, color="0.9", lw=0); ax[0].text(125, ph.min() + 0.0005, "small-range model trained here", rotation=90, ha="center", va="bottom", fontsize=6, color="0.35")
ax[0].plot(caps, ph, color=K, lw=2, label="Physics-based model"); ax[0].plot(caps, sm, "--", color=O, lw=1.5, label="Trained to 150,000 m$^3$ d$^{-1}$")
ax[0].plot(caps, lg, color=P, lw=1.3, label="Trained on 100,000–600,000 m$^3$ d$^{-1}$")
ax[0].set_xlabel("Plant capacity (10$^3$ m$^3$ d$^{-1}$)"); ax[0].set_ylabel("LCOW (USD m$^{-3}$)"); ax[0].set_ylim(0.680, 0.7295); ax[0].legend(frameon=False, loc="upper right", fontsize=6.5); tag(ax[0], "a")
es, el = (sm - ph) / ph * 100, (lg - ph) / ph * 100
ax[1].axvspan(100, 150, color="0.9", lw=0); ax[1].axhline(0, color="0.5", lw=0.8)
ax[1].plot(caps, es, "--", color=O, lw=1.5); ax[1].plot(caps, el, color=P, lw=1.3)
ax[1].set_xlabel("Plant capacity (10$^3$ m$^3$ d$^{-1}$)"); ax[1].set_ylabel("Surrogate error (%)"); tag(ax[1], "b")
fig.tight_layout(); fig.savefig("figs/fig2.png"); plt.close(fig)

# ---------------- Figure 3 ----------------
policy, _, _ = train_reinforce(n_episodes=600, lr=0.08, seed=42)
rl_alloc, rl_l, _, _, _ = run_episode(policy, np.random.default_rng(999))
gr_alloc, gr_l, _, _, _ = run_episode(None, np.random.default_rng(999), greedy=True)
rnd = [run_episode(None, np.random.default_rng(i), random_policy=True)[0] for i in range(10)]
rnd_mean = {k: float(np.mean([r[k] for r in rnd])) for k in SITE_KEYS}
print("RL alloc:", rl_alloc, rl_l); print("Greedy alloc:", gr_alloc, gr_l); print("Random mean alloc:", rnd_mean)
rlv, rav, gl = np.array(rl["rl_lcows"]), np.array(rl["random_lcows"]), rl["greedy_lcow"]
fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.9), gridspec_kw={"width_ratios": [0.8, 1.2]})
bp = ax[0].boxplot([rlv, rav], positions=[1, 2], widths=0.5, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], (B, O)): patch.set_facecolor(c); patch.set_alpha(0.45)
for med in bp["medians"]: med.set_color(K)
rng = np.random.default_rng(1)
for pos, v, c in ((1, rlv, B), (2, rav, O)): ax[0].scatter(pos + rng.uniform(-0.12, 0.12, len(v)), v, s=9, c=c, edgecolor="k", lw=0.2, zorder=3)
ax[0].axhline(gl, color=G, ls="--", lw=1.4); ax[0].text(2.45, gl + 0.0006, f"greedy = {gl:.4f}", color=G, ha="right", fontsize=7)
ax[0].set_xticks([1, 2]); ax[0].set_xticklabels(["REINFORCE", "Random"]); ax[0].set_ylabel("System LCOW (USD m$^{-3}$)"); ax[0].set_xlim(0.5, 2.5); tag(ax[0], "a")
names = ["El Moghra", "Nile Delta", "W. Desert", "Sinai"]; xs = np.arange(4); w = 0.26
for i, (al, c, lab) in enumerate(((rl_alloc, B, "REINFORCE"), (gr_alloc, G, "Greedy"), (rnd_mean, O, "Random (mean)"))):
    ax[1].bar(xs + (i - 1) * w, [al[k] / 1000 for k in SITE_KEYS], w, color=c, label=lab)
ax[1].set_xticks(xs); ax[1].set_xticklabels(names); ax[1].set_ylabel("Allocated capacity (10$^3$ m$^3$ d$^{-1}$)"); ax[1].legend(frameon=False); tag(ax[1], "b")
fig.tight_layout(); fig.savefig("figs/fig3.png"); plt.close(fig)

# ---------------- Figure 4 ----------------
import openpyxl
from external_validation_rosa_egypt import framework_sec
ws = openpyxl.load_workbook("rosa_et_al_2025_source_data.xlsx", data_only=True)["Figure 1"]
rows = list(ws.iter_rows(min_row=2, values_only=True)); sal = np.array([r[0] for r in rows], float); en = np.array([r[1] for r in rows], float)
fig, ax = plt.subplots(1, 2, figsize=(6.7, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})
ax[0].plot(sal, en, color=G, lw=2, label="Rosa et al. (2025) curve")
short = {"el_moghra_aquifer": "El Moghra", "nile_delta_shallow": "Nile Delta", "western_desert_nubian": "W. Desert", "sinai_coastal": "Sinai",
         "agricultural_drainage_nile_delta": "Drainage", "red_sea_seawater": "Red Sea"}
for r in A["presets"]:
    ax[0].scatter(r["ppt"], r["fw"], s=26, facecolor=B if r["inrange"] else "white", edgecolor=B, lw=1.2, zorder=4, label="Model presets" if r["preset"] == "el_moghra_aquifer" else None)
    dx, dy, ha = {"nile_delta_shallow": (0.7, -0.10, "left"), "western_desert_nubian": (0.9, -0.30, "left"), "agricultural_drainage_nile_delta": (0.7, 0.04, "left"),
                  "el_moghra_aquifer": (1.5, -0.12, "left"), "sinai_coastal": (0.5, 0.06, "left"), "red_sea_seawater": (-0.8, -0.28, "right")}[r["preset"]]
    ax[0].text(r["ppt"] + dx, r["fw"] + dy, short[r["preset"]], fontsize=6.3, ha=ha)
ax[0].scatter([2.892], [1.3125], marker="*", s=95, c=K, zorder=5, label="Design study, RO train")
ax[0].scatter([2.892], [1.485], marker="*", s=95, facecolor="none", edgecolor=K, lw=1.0, zorder=5, label="Design study, whole system")
ax[0].scatter([41], [2.7], marker="D", s=60, facecolor="none", edgecolor=K, lw=1.1, zorder=5, label="Hurghada (vendor case study)")
ax[0].set_ylim(0.3, 3.85); ax[0].set_xlabel("Feed-water salinity (ppt)"); ax[0].set_ylabel("SEC (kWh m$^{-3}$)")
ax[0].legend(frameon=False, loc="upper left", fontsize=6.3); tag(ax[0], "a")
recs = np.arange(0.50, 0.9001, 0.025)
sec = [framework_sec(2892, r, 300000, "brackish") for r in recs]
ax[1].plot(recs * 100, sec, color=B, lw=1.8, label="Model (membrane train)")
rv = float(np.interp(2.892, sal, en)); ax[1].axhline(rv, color=G, lw=1.6, label=f"Rosa et al. curve ({rv:.2f})")
ax[1].scatter([90], [1.3125], marker="*", s=95, c=K, zorder=5, label="Design study, RO train (1.31)")
ax[1].set_xlabel("Recovery (%)"); ax[1].set_ylabel("SEC at 2.9 g L$^{-1}$ (kWh m$^{-3}$)"); ax[1].legend(frameon=False, loc="upper left", fontsize=6.5); tag(ax[1], "b")
fig.tight_layout(); fig.savefig("figs/fig4.png"); plt.close(fig)
print("figures written")
