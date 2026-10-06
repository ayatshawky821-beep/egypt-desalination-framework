"""
ai_multi_plant_rl.py

Reinforcement learning for the MULTI-PLANT NETWORK design tier of this
paper: given a total regional water demand that a single plant should not
or cannot fully serve (land, aquifer yield, or grid-capacity limits at any
one site), sequentially allocate capacity increments across several
candidate Egyptian brackish-groundwater sites to minimize total system
LCOW, subject to a per-site maximum capacity (aquifer yield constraint)
and a per-site evaporation-pond land budget.

WHY REINFORCEMENT LEARNING, AND WHY THIS PROBLEM SPECIFICALLY: the other
AI methods in this paper (NSGA-II, Bayesian optimization, deep learning
surrogates) treat each design instance as a single-shot, fixed-dimension
optimization. Regional allocation is naturally SEQUENTIAL -- capacity is
committed to sites one increment at a time, each decision changes which
options remain open (a site nearer its yield ceiling becomes less
attractive, land budget is consumed), and the problem has a natural
terminal condition (total demand met). This is a textbook fit for a
Markov Decision Process, solved here with REINFORCE (Williams, 1992), a
foundational policy-gradient reinforcement learning algorithm: a neural
network policy maps the current allocation state to a probability
distribution over "which site gets the next increment," updated using the
episode's total (negative) cost as the training signal.

BASELINES: a greedy heuristic (always allocate to whichever site has the
lowest marginal LCOW for its next increment, subject to feasibility) and a
uniform-random feasible policy, both evaluated on the same episodes, so
the RL agent's value is demonstrated by comparison rather than asserted.

Run:
    python3 ai_multi_plant_rl.py
"""

import numpy as np
import matplotlib.pyplot as plt

from desalination_plant_design import (
    FeedWaterQuality, PlantCapacity, PlantFlows, ReverseOsmosisSystem,
    EconomicAnalysis, BrineManagementSystem, EGYPT_BRACKISH_PRESETS,
)

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10.5})

# --- Regional network definition: 4 candidate Egyptian brackish sites ---
SITE_KEYS = ["el_moghra_aquifer", "nile_delta_shallow", "western_desert_nubian", "sinai_coastal"]
SITE_MAX_CAPACITY_M3D = {  # illustrative per-site aquifer-yield ceilings
    "el_moghra_aquifer": 120_000.0,
    "nile_delta_shallow": 90_000.0,
    "western_desert_nubian": 80_000.0,
    "sinai_coastal": 60_000.0,
}
SITE_POND_BUDGET_HA = {  # calibrated to ~70% of each site's pond area at its own capacity
    "el_moghra_aquifer": 350.0,      # ceiling requires ~498 ha at full 120,000 m3/day
    "nile_delta_shallow": 180.0,     # ceiling requires ~263 ha at full 90,000 m3/day
    "western_desert_nubian": 165.0,  # ceiling requires ~234 ha at full 80,000 m3/day
    "sinai_coastal": 230.0,          # ceiling requires ~332 ha at full 60,000 m3/day (highest pond intensity per m3)
}
TOTAL_REGIONAL_DEMAND_M3D = 250_000.0  # exceeds any single site's ceiling: a genuine multi-plant problem
INCREMENT_M3D = 10_000.0
N_SITES = len(SITE_KEYS)
# Deliberately less than enough increments to saturate every site's own
# ceiling+pond budget (which together would admit 23 increments = 230,000
# m3/day): a 15-increment phased build-out budget (a realistic rollout/
# CAPEX-phasing constraint) forces genuine prioritization among sites,
# rather than a problem where every policy eventually fills every site
# and the allocation order stops mattering.
N_STEPS = 15


# ===========================================================================
# 1. Environment: marginal LCOW and feasibility for the next increment at
#    each site, given its current committed capacity.
# ===========================================================================

_marginal_cost_cache = {}


def marginal_lcow(site_key, current_capacity_m3d, increment=INCREMENT_M3D):
    """LCOW (USD/m3) of operating site_key at (current + increment), used
    as the marginal cost signal for allocating the next increment there.
    Recovery fixed at each preset's framework default for simplicity (the
    per-site recovery was already separately optimized in the companion
    single-plant and multi-objective studies; here the focus is on
    cross-site allocation, not re-optimizing recovery jointly)."""
    new_capacity = current_capacity_m3d + increment
    key = (site_key, round(new_capacity))
    if key in _marginal_cost_cache:
        return _marginal_cost_cache[key]

    feed = EGYPT_BRACKISH_PRESETS[site_key]
    from desalination_plant_design import default_recovery_for
    recovery = default_recovery_for(feed)
    cap = PlantCapacity(permeate_flow_m3d=new_capacity, recovery=recovery)
    flows = PlantFlows.from_capacity(cap)
    ro = ReverseOsmosisSystem(flow_m3d=flows.feed_to_membranes_m3d, permeate_m3d=flows.permeate_m3d,
                               concentrate_m3d=flows.concentrate_m3d, feed_tds_mg_l=feed.tds_mg_l,
                               water_type=feed.water_type).design()
    sec_lo, sec_hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
    econ = EconomicAnalysis(capacity_m3d=new_capacity, water_type=feed.water_type,
                             specific_energy_kwh_m3=(sec_lo + sec_hi) / 2).run_full_analysis()
    lcow = econ["lcow"]["lcow_usd_m3"]

    pond_ha = BrineManagementSystem(concentrate_flow_m3d=flows.concentrate_m3d,
                                     concentrate_tds_mg_l=ro["concentrate_tds_mg_l_est"],
                                     feed_tds_mg_l=feed.tds_mg_l, is_coastal_site=False,
                                     is_arid_climate=True).design_evaporation_ponds_alternative() \
        .design["evaporation_pond_option"]["required_pond_area_hectares"]

    feasible = (new_capacity <= SITE_MAX_CAPACITY_M3D[site_key]) and (pond_ha <= SITE_POND_BUDGET_HA[site_key])
    _marginal_cost_cache[key] = (lcow, feasible)
    return lcow, feasible


def feasible_sites(allocation):
    feas = []
    for site_key in SITE_KEYS:
        _, ok = marginal_lcow(site_key, allocation[site_key])
        if ok:
            feas.append(site_key)
    return feas


def total_system_lcow(allocation):
    """Capacity-weighted average LCOW across all committed sites."""
    total_cap = sum(allocation.values())
    if total_cap == 0:
        return 0.0
    weighted = 0.0
    for site_key, cap in allocation.items():
        if cap == 0:
            continue
        feed = EGYPT_BRACKISH_PRESETS[site_key]
        from desalination_plant_design import default_recovery_for
        recovery = default_recovery_for(feed)
        pcap = PlantCapacity(permeate_flow_m3d=cap, recovery=recovery)
        flows = PlantFlows.from_capacity(pcap)
        ro = ReverseOsmosisSystem(flow_m3d=flows.feed_to_membranes_m3d, permeate_m3d=flows.permeate_m3d,
                                   concentrate_m3d=flows.concentrate_m3d, feed_tds_mg_l=feed.tds_mg_l,
                                   water_type=feed.water_type).design()
        sec_lo, sec_hi = [float(x) for x in ro["specific_energy_consumption_kwh_m3_est"].split("-")]
        econ = EconomicAnalysis(capacity_m3d=cap, water_type=feed.water_type,
                                 specific_energy_kwh_m3=(sec_lo + sec_hi) / 2).run_full_analysis()
        weighted += econ["lcow"]["lcow_usd_m3"] * cap
    return weighted / total_cap


# ===========================================================================
# 2. REINFORCE policy-gradient agent (from scratch: numpy only)
# ===========================================================================

class PolicyNetwork:
    """A minimal 1-hidden-layer softmax policy: state (4 normalized
    allocation fractions + fraction of demand remaining) -> action
    probabilities over which site receives the next increment."""

    def __init__(self, state_dim, n_actions, hidden=16, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.3, size=(state_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.3, size=(hidden, n_actions))
        self.b2 = np.zeros(n_actions)

    def forward(self, state):
        h = np.tanh(state @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return probs, h

    def act(self, state, rng, action_mask=None):
        probs, h = self.forward(state)
        if action_mask is not None:
            probs = probs * action_mask
            if probs.sum() == 0:
                probs = action_mask / action_mask.sum()
            else:
                probs /= probs.sum()
        action = rng.choice(len(probs), p=probs)
        return action, probs, h

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def grads_log_prob(self, state, h, probs, action):
        """Analytic gradient of log pi(action|state) w.r.t. all parameters."""
        dlogits = -probs.copy()
        dlogits[action] += 1.0
        dW2 = np.outer(h, dlogits)
        db2 = dlogits
        dh = dlogits @ self.W2.T
        dtanh = (1 - h ** 2) * dh
        dW1 = np.outer(state, dtanh)
        db1 = dtanh
        return [dW1, db1, dW2, db2]

    def apply_grads(self, grads, lr):
        for p, g in zip(self.params(), grads):
            p += lr * g


def make_state(allocation):
    fracs = [allocation[k] / SITE_MAX_CAPACITY_M3D[k] for k in SITE_KEYS]
    remaining_frac = 1 - sum(allocation.values()) / TOTAL_REGIONAL_DEMAND_M3D
    return np.array(fracs + [remaining_frac])


def run_episode(policy, rng, greedy=False, random_policy=False):
    allocation = {k: 0.0 for k in SITE_KEYS}
    trajectory = []
    for step in range(N_STEPS):
        mask = np.array([1.0 if marginal_lcow(k, allocation[k])[1] else 0.0 for k in SITE_KEYS])
        if mask.sum() == 0:
            break  # no feasible site left for the next increment

        if greedy:
            costs = np.array([marginal_lcow(k, allocation[k])[0] if mask[i] else np.inf
                               for i, k in enumerate(SITE_KEYS)])
            action = int(np.argmin(costs))
            probs, h = None, None
        elif random_policy:
            probs = mask / mask.sum()
            action = rng.choice(N_SITES, p=probs)
            h = None
        else:
            state = make_state(allocation)
            action, probs, h = policy.act(state, rng, action_mask=mask)
            trajectory.append((state, h, probs, action))

        allocation[SITE_KEYS[action]] += INCREMENT_M3D

    total_delivered = sum(allocation.values())
    lcow = total_system_lcow(allocation)
    shortfall = TOTAL_REGIONAL_DEMAND_M3D - total_delivered
    reward = -lcow - 0.00002 * shortfall  # penalize undelivered demand
    return allocation, lcow, shortfall, reward, trajectory


def train_reinforce(n_episodes=150, lr=0.05, seed=42):
    rng = np.random.default_rng(seed)
    policy = PolicyNetwork(state_dim=N_SITES + 1, n_actions=N_SITES, hidden=16, seed=seed)
    reward_history, lcow_history = [], []
    baseline = 0.0

    for ep in range(n_episodes):
        allocation, lcow, shortfall, reward, trajectory = run_episode(policy, rng)
        baseline = 0.9 * baseline + 0.1 * reward  # simple moving-average baseline, reduces variance
        advantage = reward - baseline

        for state, h, probs, action in trajectory:
            grads = policy.grads_log_prob(state, h, probs, action)
            policy.apply_grads(grads, lr * advantage)

        reward_history.append(reward)
        lcow_history.append(lcow if shortfall <= 0 else np.nan)

    return policy, reward_history, lcow_history


# ===========================================================================
# 3. Plots
# ===========================================================================

def plot_training_curve(reward_history, filename):
    fig, ax = plt.subplots()
    window = 10
    smoothed = np.convolve(reward_history, np.ones(window) / window, mode="valid")
    ax.plot(reward_history, alpha=0.3, color="tab:blue", label="Episode reward")
    ax.plot(range(window - 1, len(reward_history)), smoothed, color="tab:red", linewidth=2, label=f"{window}-episode moving average")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward (= -LCOW, demand-shortfall penalized)")
    ax.set_title("REINFORCE Training Curve — Multi-Plant Regional Allocation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_allocation_comparison(allocations, labels, filename):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(SITE_KEYS))
    width = 0.25
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i, (alloc, label) in enumerate(zip(allocations, labels)):
        vals = [alloc[k] / 1000 for k in SITE_KEYS]
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ").title() for k in SITE_KEYS], rotation=15, ha="right")
    ax.set_ylabel("Allocated capacity (thousand m3/day)")
    ax.set_title("Regional Capacity Allocation: RL Policy vs. Greedy vs. Random")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


if __name__ == "__main__":
    print(f"Multi-plant regional allocation: {TOTAL_REGIONAL_DEMAND_M3D:,.0f} m3/day total demand "
          f"across {N_SITES} candidate sites, {N_STEPS} increments of {INCREMENT_M3D:,.0f} m3/day each.\n")
    print("Per-site constraints:")
    for k in SITE_KEYS:
        print(f"  {k:<24} max capacity = {SITE_MAX_CAPACITY_M3D[k]:>9,.0f} m3/day   "
              f"pond budget = {SITE_POND_BUDGET_HA[k]:>5.0f} ha")

    print("\nTraining REINFORCE policy...")
    policy, reward_history, lcow_history = train_reinforce(n_episodes=600, lr=0.08)

    rng_eval = np.random.default_rng(999)
    rl_alloc, rl_lcow, rl_shortfall, rl_reward, _ = run_episode(policy, rng_eval)
    greedy_alloc, greedy_lcow, greedy_shortfall, greedy_reward, _ = run_episode(None, rng_eval, greedy=True)

    random_lcows, random_shortfalls = [], []
    for _ in range(20):
        _, r_lcow, r_shortfall, _, _ = run_episode(None, rng_eval, random_policy=True)
        random_lcows.append(r_lcow)
        random_shortfalls.append(r_shortfall)

    print(f"\n{'='*72}\nRESULTS (final evaluation episode)")
    print(f"{'Policy':<20} {'Total LCOW (USD/m3)':<22} {'Demand shortfall (m3/day)'}")
    print(f"{'RL (REINFORCE)':<20} {rl_lcow:<22.4f} {rl_shortfall:.0f}")
    print(f"{'Greedy heuristic':<20} {greedy_lcow:<22.4f} {greedy_shortfall:.0f}")
    print(f"{'Random (mean of 20)':<20} {np.nanmean(random_lcows):<22.4f} {np.mean(random_shortfalls):.0f}")

    print("\nFinal RL allocation:")
    for k in SITE_KEYS:
        print(f"  {k:<24} {rl_alloc[k]:>9,.0f} m3/day")
    print("\nGreedy allocation:")
    for k in SITE_KEYS:
        print(f"  {k:<24} {greedy_alloc[k]:>9,.0f} m3/day")

    plot_training_curve(reward_history, "rl_training_curve.png")
    plot_allocation_comparison(
        [rl_alloc, greedy_alloc, {k: np.mean([run_episode(None, np.random.default_rng(i), random_policy=True)[0][k] for i in range(10)]) for k in SITE_KEYS}],
        ["RL (REINFORCE)", "Greedy heuristic", "Random (mean of 10)"],
        "rl_allocation_comparison.png",
    )

    print("\nReinforcement learning multi-plant allocation complete.")
