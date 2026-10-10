"""
ai_multi_plant_rl_coupled.py   (v1.2.0)

Coupled version of the four-site capacity-allocation problem.

In ai_multi_plant_rl.py the sites are independent (each has its own aquifer ceiling and
brine-pond budget), so a greedy rule that always funds the cheapest next increment is
near-optimal and reinforcement learning has little to add. Here the sites are COUPLED
through a shared regional brine-disposal land budget:

    sum over sites of required evaporation-pond area  <=  f x (sum of the four site pond budgets)

with f in {1.0 (no coupling), 0.9, 0.8, 0.7, 0.6}. Funding the cheapest site first can use
up the shared land and leave later increments infeasible (unmet demand, penalised in the
reward as before). The RL state gains one input: the fraction of the shared budget used.

Illustrative: f and the site pond budgets are assumptions, not data. The purpose is to test
whether RL's lack of advantage over greedy was due to the independence of the sites.

Run:  python3 ai_multi_plant_rl_coupled.py [n_seeds] [f1,f2,...]
Writes rl_coupled_results.json
"""
import sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

import ai_multi_plant_rl as base
from ai_multi_plant_rl import (SITE_KEYS, SITE_MAX_CAPACITY_M3D, SITE_POND_BUDGET_HA, TOTAL_REGIONAL_DEMAND_M3D,
                               INCREMENT_M3D, N_SITES, N_STEPS, PolicyNetwork, marginal_lcow, total_system_lcow)
from desalination_plant_design import (PlantCapacity, PlantFlows, ReverseOsmosisSystem, BrineManagementSystem,
                                       EGYPT_BRACKISH_PRESETS, default_recovery_for)

_pond_cache = {}


def pond_ha(site_key, capacity_m3d):
    if capacity_m3d <= 0:
        return 0.0
    key = (site_key, round(capacity_m3d))
    if key not in _pond_cache:
        feed = EGYPT_BRACKISH_PRESETS[site_key]
        cap = PlantCapacity(permeate_flow_m3d=capacity_m3d, recovery=default_recovery_for(feed))
        flows = PlantFlows.from_capacity(cap)
        ro = ReverseOsmosisSystem(flow_m3d=flows.feed_to_membranes_m3d, permeate_m3d=flows.permeate_m3d,
                                  concentrate_m3d=flows.concentrate_m3d, feed_tds_mg_l=feed.tds_mg_l,
                                  water_type=feed.water_type).design()
        _pond_cache[key] = BrineManagementSystem(
            concentrate_flow_m3d=flows.concentrate_m3d, concentrate_tds_mg_l=ro["concentrate_tds_mg_l_est"],
            feed_tds_mg_l=feed.tds_mg_l, is_coastal_site=False, is_arid_climate=True
        ).design_evaporation_ponds_alternative().design["evaporation_pond_option"]["required_pond_area_hectares"]
    return _pond_cache[key]


_lcow_cache = {}


def system_lcow(allocation):
    key = tuple(round(allocation[k]) for k in SITE_KEYS)
    if key not in _lcow_cache:
        _lcow_cache[key] = total_system_lcow(allocation)
    return _lcow_cache[key]


class CoupledEnv:
    def __init__(self, f):
        self.regional_cap = f * sum(SITE_POND_BUDGET_HA.values())

    def used(self, allocation):
        return sum(pond_ha(k, allocation[k]) for k in SITE_KEYS)

    def mask(self, allocation):
        used = self.used(allocation)
        m = []
        for k in SITE_KEYS:
            _, ok_site = marginal_lcow(k, allocation[k])
            extra = pond_ha(k, allocation[k] + INCREMENT_M3D) - pond_ha(k, allocation[k])
            m.append(1.0 if (ok_site and used + extra <= self.regional_cap) else 0.0)
        return np.array(m)

    def state(self, allocation):
        fr = [allocation[k] / SITE_MAX_CAPACITY_M3D[k] for k in SITE_KEYS]
        rem = 1 - sum(allocation.values()) / TOTAL_REGIONAL_DEMAND_M3D
        return np.array(fr + [rem, self.used(allocation) / self.regional_cap])

    def run(self, policy, rng, mode):
        alloc = {k: 0.0 for k in SITE_KEYS}
        traj = []
        for _ in range(N_STEPS):
            m = self.mask(alloc)
            if m.sum() == 0:
                break
            if mode == "greedy":
                costs = np.array([marginal_lcow(k, alloc[k])[0] if m[i] else np.inf for i, k in enumerate(SITE_KEYS)])
                a = int(np.argmin(costs))
            elif mode == "random":
                a = rng.choice(N_SITES, p=m / m.sum())
            else:
                s = self.state(alloc)
                a, probs, h = policy.act(s, rng, action_mask=m)
                traj.append((s, h, probs, a))
            alloc[SITE_KEYS[a]] += INCREMENT_M3D
        delivered = sum(alloc.values())
        lcow = system_lcow(alloc) if delivered > 0 else 0.0
        short = TOTAL_REGIONAL_DEMAND_M3D - delivered
        # reward uses the delivered volume against the build-out target (150,000 m3/d), shortfall penalised as before
        target = N_STEPS * INCREMENT_M3D
        reward = -lcow - 0.00002 * (TOTAL_REGIONAL_DEMAND_M3D - delivered)
        return alloc, lcow, target - delivered, reward, traj


def train(env, seed, n_episodes=600, lr=0.08):
    rng = np.random.default_rng(seed)
    pol = PolicyNetwork(state_dim=N_SITES + 2, n_actions=N_SITES, hidden=16, seed=seed)
    b = 0.0
    for _ in range(n_episodes):
        _, _, _, r, traj = env.run(pol, rng, "rl")
        b = 0.9 * b + 0.1 * r
        for s, h, probs, a in traj:
            pol.apply_grads(pol.grads_log_prob(s, h, probs, a), lr * (r - b))
    return pol


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = {}
    fs = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3]
    for f in fs:
        env = CoupledEnv(f)
        g_alloc, g_lcow, g_short, g_rew, _ = env.run(None, np.random.default_rng(0), "greedy")
        rl, rl_short, rl_rew = [], [], []
        for seed in range(n_seeds):
            pol = train(env, seed)
            a, lc, sh, rw, _ = env.run(pol, np.random.default_rng(999 + seed), "rl")
            rl.append(lc); rl_short.append(sh); rl_rew.append(rw)
        rnd = [env.run(None, np.random.default_rng(i), "random") for i in range(200)]
        out[str(f)] = {
            "regional_pond_cap_ha": env.regional_cap,
            "greedy": {"lcow": g_lcow, "unbuilt_m3d": g_short, "reward": g_rew},
            "rl": {"lcow": rl, "unbuilt_m3d": rl_short, "reward": rl_rew},
            "random": {"lcow_mean": float(np.mean([r[1] for r in rnd])), "unbuilt_mean": float(np.mean([r[2] for r in rnd])),
                       "reward_mean": float(np.mean([r[3] for r in rnd]))},
        }
        print(f"f={f}: cap {env.regional_cap:.0f} ha | greedy reward {g_rew:.4f} (unbuilt {g_short:,.0f}) | "
              f"RL reward {np.mean(rl_rew):.4f} +/- {np.std(rl_rew, ddof=1):.4f} (unbuilt mean {np.mean(rl_short):,.0f}) | "
              f"random reward {out[str(f)]['random']['reward_mean']:.4f}", flush=True)
        json.dump(out, open("rl_coupled_results.json", "w"), default=float)
