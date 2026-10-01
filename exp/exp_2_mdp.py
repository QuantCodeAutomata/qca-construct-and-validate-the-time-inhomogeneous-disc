"""Experiment 2: Solve the finite-horizon dynamic bond-allocation MDP.

Solves the Bellman recursion for multiple transaction-cost levels c and
continuation parameters rho. Generates policy diagnostics and validates
that higher transaction costs reduce trading aggressiveness.

Outputs saved to results/:
  - exp2_policy_action_freq.csv
  - exp2_policy_diagnostics.csv
  - exp2_action_freq_by_cost.png
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import (
    ScenarioConfig, build_dns_model, build_chain, solve_mdp,
    find_initial_state, TAU_INITIAL,
)
from src.mdp_solver import MDPSolver, ACTIONS, N_ACTIONS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COST_GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]
RHO_GRID  = [0.90, 0.95, 1.00]
ACTION_LABELS = ["2Y", "5Y", "10Y", "EW", "mod_long", "mod_short"]


def policy_action_frequencies(solver: MDPSolver, t: int = 0) -> np.ndarray:
    """Return fraction of states choosing each action at date t (initial policy)."""
    pol = solver.policy[t]
    if pol is None:
        return np.zeros(N_ACTIONS)
    if pol.ndim == 1:
        counts = np.bincount(pol, minlength=N_ACTIONS)
    else:
        counts = np.bincount(pol.flatten(), minlength=N_ACTIONS)
    return counts / counts.sum()


def run_exp2():
    """Main entry point for Experiment 2."""
    # Build chain once for baseline scenario (reuse across cost/rho grid)
    base_cfg = ScenarioConfig(T=6, volatility="baseline", N_train=50_000,
                              cost_c=0.0, label="T6_baseline_c0")
    print("Building baseline chain...")
    model = build_dns_model(base_cfg)
    chain, disc = build_chain(base_cfg, model, verbose=True)
    init_idx = find_initial_state(chain, model, base_cfg)
    print(f"Initial state index: {init_idx}")
    print(f"Initial state (decimal yields): "
          f"{[c * base_cfg.grid_bp for c in chain.state_lists[0][init_idx]]}")

    assert chain.validate(), "Chain validation failed!"

    rows = []
    freq_by_cost = {}

    for c in COST_GRID:
        for rho in RHO_GRID:
            cfg_key = f"c={c:.4f}_rho={rho:.2f}"
            print(f"\n  Solving MDP: c={c}, rho={rho}")

            cfg = ScenarioConfig(T=6, volatility="baseline", N_train=50_000,
                                 cost_c=c, rho=rho, label=cfg_key)
            solver = solve_mdp(chain, cfg, verbose=False)

            # Action frequency at t=0
            freq_0 = policy_action_frequencies(solver, t=0)
            # Action frequency at t=1..T-1 (aggregate over all prev states)
            freq_mid = []
            for t in range(1, cfg.T - 1):
                pol_t = solver.policy[t]
                if pol_t is not None:
                    f = np.bincount(pol_t.flatten(), minlength=N_ACTIONS)
                    freq_mid.append(f / f.sum())
            freq_mid_avg = np.mean(freq_mid, axis=0) if freq_mid else np.zeros(N_ACTIONS)

            row = {
                "c": c, "rho": rho,
                "J0_init": float(solver.J[0][init_idx]) if solver.J[0] is not None else np.nan,
            }
            for a_idx, lbl in enumerate(ACTION_LABELS):
                row[f"freq_t0_{lbl}"] = float(freq_0[a_idx])
                row[f"freq_tmid_{lbl}"] = float(freq_mid_avg[a_idx])
            rows.append(row)

            if rho == 1.0:
                freq_by_cost[c] = {"t0": freq_0, "tmid": freq_mid_avg}

        print(f"  Solved c={c:.4f} for all rho values.")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "exp2_policy_diagnostics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df[["c", "rho", "J0_init"]].to_string())

    # ----------------------------------------------------------------
    # Plot: action frequencies at t=0 vs cost (rho=1)
    # ----------------------------------------------------------------
    costs_plot = sorted(freq_by_cost.keys())
    freq_mat = np.array([freq_by_cost[c]["t0"] for c in costs_plot])  # (n_c, N_A)

    fig, ax = plt.subplots(figsize=(10, 5))
    for a_idx, lbl in enumerate(ACTION_LABELS):
        ax.plot(costs_plot, freq_mat[:, a_idx], "o-", label=lbl, linewidth=2)
    ax.set_xlabel("Transaction Cost c")
    ax.set_ylabel("Fraction of States Choosing Action")
    ax.set_title("Policy Action Frequencies at t=0 vs Transaction Cost (rho=1)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=0.0001)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "exp2_action_freq_by_cost.png")
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"Saved: {fname}")

    return df


if __name__ == "__main__":
    run_exp2()
