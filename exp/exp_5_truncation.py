"""Experiment 5: Truncation and approximation fidelity study.

Investigates how Markov-chain pruning choices affect portfolio metrics.
Hypothesis: mean terminal wealth is more stable under truncation than Q5/MDD.

Outputs saved to results/:
  - exp5_truncation_diagnostics.csv
  - exp5_truncation_portfolio_metrics.csv
  - exp5_retained_mass_vs_M.png
  - exp5_metrics_vs_M.png
"""

import sys
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dns_model import DNSModel
from src.state_discretizer import DatewiseDiscretizer
from src.markov_chain import MarkovChainBuilder
from src.pipeline import (
    ScenarioConfig, build_dns_model, solve_mdp, estimate_mvo, evaluate_all,
    find_initial_state, TAU_INITIAL, DELTA,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configurations to test
MAX_SUCCESSORS_GRID = [5, 10, 25, 50]
MAX_STATES_GRID     = [40, 80, 120]
BASE_COST           = 0.001
BASE_T              = 6


def build_chain_with_config(
    factor_paths: np.ndarray,
    tau_0: np.ndarray,
    model: DNSModel,
    T: int,
    max_states: int,
    max_successors: int,
    grid_bp: float = 0.004,
) -> tuple[MarkovChainBuilder, DatewiseDiscretizer]:
    """Build chain from pre-simulated paths with given approximation settings."""
    N, T1, _ = factor_paths.shape
    yield_paths = np.zeros((N, T1, 3), dtype=float)
    tau_mat = model.rolling_maturities(TAU_INITIAL, T)
    for t in range(T1):
        yield_paths[:, t, :] = model.yields_from_factors(factor_paths[:, t, :], tau_mat[t])

    disc = DatewiseDiscretizer(grid_size=grid_bp, max_states=max_states)
    disc.fit(yield_paths)
    idx_paths = disc.transform(yield_paths)

    chain = MarkovChainBuilder(max_states=max_states, max_successors=max_successors)
    chain.build(
        idx_paths, T, disc.state_lists, disc.state_to_idx,
        n_states_before_cap=disc.n_states_before_cap,
    )
    return chain, disc


def chain_diagnostics_summary(chain: MarkovChainBuilder, T: int) -> dict:
    """Compute summary chain diagnostics."""
    diag = chain.compute_diagnostics()
    n_before = []
    n_after  = []
    rm_means = []
    for t in range(T):
        if t in diag:
            d = diag[t]
            n_before.append(d.get("n_states_before_cap", d["n_states_after_cap"]))
            n_after.append(d["n_states_after_cap"])
            rm_means.append(d["retained_mass_stats"]["mean"])
    return {
        "mean_n_before":  float(np.mean(n_before)) if n_before else np.nan,
        "mean_n_after":   float(np.mean(n_after))  if n_after  else np.nan,
        "mean_ret_mass":  float(np.mean(rm_means))  if rm_means else np.nan,
        "min_ret_mass":   float(np.min(rm_means))   if rm_means else np.nan,
    }


def run_exp5():
    """Main entry point for Experiment 5."""
    # Pre-simulate a single large training set (reused across configs)
    base_cfg = ScenarioConfig(T=BASE_T, volatility="baseline", N_train=50_000,
                              cost_c=BASE_COST, label="exp5_base")
    model = build_dns_model(base_cfg)
    F0    = model.initial_conditions[base_cfg.initial_curve]
    print("Pre-simulating training paths (shared across configs)...")
    t0 = time.time()
    factor_paths = model.simulate_paths(base_cfg.N_train, BASE_T,
                                        F0=F0, seed=base_cfg.train_seed)
    print(f"  Simulated {base_cfg.N_train} paths in {time.time()-t0:.1f}s")

    # ----------------------------------------------------------------
    # Vary max_successors M at fixed max_states=80
    # ----------------------------------------------------------------
    print("\n=== Varying max_successors M (max_states=80) ===")
    diag_rows     = []
    metric_rows   = []
    retained_mass_all = {}  # M -> list of per-row masses

    for M in MAX_SUCCESSORS_GRID:
        print(f"\n  M={M}...")
        chain, disc = build_chain_with_config(
            factor_paths, TAU_INITIAL, model, BASE_T,
            max_states=80, max_successors=M,
        )
        diag = chain_diagnostics_summary(chain, BASE_T)

        # Collect per-row retained masses using the public method
        all_masses = []
        for t in range(BASE_T):
            all_masses.extend(chain.retained_mass_per_row(t).tolist())
        retained_mass_all[M] = all_masses

        cfg_m = ScenarioConfig(T=BASE_T, volatility="baseline",
                               cost_c=BASE_COST, max_successors=M, max_states=80,
                               N_train=50_000, N_eval=300, N_eval_seeds=2,
                               mvo_N_sim=5_000, label=f"exp5_M{M}")
        solver    = solve_mdp(chain, cfg_m, verbose=False)
        mvo_bench = estimate_mvo(chain, cfg_m)
        results   = evaluate_all(chain, solver, mvo_bench, cfg_m, model, verbose=False)

        m_dp  = results["dp"]["metrics"]
        m_ew  = results["ew"]["metrics"]
        m_mvo = results["mvo"]["metrics"]

        diag_rows.append({
            "config": f"M={M}",
            "max_successors": M, "max_states": 80,
            **diag,
        })
        for strat, m in [("DP", m_dp), ("EW", m_ew), ("MVO", m_mvo)]:
            metric_rows.append({
                "config": f"M={M}", "max_successors": M, "max_states": 80,
                "strategy": strat,
                "mean_tw":   m["mean_terminal"],
                "median_tw": m["median_terminal"],
                "q5_tw":     m["q5_terminal"],
                "mean_mdd":  m["mean_mdd"],
            })
        print(f"  M={M}: ret_mass={diag['mean_ret_mass']:.4f}, "
              f"DP mean={m_dp['mean_terminal']:.4f}, "
              f"DP q5={m_dp['q5_terminal']:.4f}, "
              f"DP mdd={m_dp['mean_mdd']:.4f}")

    # ----------------------------------------------------------------
    # Vary max_states at fixed M=25
    # ----------------------------------------------------------------
    print("\n=== Varying max_states (M=25) ===")
    for S in MAX_STATES_GRID:
        print(f"\n  S={S}...")
        chain, disc = build_chain_with_config(
            factor_paths, TAU_INITIAL, model, BASE_T,
            max_states=S, max_successors=25,
        )
        diag = chain_diagnostics_summary(chain, BASE_T)

        cfg_s = ScenarioConfig(T=BASE_T, volatility="baseline",
                               cost_c=BASE_COST, max_successors=25, max_states=S,
                               N_train=50_000, N_eval=300, N_eval_seeds=2,
                               mvo_N_sim=5_000, label=f"exp5_S{S}")
        solver    = solve_mdp(chain, cfg_s, verbose=False)
        mvo_bench = estimate_mvo(chain, cfg_s)
        results   = evaluate_all(chain, solver, mvo_bench, cfg_s, model, verbose=False)

        m_dp  = results["dp"]["metrics"]
        m_ew  = results["ew"]["metrics"]
        m_mvo = results["mvo"]["metrics"]

        diag_rows.append({
            "config": f"S={S}", "max_successors": 25, "max_states": S,
            **diag,
        })
        for strat, m in [("DP", m_dp), ("EW", m_ew), ("MVO", m_mvo)]:
            metric_rows.append({
                "config": f"S={S}", "max_successors": 25, "max_states": S,
                "strategy": strat,
                "mean_tw":   m["mean_terminal"],
                "median_tw": m["median_terminal"],
                "q5_tw":     m["q5_terminal"],
                "mean_mdd":  m["mean_mdd"],
            })
        print(f"  S={S}: ret_mass={diag['mean_ret_mass']:.4f}, "
              f"DP mean={m_dp['mean_terminal']:.4f}, "
              f"DP q5={m_dp['q5_terminal']:.4f}")

    # ----------------------------------------------------------------
    # Save CSVs
    # ----------------------------------------------------------------
    df_diag    = pd.DataFrame(diag_rows)
    df_metrics = pd.DataFrame(metric_rows)
    df_diag.to_csv(os.path.join(RESULTS_DIR, "exp5_truncation_diagnostics.csv"), index=False)
    df_metrics.to_csv(os.path.join(RESULTS_DIR, "exp5_truncation_portfolio_metrics.csv"), index=False)
    print("\n" + df_diag.to_string())
    print("\n" + df_metrics.to_string())

    # ----------------------------------------------------------------
    # Plot 1: Retained mass histograms for different M values
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, len(MAX_SUCCESSORS_GRID), figsize=(16, 4))
    for ax, M in zip(axes, MAX_SUCCESSORS_GRID):
        masses = retained_mass_all.get(M, [])
        if masses:
            ax.hist(masses, bins=20, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Retained Mass")
        ax.set_ylabel("Count")
        ax.set_title(f"M={M}")
        ax.set_xlim(0, 1.05)
    fig.suptitle("Retained Transition Mass by Successor Cap M", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp5_retained_mass_vs_M.png"), dpi=100)
    plt.close()

    # ----------------------------------------------------------------
    # Plot 2: Mean / Q5 / MDD vs M for DP
    # ----------------------------------------------------------------
    dp_data = df_metrics[
        (df_metrics["strategy"] == "DP") &
        (df_metrics["config"].str.startswith("M="))
    ].copy()
    dp_data["M"] = dp_data["max_successors"].astype(int)
    dp_data = dp_data.sort_values("M")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, lbl in [
        (axes[0], "mean_tw",  "Mean Terminal Wealth"),
        (axes[1], "q5_tw",   "Q5 Terminal Wealth"),
        (axes[2], "mean_mdd", "Mean MDD"),
    ]:
        ax.plot(dp_data["M"], dp_data[col], "bo-", linewidth=2)
        ax.set_xlabel("Successor Cap M")
        ax.set_ylabel(lbl)
        ax.set_title(lbl)
        ax.grid(True, alpha=0.3)
    fig.suptitle("DP Portfolio Metrics vs Successor Cap M (max_states=80)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp5_metrics_vs_M.png"), dpi=100)
    plt.close()

    print(f"\nSaved: exp5 results to {RESULTS_DIR}")
    return df_diag, df_metrics


if __name__ == "__main__":
    run_exp5()
