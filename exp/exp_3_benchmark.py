"""Experiment 3: Benchmark construction and paired evaluation.

Evaluates DP policy against equal-weight buy-and-hold and static MVO
buy-and-hold using common random numbers. Tests whether DP delivers
higher mean/median terminal wealth.

Outputs saved to results/:
  - exp3_terminal_wealth_T{T}.csv
  - exp3_wealth_distribution_T{T}.png
  - exp3_paired_stats_T{T}.csv
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
    ScenarioConfig, run_scenario, build_dns_model, build_chain,
    solve_mdp, estimate_mvo, evaluate_all,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COST_GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]
HORIZONS  = [6, 12]


def run_exp3():
    """Main entry point for Experiment 3."""
    all_summary_rows = []
    all_paired_rows  = []

    for T in HORIZONS:
        print(f"\n{'='*60}")
        print(f"Horizon T={T}")
        print(f"{'='*60}")

        # Build chain ONCE per horizon (chain doesn't depend on cost_c or rho)
        chain_cfg = ScenarioConfig(
            T=T, volatility="baseline", cost_c=0.001,
            N_train=50_000, N_eval=500, N_eval_seeds=3,
            mvo_N_sim=10_000,
            label=f"T{T}_chain_build",
        )
        model = build_dns_model(chain_cfg)
        chain, disc = build_chain(chain_cfg, model, verbose=True)
        print(f"  Chain built: T={T}, {sum(len(s) for s in chain.state_lists)} total states")

        for c in COST_GRID:
            print(f"\n  --- Cost c={c:.4f} ---")
            cfg = ScenarioConfig(
                T=T, volatility="baseline", cost_c=c,
                N_train=50_000, N_eval=500, N_eval_seeds=3,
                mvo_N_sim=10_000,
                label=f"T{T}_baseline_c{c}",
            )
            solver    = solve_mdp(chain, cfg, verbose=False)
            mvo_bench = estimate_mvo(chain, cfg)
            results   = evaluate_all(chain, solver, mvo_bench, cfg, model, verbose=False)

            m_dp  = results["dp"]["metrics"]
            m_ew  = results["ew"]["metrics"]
            m_mvo = results["mvo"]["metrics"]

            for strat, m in [("DP", m_dp), ("EW", m_ew), ("MVO", m_mvo)]:
                row = {"T": T, "c": c, "strategy": strat,
                       "mean_tw": m["mean_terminal"],
                       "median_tw": m["median_terminal"],
                       "q5_tw": m["q5_terminal"],
                       "mean_mdd": m["mean_mdd"]}
                if strat == "DP":
                    row["mean_turnover"] = m.get("mean_turnover", np.nan)
                    row["mean_cum_cost"] = m.get("mean_cum_cost", np.nan)
                all_summary_rows.append(row)

            p_ew  = results["paired_dp_vs_ew"]
            p_mvo = results["paired_dp_vs_mvo"]
            for bench, p in [("EW", p_ew), ("MVO", p_mvo)]:
                all_paired_rows.append({
                    "T": T, "c": c, "benchmark": bench,
                    "mean_diff": p["mean_diff"],
                    "std_error": p["std_error"],
                    "t_stat":    p["t_stat"],
                    "n_paths":   p["n"],
                })

            print(f"  DP mean={m_dp['mean_terminal']:.4f}, "
                  f"EW mean={m_ew['mean_terminal']:.4f}, "
                  f"MVO mean={m_mvo['mean_terminal']:.4f}")
            print(f"  Paired DP-EW t={p_ew['t_stat']:.2f}, "
                  f"DP-MVO t={p_mvo['t_stat']:.2f}")

    # ----------------------------------------------------------------
    # Save CSVs
    # ----------------------------------------------------------------
    df_summary = pd.DataFrame(all_summary_rows)
    df_paired  = pd.DataFrame(all_paired_rows)

    csv1 = os.path.join(RESULTS_DIR, "exp3_terminal_wealth_summary.csv")
    csv2 = os.path.join(RESULTS_DIR, "exp3_paired_stats.csv")
    df_summary.to_csv(csv1, index=False)
    df_paired.to_csv(csv2, index=False)
    print(f"\nSaved: {csv1}")
    print(f"Saved: {csv2}")
    print(df_summary.to_string())
    print(df_paired.to_string())

    # ----------------------------------------------------------------
    # Plot: mean terminal wealth vs cost for both horizons
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, T in zip(axes, HORIZONS):
        sub = df_summary[df_summary["T"] == T]
        for strat, color, ls in [("DP", "blue", "-"), ("EW", "green", "--"), ("MVO", "red", ":")]:
            s = sub[sub["strategy"] == strat]
            ax.plot(s["c"], s["mean_tw"], color=color, linestyle=ls,
                    marker="o", label=strat, linewidth=2)
        ax.set_xlabel("Transaction Cost c")
        ax.set_ylabel("Mean Terminal Wealth")
        ax.set_title(f"T={T} months")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("symlog", linthresh=0.0001)
    fig.suptitle("Mean Terminal Wealth vs Transaction Cost", fontsize=13)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "exp3_mean_wealth_by_cost.png")
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"Saved: {fname}")

    # Plot: turnover vs cost (T=6)
    dp_data = df_summary[(df_summary["T"] == 6) & (df_summary["strategy"] == "DP")]
    if "mean_turnover" in dp_data.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dp_data["c"], dp_data["mean_turnover"], "bo-", linewidth=2, label="Turnover")
        ax_r = ax.twinx()
        ax_r.plot(dp_data["c"], dp_data["mean_cum_cost"], "rs--", linewidth=2, label="Cum. Cost")
        ax.set_xlabel("Transaction Cost c")
        ax.set_ylabel("Average Turnover", color="blue")
        ax_r.set_ylabel("Mean Cumulative Cost", color="red")
        ax.set_title("T=6: DP Turnover and Cumulative Cost vs c")
        ax.set_xscale("symlog", linthresh=0.0001)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(RESULTS_DIR, "exp3_turnover_vs_cost.png")
        plt.savefig(fname, dpi=100)
        plt.close()
        print(f"Saved: {fname}")

    return df_summary, df_paired


if __name__ == "__main__":
    run_exp3()
