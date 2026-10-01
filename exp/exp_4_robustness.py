"""Experiment 4: Scenario robustness analysis.

Tests DP advantage across transaction costs, horizons, volatility regimes,
initial curve shapes, and continuation parameters.

Outputs saved to results/:
  - exp4_vol_sensitivity.csv
  - exp4_initial_curve.csv
  - exp4_rho_sensitivity.csv
  - exp4_summary_plots.png
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dns_model import DNSModel
from src.pipeline import (
    ScenarioConfig, run_scenario, build_dns_model, build_chain,
    solve_mdp, estimate_mvo, evaluate_all, find_initial_state,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ACTION_LABELS = ["2Y", "5Y", "10Y", "EW", "mod_long", "mod_short"]


def run_vol_sensitivity():
    """Test T=6, c=0.001, rho=1 across volatility regimes."""
    print("\n=== Volatility Sensitivity ===")
    rows = []
    for vol in ["low", "baseline", "high"]:
        cfg = ScenarioConfig(
            T=6, volatility=vol, cost_c=0.001, rho=1.0,
            N_train=50_000, N_eval=500, N_eval_seeds=3, mvo_N_sim=10_000,
            label=f"vol_{vol}",
        )
        out = run_scenario(cfg, verbose=False)
        for strat, key in [("DP", "dp"), ("EW", "ew"), ("MVO", "mvo")]:
            m = out["results"][key]["metrics"]
            rows.append({
                "volatility": vol, "strategy": strat,
                "mean_tw":    m["mean_terminal"],
                "median_tw":  m["median_terminal"],
                "q5_tw":      m["q5_terminal"],
                "mean_mdd":   m["mean_mdd"],
            })
        print(f"  {vol}: DP={out['results']['dp']['metrics']['mean_terminal']:.4f}, "
              f"EW={out['results']['ew']['metrics']['mean_terminal']:.4f}, "
              f"MVO={out['results']['mvo']['metrics']['mean_terminal']:.4f}")
    return pd.DataFrame(rows)


def run_initial_curve_sensitivity():
    """Test T=6, c=0.001, rho=1 across initial yield-curve shapes."""
    print("\n=== Initial Curve Sensitivity ===")
    rows = []
    for ic in ["long_run_mean", "upward_sloping", "inverted", "original"]:
        cfg = ScenarioConfig(
            T=6, volatility="baseline", initial_curve=ic, cost_c=0.001, rho=1.0,
            N_train=50_000, N_eval=500, N_eval_seeds=3, mvo_N_sim=10_000,
            label=f"ic_{ic}",
        )
        # Check if the initial_curve key exists
        model = build_dns_model(cfg)
        if ic not in model.initial_conditions:
            print(f"  Skipping {ic} (not defined in model).")
            continue

        out = run_scenario(cfg, verbose=False)
        for strat, key in [("DP", "dp"), ("EW", "ew"), ("MVO", "mvo")]:
            m = out["results"][key]["metrics"]
            rows.append({
                "initial_curve": ic, "strategy": strat,
                "mean_tw":    m["mean_terminal"],
                "median_tw":  m["median_terminal"],
                "q5_tw":      m["q5_terminal"],
                "mean_mdd":   m["mean_mdd"],
            })
            if strat == "MVO":
                rows[-1]["mvo_weights"] = str(out["results"]["mvo_weights"].round(3))

        dp_m = out["results"]["dp"]["metrics"]["mean_terminal"]
        mvo_m = out["results"]["mvo"]["metrics"]["mean_terminal"]
        p_dp_mvo = out["results"]["paired_dp_vs_mvo"]
        print(f"  {ic}: DP={dp_m:.4f}, MVO={mvo_m:.4f}, "
              f"t-stat DP-MVO={p_dp_mvo['t_stat']:.2f}")
    return pd.DataFrame(rows)


def run_rho_sensitivity():
    """Test T=6, c in {0, 0.001, 0.01} across rho values."""
    print("\n=== Continuation Parameter Sensitivity ===")
    rows = []
    for c in [0.0, 0.001, 0.01]:
        for rho in [0.90, 0.95, 1.00]:
            base_cfg = ScenarioConfig(
                T=6, volatility="baseline", cost_c=c, rho=rho,
                N_train=50_000, N_eval=500, N_eval_seeds=3, mvo_N_sim=10_000,
                label=f"rho_c{c}_rho{rho}",
            )
            out = run_scenario(base_cfg, verbose=False)
            m = out["results"]["dp"]["metrics"]
            rows.append({
                "c": c, "rho": rho,
                "mean_tw":   m["mean_terminal"],
                "median_tw": m["median_terminal"],
                "q5_tw":     m["q5_terminal"],
                "mean_mdd":  m["mean_mdd"],
                "mean_turnover": m.get("mean_turnover", np.nan),
            })
            print(f"  c={c:.4f}, rho={rho:.2f}: DP mean={m['mean_terminal']:.4f}")
    return pd.DataFrame(rows)


def run_exp4():
    """Main entry point for Experiment 4."""
    # -- Volatility sensitivity --
    df_vol = run_vol_sensitivity()
    df_vol.to_csv(os.path.join(RESULTS_DIR, "exp4_vol_sensitivity.csv"), index=False)
    print(df_vol.to_string())

    # -- Initial curve sensitivity --
    df_ic = run_initial_curve_sensitivity()
    if not df_ic.empty:
        df_ic.to_csv(os.path.join(RESULTS_DIR, "exp4_initial_curve.csv"), index=False)
        print(df_ic.to_string())

    # -- Rho sensitivity --
    df_rho = run_rho_sensitivity()
    df_rho.to_csv(os.path.join(RESULTS_DIR, "exp4_rho_sensitivity.csv"), index=False)
    print(df_rho.to_string())

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: vol sensitivity
    ax = axes[0]
    if not df_vol.empty:
        vol_order = ["low", "baseline", "high"]
        for strat, c, ls in [("DP", "blue", "-"), ("EW", "green", "--"), ("MVO", "red", ":")]:
            s = df_vol[df_vol["strategy"] == strat]
            s = s.set_index("volatility").reindex(vol_order)
            ax.plot(range(len(vol_order)), s["mean_tw"], color=c, linestyle=ls,
                    marker="o", label=strat, linewidth=2)
        ax.set_xticks(range(len(vol_order)))
        ax.set_xticklabels(vol_order)
        ax.set_ylabel("Mean Terminal Wealth")
        ax.set_title("Volatility Sensitivity (T=6, c=0.001)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 2: initial curve sensitivity
    ax = axes[1]
    if not df_ic.empty:
        ics = df_ic["initial_curve"].unique().tolist()
        for strat, c, ls in [("DP", "blue", "-"), ("EW", "green", "--"), ("MVO", "red", ":")]:
            s = df_ic[df_ic["strategy"] == strat]
            means = [s[s["initial_curve"] == ic]["mean_tw"].values[0]
                     if len(s[s["initial_curve"] == ic]) > 0 else np.nan for ic in ics]
            ax.plot(range(len(ics)), means, color=c, linestyle=ls, marker="o",
                    label=strat, linewidth=2)
        ax.set_xticks(range(len(ics)))
        ax.set_xticklabels([ic[:8] for ic in ics], rotation=20)
        ax.set_ylabel("Mean Terminal Wealth")
        ax.set_title("Initial Curve Sensitivity (T=6, c=0.001)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 3: rho sensitivity at c=0
    ax = axes[2]
    for c_val, c_color in [(0.0, "blue"), (0.001, "green"), (0.01, "red")]:
        sub = df_rho[df_rho["c"] == c_val]
        ax.plot(sub["rho"], sub["mean_tw"], color=c_color, marker="o",
                label=f"c={c_val}", linewidth=2)
    ax.set_xlabel("Continuation Parameter rho")
    ax.set_ylabel("Mean Terminal Wealth")
    ax.set_title("rho Sensitivity (T=6, DP)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Robustness Analysis: Volatility, Initial Curve, rho", fontsize=13)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "exp4_robustness_plots.png")
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"\nSaved: {fname}")

    return df_vol, df_ic, df_rho


if __name__ == "__main__":
    run_exp4()
