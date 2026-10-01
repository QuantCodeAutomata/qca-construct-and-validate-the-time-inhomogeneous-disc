"""Experiment 1: Construct and validate the time-inhomogeneous Markov chain.

Builds a monthly, time-inhomogeneous Markov chain over discretized DNS bond
yields. Validates state capping, successor truncation, retained-mass diagnostics,
and comparison with underlying DNS simulation volatility.

Outputs (saved to results/):
  - exp1_chain_diagnostics_T{T}_vol{vol}.csv
  - exp1_retained_mass_T{T}_vol{vol}.png
  - exp1_yield_comparison_T{T}_vol{vol}.png
"""

import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dns_model import DNSModel
from src.state_discretizer import DatewiseDiscretizer
from src.markov_chain import MarkovChainBuilder
from src.pipeline import ScenarioConfig, build_dns_model, build_chain

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TAU_INITIAL = np.array([2.0, 5.0, 10.0])


def compute_dns_conditional_vol(
    model: DNSModel,
    F0: np.ndarray,
    T: int,
    N: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """Compute DNS one-step conditional yield standard deviations.

    Returns shape (T, 3) – std of y_{t+1} - y_t for each date t and bond.
    """
    factor_paths = model.simulate_paths(N, T, F0=F0, seed=seed)
    tau = model.rolling_maturities(TAU_INITIAL, T)
    yield_paths = np.zeros((N, T + 1, 3), dtype=float)
    for t in range(T + 1):
        yield_paths[:, t, :] = model.yields_from_factors(factor_paths[:, t, :], tau[t])
    diffs = yield_paths[:, 1:, :] - yield_paths[:, :-1, :]  # (N, T, 3)
    return diffs.std(axis=0)  # (T, 3)


def compute_chain_conditional_vol(
    chain: MarkovChainBuilder,
    init_idx: int,
    N_eval: int = 5_000,
    seed: int = 99,
    grid_size: float = 0.004,
) -> np.ndarray:
    """Compute chain-implied one-step conditional yield standard deviations.

    Simulates N_eval paths from init_idx and computes yield std per date.
    Returns shape (T, 3).
    """
    from src.evaluator import PathSimulator

    sim = PathSimulator(chain)
    paths = sim.simulate_paths(init_idx, N_eval, seed=seed)

    T = chain.T
    yield_paths = np.zeros((N_eval, T + 1, 3), dtype=float)
    for t in range(T + 1):
        coords = np.array(chain.state_lists[t], dtype=float) * grid_size
        yield_paths[:, t, :] = coords[paths[:, t]]

    diffs = yield_paths[:, 1:, :] - yield_paths[:, :-1, :]
    return diffs.std(axis=0)


def run_exp1():
    """Main entry point for Experiment 1."""
    scenarios = [
        ScenarioConfig(T=6,  volatility="baseline", N_train=50_000, label="T6_baseline"),
        ScenarioConfig(T=12, volatility="baseline", N_train=50_000, label="T12_baseline"),
        ScenarioConfig(T=6,  volatility="low",      N_train=50_000, label="T6_low"),
        ScenarioConfig(T=6,  volatility="high",     N_train=50_000, label="T6_high"),
    ]

    all_diag_rows = []

    for cfg in scenarios:
        print(f"\n--- Building chain: {cfg.label} ---")
        model = build_dns_model(cfg)
        chain, disc = build_chain(cfg, model, verbose=True)

        assert chain.validate(), f"Chain validation failed for {cfg.label}!"
        print(f"  Chain validated: OK")

        # ----------------------------------------------------------------
        # Diagnostics table
        # ----------------------------------------------------------------
        diag = chain.compute_diagnostics()
        for t in range(cfg.T):
            d    = diag[t]
            rm   = d["retained_mass_stats"]
            row  = {
                "scenario":          cfg.label,
                "T":                 cfg.T,
                "volatility":        cfg.volatility,
                "date":              t,
                "n_states":          d["n_states_after_cap"],
                "retained_mass_mean": rm["mean"],
                "retained_mass_min":  rm["min"],
                "retained_mass_max":  rm["max"],
                "n_rows_lt_25":       d["n_rows_lt_25_successors"],
                "n_empty_rows":       d["n_empty_rows"],
            }
            all_diag_rows.append(row)
            print(
                f"  t={t}: states={d['n_states_after_cap']}, "
                f"ret_mass={rm['mean']:.4f} [{rm['min']:.4f}, {rm['max']:.4f}], "
                f"rows<25={d['n_rows_lt_25_successors']}"
            )

        # ----------------------------------------------------------------
        # Retained-mass histogram
        # ----------------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        for t in range(min(cfg.T, 6)):
            masses = chain.retained_mass_per_row(t)
            axes[t].hist(masses, bins=20, range=(0, 1), edgecolor="black", color="steelblue")
            axes[t].set_title(f"Date t={t}")
            axes[t].set_xlabel("Retained Mass")
            axes[t].set_ylabel("Count")
            axes[t].axvline(np.mean(masses), color="red", linestyle="--",
                           label=f"Mean={np.mean(masses):.3f}")
            axes[t].legend(fontsize=8)
        for t in range(min(cfg.T, 6), 6):
            axes[t].set_visible(False)
        fig.suptitle(f"Retained Transition Mass per Origin State — {cfg.label}", fontsize=13)
        plt.tight_layout()
        fname = os.path.join(RESULTS_DIR, f"exp1_retained_mass_{cfg.label}.png")
        plt.savefig(fname, dpi=100)
        plt.close()
        print(f"  Saved: {fname}")

        # ----------------------------------------------------------------
        # Conditional volatility comparison
        # ----------------------------------------------------------------
        F0 = model.initial_conditions["long_run_mean"]
        dns_vol = compute_dns_conditional_vol(model, F0, cfg.T, N=20_000, seed=42)

        # Find initial state index
        from scipy.spatial.distance import cdist
        tau_0 = TAU_INITIAL
        y0 = model.yields_from_factors(F0, tau_0)
        coord_0 = tuple(int(round(v / cfg.grid_bp)) for v in y0)
        if coord_0 in chain.state_to_idx[0]:
            init_idx = chain.state_to_idx[0][coord_0]
        else:
            states_arr = np.array(chain.state_lists[0], dtype=float) * cfg.grid_bp
            init_idx = int(np.argmin(cdist([y0], states_arr, metric="euclidean")[0]))

        chain_vol = compute_chain_conditional_vol(
            chain, init_idx, N_eval=5_000, seed=99, grid_size=cfg.grid_bp
        )

        bond_labels = ["2Y", "5Y", "10Y"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        t_axis = np.arange(1, cfg.T + 1)
        for k, label in enumerate(bond_labels):
            axes[k].plot(t_axis, dns_vol[:, k] * 1e4, "b-o", label="DNS Simulation", linewidth=2)
            axes[k].plot(t_axis, chain_vol[:, k] * 1e4, "r--s", label="Markov Chain", linewidth=2)
            axes[k].set_title(f"Bond {label}")
            axes[k].set_xlabel("Time Step")
            axes[k].set_ylabel("Conditional Std (bps)")
            axes[k].legend()
            axes[k].grid(True, alpha=0.3)
        fig.suptitle(f"One-Step Yield Conditional Volatility — {cfg.label}", fontsize=13)
        plt.tight_layout()
        fname = os.path.join(RESULTS_DIR, f"exp1_cond_vol_{cfg.label}.png")
        plt.savefig(fname, dpi=100)
        plt.close()
        print(f"  Saved: {fname}")

    # ----------------------------------------------------------------
    # Save diagnostics CSV
    # ----------------------------------------------------------------
    df_diag = pd.DataFrame(all_diag_rows)
    csv_path = os.path.join(RESULTS_DIR, "exp1_chain_diagnostics.csv")
    df_diag.to_csv(csv_path, index=False)
    print(f"\nSaved diagnostics: {csv_path}")
    print(df_diag.to_string())

    return df_diag


if __name__ == "__main__":
    run_exp1()
