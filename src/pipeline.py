"""Common pipeline utilities for DNS Markov chain experiments.

Provides end-to-end workflow functions used across all five experiments:
chain construction, MDP solving, benchmark estimation, and strategy evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.dns_model import DNSModel
from src.state_discretizer import DatewiseDiscretizer
from src.markov_chain import MarkovChainBuilder
from src.mdp_solver import MDPSolver, ACTIONS
from src.benchmarks import EqualWeightBenchmark, MVOBenchmark
from src.evaluator import PathSimulator, StrategyEvaluator

TAU_INITIAL = np.array([2.0, 5.0, 10.0])
DELTA       = 1.0 / 12.0
GRID_SIZE   = 0.004


@dataclass
class ScenarioConfig:
    """Configuration for a single experiment scenario."""
    T:              int              = 6
    volatility:     str              = "baseline"
    initial_curve:  str              = "long_run_mean"
    cost_c:         float            = 0.001
    rho:            float            = 1.0
    N_train:        int              = 100_000
    N_eval:         int              = 1_000
    N_eval_seeds:   int              = 3
    mvo_N_sim:      int              = 20_000
    mvo_seed:       int              = 99
    train_seed:     int              = 0
    max_states:     int              = 80
    max_successors: int              = 25
    grid_bp:        float            = 0.004
    gamma_mvo:      float            = 5.0
    label:          str              = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = (
                f"T{self.T}_vol{self.volatility}_ic{self.initial_curve}"
                f"_c{self.cost_c}_rho{self.rho}"
            )


def build_dns_model(cfg: ScenarioConfig) -> DNSModel:
    """Build a DNSModel with the volatility scenario from cfg."""
    sigma_eta = DNSModel.VOLATILITY_SCENARIOS[cfg.volatility]
    return DNSModel(sigma_eta=sigma_eta)


def build_chain(
    cfg: ScenarioConfig,
    model: Optional[DNSModel] = None,
    verbose: bool = False,
) -> tuple[MarkovChainBuilder, DatewiseDiscretizer]:
    """Simulate DNS paths, discretize, and build the Markov chain.

    Parameters
    ----------
    cfg : ScenarioConfig
    model : DNSModel or None
        If None, built from cfg.
    verbose : bool
        Print timing info.

    Returns
    -------
    (chain, discretizer)
    """
    if model is None:
        model = build_dns_model(cfg)

    F0 = model.initial_conditions[cfg.initial_curve]
    tau = model.rolling_maturities(TAU_INITIAL, cfg.T)  # (T+1, K)

    t0 = time.time()
    factor_paths = model.simulate_paths(cfg.N_train, cfg.T, F0=F0, seed=cfg.train_seed)

    # Convert to yield paths: shape (N_train, T+1, K)
    N, T1, _ = factor_paths.shape
    yield_paths = np.zeros((N, T1, 3), dtype=float)
    for t in range(T1):
        yield_paths[:, t, :] = model.yields_from_factors(factor_paths[:, t, :], tau[t])

    t1 = time.time()
    if verbose:
        print(f"  DNS simulation: {t1-t0:.2f}s")

    disc = DatewiseDiscretizer(grid_size=cfg.grid_bp, max_states=cfg.max_states)
    disc.fit(yield_paths)
    idx_paths = disc.transform(yield_paths)

    t2 = time.time()
    if verbose:
        print(f"  Discretization: {t2-t1:.2f}s")

    chain = MarkovChainBuilder(max_states=cfg.max_states, max_successors=cfg.max_successors)
    chain.build(
        idx_paths, cfg.T,
        disc.state_lists, disc.state_to_idx,
        n_states_before_cap=disc.n_states_before_cap,
    )

    t3 = time.time()
    if verbose:
        print(f"  Chain building: {t3-t2:.2f}s")

    return chain, disc


def solve_mdp(
    chain: MarkovChainBuilder,
    cfg: ScenarioConfig,
    verbose: bool = False,
) -> MDPSolver:
    """Solve the MDP by backward induction.

    Parameters
    ----------
    chain : MarkovChainBuilder
    cfg : ScenarioConfig
    verbose : bool

    Returns
    -------
    MDPSolver with solved policy.
    """
    t0 = time.time()
    solver = MDPSolver(
        chain=chain,
        tau_initial=TAU_INITIAL,
        delta=DELTA,
        rho=cfg.rho,
        cost_c=cfg.cost_c,
        grid_size=cfg.grid_bp,
    )
    solver.solve()
    t1 = time.time()
    if verbose:
        print(f"  MDP solve: {t1-t0:.2f}s")
    return solver


def estimate_mvo(
    chain: MarkovChainBuilder,
    cfg: ScenarioConfig,
) -> MVOBenchmark:
    """Estimate MVO benchmark from independent simulation sample."""
    bench = MVOBenchmark.from_chain_simulation(
        chain=chain,
        tau_initial=TAU_INITIAL,
        N_sim=cfg.mvo_N_sim,
        seed=cfg.mvo_seed,
        gamma=cfg.gamma_mvo,
        grid_size=cfg.grid_bp,
        delta=DELTA,
    )
    return bench


def find_initial_state(chain: MarkovChainBuilder, model: DNSModel, cfg: ScenarioConfig) -> int:
    """Find the index of the initial state s_0 in chain.state_lists[0].

    Uses the DNS model to compute initial yields and finds the nearest
    retained state by Euclidean distance.

    Parameters
    ----------
    chain : MarkovChainBuilder
    model : DNSModel
    cfg : ScenarioConfig

    Returns
    -------
    int : index in chain.state_lists[0]
    """
    from scipy.spatial.distance import cdist

    F0 = model.initial_conditions[cfg.initial_curve]
    tau_0 = TAU_INITIAL   # at t=0 maturities = initial
    y0 = model.yields_from_factors(F0, tau_0)   # (K,)

    # Discretize y0
    coord_0 = tuple(int(round(v / cfg.grid_bp)) for v in y0)

    # Check if coord_0 is retained
    if coord_0 in chain.state_to_idx[0]:
        return chain.state_to_idx[0][coord_0]

    # Nearest retained state
    states_arr = np.array(chain.state_lists[0], dtype=float) * cfg.grid_bp  # (n_0, K)
    y0_arr = np.array([y0])
    dists = cdist(y0_arr, states_arr, metric="euclidean")[0]
    return int(np.argmin(dists))


def evaluate_all(
    chain: MarkovChainBuilder,
    solver: MDPSolver,
    mvo_bench: MVOBenchmark,
    cfg: ScenarioConfig,
    model: DNSModel,
    verbose: bool = False,
) -> dict:
    """Evaluate DP, equal-weight, and MVO on common random paths.

    Parameters
    ----------
    chain : MarkovChainBuilder
    solver : MDPSolver
    mvo_bench : MVOBenchmark
    cfg : ScenarioConfig
    model : DNSModel
    verbose : bool

    Returns
    -------
    dict with keys:
        'dp', 'ew', 'mvo': each a dict with terminal_wealth and metrics
        'paired_dp_vs_ew', 'paired_dp_vs_mvo': paired t-stats
        'seeds': list of evaluation seeds used
    """
    evaluator = StrategyEvaluator(
        chain=chain,
        tau_initial=TAU_INITIAL,
        delta=DELTA,
        cost_c=cfg.cost_c,
        grid_size=cfg.grid_bp,
    )
    simulator = PathSimulator(chain)
    ew_bench  = EqualWeightBenchmark(K=3)

    init_idx = find_initial_state(chain, model, cfg)

    all_tw_dp  = []
    all_tw_ew  = []
    all_tw_mvo = []
    all_wp_dp  = []
    all_wp_ew  = []
    all_wp_mvo = []
    all_costs  = []
    all_turns  = []

    base_eval_seed = 1000
    for seed_offset in range(cfg.N_eval_seeds):
        seed = base_eval_seed + seed_offset * 13
        paths = simulator.simulate_paths(init_idx, cfg.N_eval, seed=seed)

        if verbose:
            print(f"  Evaluating seed {seed}...")

        res_dp  = evaluator.evaluate_dp_vectorized(paths, solver)
        res_ew  = evaluator.evaluate_static_vectorized(paths, ew_bench.weights)
        res_mvo = evaluator.evaluate_static_vectorized(paths, mvo_bench.weights)

        all_tw_dp.append(res_dp["terminal_wealth"])
        all_tw_ew.append(res_ew["terminal_wealth"])
        all_tw_mvo.append(res_mvo["terminal_wealth"])
        all_wp_dp.append(res_dp["wealth_paths"])
        all_wp_ew.append(res_ew["wealth_paths"])
        all_wp_mvo.append(res_mvo["wealth_paths"])
        all_costs.append(res_dp["cumulative_costs"])
        all_turns.append(res_dp["turnovers"])

    # Pool across seeds
    tw_dp  = np.concatenate(all_tw_dp)
    tw_ew  = np.concatenate(all_tw_ew)
    tw_mvo = np.concatenate(all_tw_mvo)
    wp_dp  = np.concatenate(all_wp_dp, axis=0)
    wp_ew  = np.concatenate(all_wp_ew, axis=0)
    wp_mvo = np.concatenate(all_wp_mvo, axis=0)
    cum_costs = np.concatenate(all_costs)
    turnovers  = np.concatenate(all_turns)

    m_dp  = evaluator.compute_metrics(wp_dp)
    m_ew  = evaluator.compute_metrics(wp_ew)
    m_mvo = evaluator.compute_metrics(wp_mvo)

    m_dp["mean_turnover"] = float(np.mean(turnovers))
    m_dp["mean_cum_cost"] = float(np.mean(cum_costs))

    paired_ew  = evaluator.compute_paired_stats(tw_dp, tw_ew)
    paired_mvo = evaluator.compute_paired_stats(tw_dp, tw_mvo)

    return {
        "dp":               {"terminal_wealth": tw_dp, "metrics": m_dp, "wealth_paths": wp_dp},
        "ew":               {"terminal_wealth": tw_ew, "metrics": m_ew, "wealth_paths": wp_ew},
        "mvo":              {"terminal_wealth": tw_mvo, "metrics": m_mvo, "wealth_paths": wp_mvo},
        "mvo_weights":      mvo_bench.weights,
        "paired_dp_vs_ew":  paired_ew,
        "paired_dp_vs_mvo": paired_mvo,
        "seeds":            list(range(base_eval_seed, base_eval_seed + cfg.N_eval_seeds * 13, 13)),
    }


def run_scenario(
    cfg: ScenarioConfig,
    verbose: bool = True,
) -> dict:
    """Full pipeline: build chain → solve MDP → estimate MVO → evaluate.

    Parameters
    ----------
    cfg : ScenarioConfig
    verbose : bool

    Returns
    -------
    dict with 'chain', 'solver', 'mvo_bench', 'results', 'chain_diagnostics'
    """
    if verbose:
        print(f"\n=== Scenario: {cfg.label} ===")

    model = build_dns_model(cfg)
    chain, disc = build_chain(cfg, model, verbose=verbose)

    if verbose:
        diag = chain.compute_diagnostics()
        for t in diag:
            d = diag[t]
            rm = d["retained_mass_stats"]
            print(f"  Date {t}: states={d['n_states_after_cap']}, "
                  f"ret_mass_mean={rm['mean']:.4f}, "
                  f"rows<25_succ={d['n_rows_lt_25_successors']}")

    solver = solve_mdp(chain, cfg, verbose=verbose)
    mvo_bench = estimate_mvo(chain, cfg)

    if verbose:
        print(f"  MVO weights: {mvo_bench.weights}")

    results = evaluate_all(chain, solver, mvo_bench, cfg, model, verbose=verbose)

    chain_diag = chain.compute_diagnostics()

    return {
        "chain":             chain,
        "disc":              disc,
        "solver":            solver,
        "mvo_bench":         mvo_bench,
        "results":           results,
        "chain_diagnostics": chain_diag,
        "model":             model,
        "cfg":               cfg,
    }
