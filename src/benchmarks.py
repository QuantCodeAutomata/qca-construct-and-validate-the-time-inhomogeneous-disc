"""Static buy-and-hold benchmark strategies for bond portfolio comparison.

Implements:
  - EqualWeightBenchmark: fixed (1/3, 1/3, 1/3) allocation, no rebalancing.
  - MVOBenchmark: long-only, fully invested mean-variance optimal static weights,
    estimated on an independent simulation sample using CVXPY.

Reference: Pericoli-Taboga (2016) evaluation framework.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

# Using CVXPY for long-only MVO — Context7 confirmed
try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:                        # pragma: no cover
    _CVXPY_AVAILABLE = False

# Fallback: scipy.optimize for constrained minimization
from scipy.optimize import minimize


class EqualWeightBenchmark:
    """Equal-weight buy-and-hold benchmark.

    Holds K = 3 bonds with initial weights (1/3, 1/3, 1/3).
    There is no rebalancing after inception, so turnover is zero post-t=0.

    Attributes
    ----------
    weights : np.ndarray, shape (K,)
        Initial portfolio weights.
    """

    def __init__(self, K: int = 3) -> None:
        self.weights: np.ndarray = np.full(K, 1.0 / K, dtype=float)


class MVOBenchmark:
    """Static long-only, fully invested mean-variance optimal benchmark.

    Solves the quadratic program:
        maximize  mu_r' w - (gamma/2) * w' Sigma w
        subject to  w >= 0,  sum(w) = 1

    where mu_r and Sigma are estimated from an independent simulation sample.

    Parameters
    ----------
    gamma : float
        Risk-aversion parameter. Default: 5.0.
    K : int
        Number of assets. Default: 3.

    Attributes
    ----------
    weights : np.ndarray or None
        Estimated MVO weights after calling fit().
    mu_r : np.ndarray or None
        Estimated mean returns.
    Sigma : np.ndarray or None
        Estimated return covariance.
    """

    def __init__(self, gamma: float = 5.0, K: int = 3) -> None:
        self.gamma: float = gamma
        self.K: int = K
        self.weights: Optional[np.ndarray] = None
        self.mu_r: Optional[np.ndarray] = None
        self.Sigma: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fitting from returns
    # ------------------------------------------------------------------

    def fit(self, returns_matrix: np.ndarray) -> None:
        """Estimate MVO weights from a matrix of per-asset returns.

        Parameters
        ----------
        returns_matrix : np.ndarray, shape (N_scenarios, K)
            Simulated horizon returns for each bond (total return over horizon).
        """
        mu_r  = np.mean(returns_matrix, axis=0)         # (K,)
        Sigma = np.cov(returns_matrix.T, ddof=1)        # (K, K)

        # Regularise for numerical stability
        Sigma += 1e-10 * np.eye(self.K)

        self.mu_r  = mu_r
        self.Sigma = Sigma
        self.weights = self._solve_mvo(mu_r, Sigma)

    def fit_from_wealth_paths(self, wealth_paths: np.ndarray) -> None:
        """Estimate MVO weights from per-bond wealth paths.

        Parameters
        ----------
        wealth_paths : np.ndarray, shape (N_scenarios, T+1, K)
            Per-bond wealth evolution. Entry [n, t, k] is the value of a
            unit initial investment in bond k at time t along path n.
        """
        # Horizon returns: W[:, T, :] / W[:, 0, :] - 1
        horizon_returns = wealth_paths[:, -1, :] / wealth_paths[:, 0, :] - 1.0
        self.fit(horizon_returns)

    # ------------------------------------------------------------------
    # Internal solver
    # ------------------------------------------------------------------

    def _solve_mvo(self, mu_r: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """Solve the mean-variance quadratic program.

        Tries CVXPY first; falls back to scipy if unavailable or failed.

        Returns
        -------
        np.ndarray, shape (K,)
            Optimal portfolio weights summing to 1, all >= 0.
        """
        if _CVXPY_AVAILABLE:
            try:
                w = cp.Variable(self.K)
                objective = cp.Maximize(
                    mu_r @ w - (self.gamma / 2.0) * cp.quad_form(w, Sigma)
                )
                constraints = [w >= 0, cp.sum(w) == 1]
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.ECOS, warm_start=False)

                if w.value is not None and np.all(np.isfinite(w.value)):
                    result = np.maximum(w.value, 0.0)
                    result /= result.sum()
                    return result
            except Exception:   # pragma: no cover
                pass

        # Scipy fallback: minimise -mu'w + gamma/2 * w'Sigma w
        def neg_utility(w: np.ndarray) -> float:
            return -mu_r @ w + (self.gamma / 2.0) * (w @ Sigma @ w)

        def neg_utility_grad(w: np.ndarray) -> np.ndarray:
            return -mu_r + self.gamma * Sigma @ w

        x0 = np.full(self.K, 1.0 / self.K)
        constraints_scipy = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0.0, 1.0)] * self.K
        result = minimize(
            neg_utility,
            x0,
            jac=neg_utility_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_scipy,
        )

        if result.success:
            sol = np.maximum(result.x, 0.0)
            sol /= sol.sum()
            return sol

        # Last resort: equal weights
        warnings.warn("MVO optimisation failed; falling back to equal weights.")
        return np.full(self.K, 1.0 / self.K)

    # ------------------------------------------------------------------
    # MVO from simulation for independent estimation sample
    # ------------------------------------------------------------------

    @classmethod
    def from_chain_simulation(
        cls,
        chain,
        tau_initial: np.ndarray,
        N_sim: int = 20_000,
        seed: int = 99,
        gamma: float = 5.0,
        grid_size: float = 0.004,
        delta: float = 1.0 / 12.0,
    ) -> "MVOBenchmark":
        """Construct and fit an MVOBenchmark from a chain simulation.

        Simulates independent paths on the Markov chain to estimate
        expected returns and covariance for the three bonds, then solves MVO.

        Parameters
        ----------
        chain : MarkovChainBuilder
            Fitted chain.
        tau_initial : np.ndarray, shape (K,)
            Initial bond maturities.
        N_sim : int
            Number of simulation paths for estimation. Default: 20,000.
        seed : int
            Random seed (must differ from evaluation seeds).
        gamma : float
            MVO risk-aversion parameter.
        grid_size : float
            Decimal yield grid size. Default: 0.004.
        delta : float
            Monthly step in years.

        Returns
        -------
        MVOBenchmark
            Fitted benchmark instance.
        """
        rng = np.random.default_rng(seed)
        T = chain.T
        K = len(tau_initial)

        # Simulate paths from an arbitrary initial state (most-visited at t=0)
        # Use the first state in state_lists[0] as starting point.
        start_idx = 0  # index of initial state in state_lists[0]

        # Simulate Markov chain paths
        paths = np.empty((N_sim, T + 1), dtype=int)
        paths[:, 0] = start_idx

        for t in range(T):
            n_states_t = len(chain.state_lists[t])
            for s_idx in range(n_states_t):
                # Identify paths currently in state s_idx
                mask = paths[:, t] == s_idx
                n_in_state = mask.sum()
                if n_in_state == 0:
                    continue
                trans = chain.transition_matrices[t].get(s_idx, {})
                if not trans:
                    # Stay: no valid transition (repair)
                    paths[mask, t + 1] = s_idx if s_idx < len(chain.state_lists[t + 1]) else 0
                    continue
                succ_idx  = np.array(list(trans.keys()), dtype=int)
                succ_prob = np.array(list(trans.values()), dtype=float)
                chosen    = rng.choice(succ_idx, size=n_in_state, p=succ_prob)
                paths[mask, t + 1] = chosen

        # Compute per-bond wealth evolution for each path
        def state_yields(t_: int, s_idx_: int) -> np.ndarray:
            return np.array(chain.state_lists[t_][s_idx_], dtype=float) * grid_size

        def bond_prices(t_: int, s_idx_: int) -> np.ndarray:
            mats = tau_initial - t_ * delta
            yields_ = state_yields(t_, s_idx_)
            return np.exp(-mats * yields_)

        # Horizon returns for each bond: product of price relatives
        horizon_returns = np.zeros((N_sim, K), dtype=float)
        for n in range(N_sim):
            wealth_k = np.ones(K)
            for t in range(T):
                s_t   = paths[n, t]
                s_tp1 = paths[n, t + 1]
                B_t   = bond_prices(t, s_t)
                B_tp1 = bond_prices(t + 1, s_tp1)
                wealth_k *= B_tp1 / B_t
            horizon_returns[n] = wealth_k - 1.0

        bench = cls(gamma=gamma, K=K)
        bench.fit(horizon_returns)
        return bench
