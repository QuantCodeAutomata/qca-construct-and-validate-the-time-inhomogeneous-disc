"""Path simulation and strategy evaluation on discrete Markov chain.

Provides:
  - PathSimulator: generate Markov chain state paths using common random numbers.
  - StrategyEvaluator: evaluate DP, equal-weight, and MVO strategies and compute
    performance metrics (mean/median/Q5 terminal wealth, MDD, turnover, t-statistics).

Reference: Pericoli-Taboga (2016) evaluation framework.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PathSimulator:
    """Simulate Markov chain state-index paths from a given initial state.

    Parameters
    ----------
    chain : MarkovChainBuilder
        A fitted chain with transition_matrices and state_lists.
    """

    def __init__(self, chain) -> None:
        self.chain = chain

    def simulate_paths(
        self,
        initial_state_idx: int,
        N_paths: int,
        seed: int = 42,
    ) -> np.ndarray:
        """Simulate forward state-index paths from the initial state.

        Parameters
        ----------
        initial_state_idx : int
            Index of the starting state in chain.state_lists[0].
        N_paths : int
            Number of Monte Carlo paths.
        seed : int
            Random seed for reproducibility. Default: 42.

        Returns
        -------
        np.ndarray of int, shape (N_paths, T+1)
            State index paths. paths[n, t] is the index into state_lists[t]
            for path n at date t.
        """
        rng = np.random.default_rng(seed)
        T   = self.chain.T
        paths = np.empty((N_paths, T + 1), dtype=int)
        paths[:, 0] = initial_state_idx

        for t in range(T):
            # Identify unique states at current date and simulate next states
            unique_states = np.unique(paths[:, t])
            for s_idx in unique_states:
                mask = paths[:, t] == s_idx
                n_in = int(mask.sum())
                if n_in == 0:
                    continue
                trans = self.chain.transition_matrices[t].get(s_idx, {})
                if not trans:
                    # Fallback: stay (shouldn't happen after valid chain construction)
                    n_next = len(self.chain.state_lists[t + 1])
                    paths[mask, t + 1] = min(s_idx, n_next - 1)
                    continue
                succ_idx  = np.array(list(trans.keys()), dtype=int)
                succ_prob = np.array(list(trans.values()), dtype=float)
                paths[mask, t + 1] = rng.choice(succ_idx, size=n_in, p=succ_prob)

        return paths


class StrategyEvaluator:
    """Evaluate portfolio strategies on pre-simulated Markov chain paths.

    Parameters
    ----------
    chain : MarkovChainBuilder
        Fitted chain.
    tau_initial : np.ndarray, shape (K,)
        Initial bond maturities in years.
    delta : float
        Monthly time step in years. Default: 1/12.
    cost_c : float
        Proportional transaction cost for DP evaluation. Default: 0.001.
    grid_size : float
        Decimal yield grid size. Default: 0.004.
    """

    def __init__(
        self,
        chain,
        tau_initial: np.ndarray,
        delta: float = 1.0 / 12.0,
        cost_c: float = 0.001,
        grid_size: float = 0.004,
    ) -> None:
        self.chain       = chain
        self.tau_initial = np.asarray(tau_initial, dtype=float)
        self.delta       = delta
        self.cost_c      = cost_c
        self.grid_size   = grid_size
        self.K           = len(self.tau_initial)

        # Precompute bond prices for all (date, state)
        self._prices = self._precompute_prices()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precompute_prices(self) -> list[np.ndarray]:
        """Precompute bond prices prices[t][state_idx, k]."""
        prices = []
        T = self.chain.T
        for t in range(T + 1):
            mats = self.tau_initial - t * self.delta   # (K,)
            n_st = len(self.chain.state_lists[t])
            P = np.empty((n_st, self.K), dtype=float)
            for s_idx, coord in enumerate(self.chain.state_lists[t]):
                yields_s = np.array(coord, dtype=float) * self.grid_size
                P[s_idx] = np.exp(-mats * yields_s)
            prices.append(P)
        return prices

    def _state_yields(self, t: int, state_idx: int) -> np.ndarray:
        """Decimal yields for state_idx at date t."""
        coord = self.chain.state_lists[t][state_idx]
        return np.array(coord, dtype=float) * self.grid_size

    def _bond_prices(self, t: int, state_idx: int) -> np.ndarray:
        """Bond prices for state_idx at date t (cached)."""
        return self._prices[t][state_idx]

    # ------------------------------------------------------------------
    # DP strategy evaluation
    # ------------------------------------------------------------------

    def evaluate_dp(
        self,
        state_paths: np.ndarray,
        policy,  # MDPSolver instance
    ) -> dict:
        """Evaluate the DP policy on given state paths.

        Parameters
        ----------
        state_paths : np.ndarray of int, shape (N_paths, T+1)
            State index paths from PathSimulator.
        policy : MDPSolver
            Solved MDP policy with a get_action() method.

        Returns
        -------
        dict with keys:
            'terminal_wealth' : np.ndarray, shape (N_paths,)
            'wealth_paths'    : np.ndarray, shape (N_paths, T+1)
            'cumulative_costs': np.ndarray, shape (N_paths,)
            'turnovers'       : np.ndarray, shape (N_paths,)
        """
        N, T1 = state_paths.shape
        T = T1 - 1
        wealth_paths = np.ones((N, T1), dtype=float)
        cum_costs    = np.zeros(N, dtype=float)
        turnovers    = np.zeros(N, dtype=float)

        # At t=0: choose initial action, no cost
        s0 = state_paths[:, 0]
        # All paths start from same state (assumed); vectorised lookup
        # Map state index -> action for t=0
        unique_s0 = np.unique(s0)
        action_map_0 = {int(s): policy.get_action(0, int(s)) for s in unique_s0}
        prev_actions = np.array([action_map_0[int(s)] for s in s0], dtype=int)
        prev_states  = s0.copy()

        from src.mdp_solver import ACTIONS  # noqa: PLC0415

        for t in range(1, T1):
            s_curr = state_paths[:, t]
            s_prev = state_paths[:, t - 1]

            # Pre-rebalance weights
            B_prev = self._prices[t - 1]  # (n_states_{t-1}, K)
            B_curr = self._prices[t]      # (n_states_t, K)

            for n in range(N):
                sp   = int(s_prev[n])
                sc   = int(s_curr[n])
                pa   = int(prev_actions[n])
                x_prev_w  = ACTIONS[pa]
                B_p  = B_prev[sp]    # (K,)
                B_c  = B_curr[sc]    # (K,)

                # Pre-rebalance weights
                raw   = x_prev_w * (B_c / B_p)
                total = raw.sum()
                x_prime = raw / total if total > 0 else x_prev_w.copy()

                # Choose action
                if t < T:
                    a_idx = policy.get_action(t, sc, pa, sp)
                else:
                    # At T (last step), just collect returns with prev action
                    # But there's no rebalancing AT T, portfolio was set at T-1
                    # Actually we only rebalance at t=0..T-1 (T steps)
                    # and compute returns through step t
                    a_idx = pa  # hold previous action at terminal step

                x_new = ACTIONS[a_idx]

                # Transaction cost at t (rebalancing occurs at each non-terminal date)
                if t < T:
                    phi = self.cost_c * np.sum(np.abs(x_new - x_prime))
                else:
                    phi = 0.0

                # Portfolio return from t-1 to t using previous weights
                # (but new weights are chosen at t for the next period)
                # Wealth update: W_t = W_{t-1} * (1 + g_{t-1->t}) * (1 - phi_t)
                # where g uses x_prev_w and prices at t-1 and t
                # Actually the rebalancing at t means:
                # - Before rebalancing at t: value is W_{t-1} * (1 + portfolio return)
                # - After rebalancing: deduct phi_t cost
                # g from t-1 to t with weights x_prev (set at t-1)
                g = float(np.dot(x_prev_w, B_c / B_p - 1.0))
                W_gross = wealth_paths[n, t - 1] * (1.0 + g)

                if t < T:
                    wealth_paths[n, t] = W_gross * (1.0 - phi)
                    cum_costs[n]      += W_gross * phi
                    turnovers[n]      += np.sum(np.abs(x_new - x_prime))
                    prev_actions[n]    = a_idx
                    prev_states[n]     = sc
                else:
                    # Terminal period: collect return, no rebalancing
                    wealth_paths[n, t] = W_gross

        return {
            "terminal_wealth": wealth_paths[:, T],
            "wealth_paths":    wealth_paths,
            "cumulative_costs": cum_costs,
            "turnovers":       turnovers,
        }

    # ------------------------------------------------------------------
    # Static buy-and-hold evaluation
    # ------------------------------------------------------------------

    def evaluate_static(
        self,
        state_paths: np.ndarray,
        weights: np.ndarray,
    ) -> dict:
        """Evaluate a static buy-and-hold strategy on given state paths.

        No rebalancing occurs after t=0. Turnover is zero post-inception.

        Parameters
        ----------
        state_paths : np.ndarray of int, shape (N_paths, T+1)
        weights : np.ndarray, shape (K,)
            Initial static portfolio weights.

        Returns
        -------
        dict with keys:
            'terminal_wealth': np.ndarray, shape (N_paths,)
            'wealth_paths'   : np.ndarray, shape (N_paths, T+1)
        """
        N, T1 = state_paths.shape
        T = T1 - 1
        wealth_paths = np.ones((N, T1), dtype=float)

        # Dynamic weights evolve with market
        current_weights = np.tile(weights, (N, 1))  # (N, K)

        for t in range(1, T1):
            s_prev = state_paths[:, t - 1]
            s_curr = state_paths[:, t]

            # Compute per-path portfolio returns
            for n in range(N):
                sp  = int(s_prev[n])
                sc  = int(s_curr[n])
                B_p = self._prices[t - 1][sp]   # (K,)
                B_c = self._prices[t][sc]        # (K,)
                g   = float(np.dot(current_weights[n], B_c / B_p - 1.0))
                wealth_paths[n, t] = wealth_paths[n, t - 1] * (1.0 + g)

                # Update drifted weights (no rebalancing)
                raw   = current_weights[n] * (B_c / B_p)
                total = raw.sum()
                current_weights[n] = raw / total if total > 0 else current_weights[n]

        return {
            "terminal_wealth": wealth_paths[:, T],
            "wealth_paths":    wealth_paths,
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, wealth_paths: np.ndarray) -> dict:
        """Compute standard performance metrics from wealth paths.

        Parameters
        ----------
        wealth_paths : np.ndarray, shape (N_paths, T+1)
            Wealth evolution where wealth_paths[:, 0] = 1 and
            wealth_paths[:, T] = terminal wealth.

        Returns
        -------
        dict with keys:
            'mean_terminal'  : float
            'median_terminal': float
            'q5_terminal'    : float (5th percentile)
            'mean_mdd'       : float (mean max-drawdown in %)
        """
        tw = wealth_paths[:, -1]
        mean_tw   = float(np.mean(tw))
        median_tw = float(np.median(tw))
        q5_tw     = float(np.percentile(tw, 5))

        # Maximum drawdown: 100 * max_t(1 - W_t / max_{u<=t} W_u)
        running_max = np.maximum.accumulate(wealth_paths, axis=1)  # (N, T+1)
        drawdowns   = 1.0 - wealth_paths / np.where(running_max > 0, running_max, 1.0)
        mdd_per_path = np.max(drawdowns, axis=1) * 100.0           # (N,)
        mean_mdd    = float(np.mean(mdd_per_path))

        return {
            "mean_terminal":   mean_tw,
            "median_terminal": median_tw,
            "q5_terminal":     q5_tw,
            "mean_mdd":        mean_mdd,
        }

    # ------------------------------------------------------------------
    # Paired t-statistics
    # ------------------------------------------------------------------

    def compute_paired_stats(
        self,
        tw_dp: np.ndarray,
        tw_bench: np.ndarray,
    ) -> dict:
        """Compute paired-difference statistics between two terminal wealth arrays.

        Parameters
        ----------
        tw_dp : np.ndarray, shape (N,)
            Terminal wealth under the DP strategy.
        tw_bench : np.ndarray, shape (N,)
            Terminal wealth under the benchmark strategy.

        Returns
        -------
        dict with keys:
            'mean_diff' : float, mean(tw_dp - tw_bench)
            'std_error' : float, se = std(D) / sqrt(n)
            't_stat'    : float, mean_diff / se
            'n'         : int, number of paths
        """
        D  = tw_dp - tw_bench
        n  = len(D)
        mean_D = float(np.mean(D))
        se_D   = float(np.std(D, ddof=1) / np.sqrt(n))
        t_stat = mean_D / se_D if se_D > 0 else float("inf")
        return {"mean_diff": mean_D, "std_error": se_D, "t_stat": t_stat, "n": n}

    # ------------------------------------------------------------------
    # Vectorised static evaluation (faster)
    # ------------------------------------------------------------------

    def evaluate_static_vectorized(
        self,
        state_paths: np.ndarray,
        weights: np.ndarray,
    ) -> dict:
        """Vectorised static buy-and-hold evaluation (no per-path Python loop).

        Parameters
        ----------
        state_paths : np.ndarray of int, shape (N_paths, T+1)
        weights : np.ndarray, shape (K,)

        Returns
        -------
        Same dict as evaluate_static().
        """
        N, T1 = state_paths.shape
        T = T1 - 1

        # Build price arrays: prices_all[t] = prices[t] shape (n_states_t, K)
        # Extract prices along paths
        wealth_paths = np.ones((N, T1), dtype=float)

        # Current weights for all paths
        w_curr = np.tile(weights, (N, 1)).astype(float)  # (N, K)

        for t in range(1, T1):
            sp_arr = state_paths[:, t - 1]   # (N,)
            sc_arr = state_paths[:, t]        # (N,)

            B_prev_all = self._prices[t - 1][sp_arr]   # (N, K)
            B_curr_all = self._prices[t][sc_arr]        # (N, K)

            ratio = B_curr_all / B_prev_all             # (N, K)
            g_all = np.sum(w_curr * (ratio - 1.0), axis=1)  # (N,)
            wealth_paths[:, t] = wealth_paths[:, t - 1] * (1.0 + g_all)

            # Drift weights (no rebalancing)
            raw   = w_curr * ratio
            total = raw.sum(axis=1, keepdims=True)
            w_curr = raw / np.where(total > 0, total, 1.0)

        return {
            "terminal_wealth": wealth_paths[:, T],
            "wealth_paths":    wealth_paths,
        }

    # ------------------------------------------------------------------
    # Vectorised DP evaluation (faster)
    # ------------------------------------------------------------------

    def evaluate_dp_vectorized(
        self,
        state_paths: np.ndarray,
        policy,
    ) -> dict:
        """Vectorised DP policy evaluation (reduced Python overhead).

        Uses the same logic as evaluate_dp() but minimises per-path inner loops.

        Parameters
        ----------
        state_paths : np.ndarray of int, shape (N_paths, T+1)
        policy : MDPSolver

        Returns
        -------
        Same dict as evaluate_dp().
        """
        from src.mdp_solver import ACTIONS  # noqa: PLC0415

        N, T1 = state_paths.shape
        T = T1 - 1

        wealth_paths = np.ones((N, T1), dtype=float)
        cum_costs    = np.zeros(N, dtype=float)
        turnovers    = np.zeros(N, dtype=float)

        # t=0 actions (vectorised via precomputed map)
        s0 = state_paths[:, 0]
        unique_s0 = np.unique(s0)
        act0_map = {int(s): policy.get_action(0, int(s)) for s in unique_s0}
        prev_actions = np.array([act0_map[int(s)] for s in s0], dtype=int)  # (N,)
        prev_weights = ACTIONS[prev_actions]   # (N, K) – weights set at t=0

        for t in range(1, T1):
            s_prev_arr = state_paths[:, t - 1]
            s_curr_arr = state_paths[:, t]

            B_prev_all = self._prices[t - 1][s_prev_arr]  # (N, K)
            B_curr_all = self._prices[t][s_curr_arr]       # (N, K)

            # Portfolio return using previous weights
            ratio = B_curr_all / B_prev_all                # (N, K)
            g_all = np.sum(prev_weights * (ratio - 1.0), axis=1)  # (N,)
            W_gross = wealth_paths[:, t - 1] * (1.0 + g_all)

            if t < T:
                # Pre-rebalance weights
                raw   = prev_weights * ratio
                total = raw.sum(axis=1, keepdims=True)
                x_prime = raw / np.where(total > 0, total, 1.0)  # (N, K)

                # Choose new actions – per-path lookup
                new_actions = np.empty(N, dtype=int)
                for n in range(N):
                    new_actions[n] = policy.get_action(
                        t,
                        int(s_curr_arr[n]),
                        int(prev_actions[n]),
                        int(s_prev_arr[n]),
                    )
                x_new = ACTIONS[new_actions]   # (N, K)

                # Transaction costs
                phi_all = self.cost_c * np.sum(np.abs(x_new - x_prime), axis=1)  # (N,)
                wealth_paths[:, t] = W_gross * (1.0 - phi_all)
                cum_costs          += W_gross * phi_all
                turnovers          += np.sum(np.abs(x_new - x_prime), axis=1)

                prev_actions = new_actions
                prev_weights = x_new
            else:
                # Terminal: no rebalancing
                wealth_paths[:, t] = W_gross

        return {
            "terminal_wealth": wealth_paths[:, T],
            "wealth_paths":    wealth_paths,
            "cumulative_costs": cum_costs,
            "turnovers":        turnovers,
        }
