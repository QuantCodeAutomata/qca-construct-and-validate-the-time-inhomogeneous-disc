"""Finite-horizon Markov decision process solver for bond portfolio allocation.

Implements backward-induction Bellman recursion over a discrete-state,
time-inhomogeneous Markov chain. The augmented state at each date t >= 1
is (current_state_idx, previous_action_idx, previous_state_idx), because
transaction costs depend on the pre-rebalancing drift from the previous step.

Reference: Pericoli-Taboga (2016)-style dynamic programming for bond portfolios.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# -------------------------------------------------------------------------
# Module-level action set (6 long-only, fully invested portfolios)
# -------------------------------------------------------------------------
ACTIONS: np.ndarray = np.array(
    [
        [1.0,        0.0,        0.0       ],  # pure 2Y
        [0.0,        1.0,        0.0       ],  # pure 5Y
        [0.0,        0.0,        1.0       ],  # pure 10Y
        [1.0 / 3.0,  1.0 / 3.0,  1.0 / 3.0],  # equal-weight
        [0.2,        0.3,        0.5       ],  # tilted long
        [0.5,        0.3,        0.2       ],  # tilted short
    ],
    dtype=float,
)
N_ACTIONS: int = len(ACTIONS)


def bond_price(maturity: float, yield_decimal: float) -> float:
    """Zero-coupon bond price under continuous compounding.

    Parameters
    ----------
    maturity : float
        Residual maturity in years (must be > 0).
    yield_decimal : float
        Bond yield in decimal form (e.g., 0.05 for 5%).

    Returns
    -------
    float
        Price B = exp(-maturity * yield).
    """
    return float(np.exp(-maturity * yield_decimal))


class MDPSolver:
    """Backward-induction solver for the bond-allocation MDP.

    The value function is risk-neutral expected terminal wealth minus 1.
    The augmented state space at t >= 1 is
    (curr_state_idx, prev_action_idx, prev_state_idx).

    Parameters
    ----------
    chain : MarkovChainBuilder
        A fitted chain with ``transition_matrices``, ``state_lists``, and ``T``.
    tau_initial : array-like, shape (K,)
        Initial maturities of the K = 3 bonds in years.
    delta : float
        Monthly time step (1/12 years). Default: 1/12.
    rho : float
        Continuation parameter in {0.90, 0.95, 1.00}. Default: 1.0.
    cost_c : float
        Proportional round-trip transaction cost. Default: 0.001.
    grid_size : float
        Decimal yield grid size matching the chain. Default: 0.004 (40 bp).
    """

    def __init__(
        self,
        chain,
        tau_initial: np.ndarray,
        delta: float = 1.0 / 12.0,
        rho: float = 1.0,
        cost_c: float = 0.001,
        grid_size: float = 0.004,
    ) -> None:
        self.chain      = chain
        self.tau_initial = np.asarray(tau_initial, dtype=float)
        self.delta      = delta
        self.rho        = rho
        self.cost_c     = cost_c
        self.grid_size  = grid_size
        self.K          = len(self.tau_initial)
        self.T          = chain.T

        # Populated by solve()
        # J[0]: dict {curr_state_idx: value}
        # J[t>0]: np.ndarray shape (n_states_t, N_ACTIONS, n_states_{t-1})
        self.J:      list = []
        # Policy:
        # policy[0]: np.ndarray shape (n_states_0,) -> action index
        # policy[t>0]: np.ndarray shape (n_states_t, N_ACTIONS, n_states_{t-1})
        self.policy: list = []
        self.prices: list[np.ndarray] = []  # prices[t] shape (n_states_t, K)

        self._precompute_prices()

    # ------------------------------------------------------------------
    # Precompute bond prices at every date and state
    # ------------------------------------------------------------------

    def _precompute_prices(self) -> None:
        """Precompute bond prices for all dates and states.

        prices[t] is a float array of shape (n_states_t, K) where
        prices[t][s, k] = exp(-(tau_k - t*delta) * yield_{t,s,k}).
        Yields are decoded from integer coordinate tuples as coord * grid_size.
        """
        self.prices = []
        for t in range(self.T + 1):
            n_states_t = len(self.chain.state_lists[t])
            P = np.empty((n_states_t, self.K), dtype=float)
            maturities_t = self.tau_initial - t * self.delta   # (K,)
            for s_idx, coord_tuple in enumerate(self.chain.state_lists[t]):
                yields_s = np.array(coord_tuple, dtype=float) * self.grid_size  # (K,)
                P[s_idx] = np.exp(-maturities_t * yields_s)
            self.prices.append(P)

    # ------------------------------------------------------------------
    # Helper calculations
    # ------------------------------------------------------------------

    def _pre_rebalance_weights(
        self,
        prev_weights: np.ndarray,
        prev_state_idx: int,
        curr_state_idx: int,
        t: int,
    ) -> np.ndarray:
        """Compute pre-rebalancing portfolio weights after market moves.

        After holding prev_weights at date t-1 (state prev_state_idx),
        prices move and the portfolio drifts. Compute the new fractions
        before any rebalancing.

        Parameters
        ----------
        prev_weights : np.ndarray, shape (K,)
            Portfolio weights at date t-1.
        prev_state_idx : int
            State index at date t-1.
        curr_state_idx : int
            State index at date t (after market move).
        t : int
            Current date (>= 1).

        Returns
        -------
        np.ndarray, shape (K,)
            Drift-adjusted weights that sum to 1.
        """
        B_prev = self.prices[t - 1][prev_state_idx]   # (K,)
        B_curr = self.prices[t][curr_state_idx]        # (K,)
        raw = prev_weights * (B_curr / B_prev)
        total = raw.sum()
        return raw / total if total > 0 else prev_weights.copy()

    def _transaction_cost(
        self,
        new_weights: np.ndarray,
        pre_weights: np.ndarray,
    ) -> float:
        """Proportional transaction cost fraction of wealth.

        phi = cost_c * sum_k |new_weights_k - pre_weights_k|

        Parameters
        ----------
        new_weights : np.ndarray, shape (K,)
            Chosen portfolio weights after rebalancing.
        pre_weights : np.ndarray, shape (K,)
            Pre-rebalancing drift-adjusted weights.

        Returns
        -------
        float
            Cost fraction phi in [0, 2*cost_c].
        """
        return float(self.cost_c * np.sum(np.abs(new_weights - pre_weights)))

    def _one_step_return(
        self,
        weights: np.ndarray,
        curr_state_idx: int,
        next_state_idx: int,
        t: int,
    ) -> float:
        """Compute one-step portfolio return from t to t+1.

        g = sum_k weights_k * (B_{t+1,k} / B_{t,k} - 1)

        Parameters
        ----------
        weights : np.ndarray, shape (K,)
            Portfolio weights at date t.
        curr_state_idx : int
            State index at date t.
        next_state_idx : int
            State index at date t+1.
        t : int
            Current date.

        Returns
        -------
        float
            Portfolio return g.
        """
        B_curr = self.prices[t][curr_state_idx]       # (K,)
        B_next = self.prices[t + 1][next_state_idx]   # (K,)
        return float(np.dot(weights, B_next / B_curr - 1.0))

    # ------------------------------------------------------------------
    # Backward induction
    # ------------------------------------------------------------------

    def solve(self) -> None:
        """Solve the MDP by backward induction using vectorised numpy operations.

        Key identity used for efficiency:
        J_t(s, prev_a, prev_s) = max_{new_a} {(1 - phi[new_a,prev_a,prev_s,s]) * A[s,new_a] - 1}
        where A[s,a] = sum_{s'} p(s'|s) * (1+g(s,a,s')) * (1+rho*J_{t+1}(s',a,s))

        phi factors out of the sum because it depends only on (s,prev_s,prev_a,new_a),
        not on the next state s'. This reduces per-date complexity by ~n_succ factor.

        Populates:
        self.J[t] : np.ndarray, shape (n_t, N_ACTIONS, n_{t-1}) for t >= 1
                    np.ndarray, shape (n_0,) for t = 0
        self.policy[t] : same shape as J[t], stores argmax action index
        """
        T = self.T
        self.J      = [None] * (T + 1)
        self.policy = [None] * T

        # Terminal condition: J[T] = 0 everywhere
        n_T = len(self.chain.state_lists[T])
        if T > 0:
            n_Tm1 = len(self.chain.state_lists[T - 1])
            self.J[T] = np.zeros((n_T, N_ACTIONS, n_Tm1), dtype=float)

        # ------------------------------------------------------------------
        # Backward pass: t = T-1 down to 1
        # ------------------------------------------------------------------
        for t in range(T - 1, 0, -1):
            n_t   = len(self.chain.state_lists[t])
            n_tm1 = len(self.chain.state_lists[t - 1])
            J_next = self.J[t + 1]   # (n_{t+1}, N_A, n_t)

            B_curr = self.prices[t]       # (n_t, K)
            B_next = self.prices[t + 1]   # (n_{t+1}, K)
            B_prev = self.prices[t - 1]   # (n_tm1, K)

            # ----------------------------------------------------------
            # Step 1: Compute A[s, new_a] = sum_s' p * (1+g)*(1+rho*J_next)
            # ----------------------------------------------------------
            A = np.zeros((n_t, N_ACTIONS), dtype=float)
            for s_idx in range(n_t):
                successors = self.chain.transition_matrices[t].get(s_idx, {})
                if not successors:
                    continue
                succ_idx  = np.array(list(successors.keys()), dtype=int)
                succ_prob = np.array(list(successors.values()), dtype=float)

                ratio_s = B_next[succ_idx] / B_curr[s_idx]    # (n_succ, K)
                # g[new_a, s'] = ACTIONS @ ratio_s.T - 1
                g_s = ACTIONS @ ratio_s.T - 1.0                # (N_A, n_succ)

                # J_next[s', new_a, s_idx_as_prev] shape: (n_succ, N_A)
                J_next_s = J_next[succ_idx, :, s_idx]          # (n_succ, N_A)

                # factor[new_a, s'] = (1+g) * (1 + rho*J_next)
                factor = (1.0 + g_s) * (1.0 + self.rho * J_next_s.T)  # (N_A, n_succ)
                A[s_idx] = factor @ succ_prob                  # (N_A,)

            # ----------------------------------------------------------
            # Step 2: Compute phi[new_a, prev_a, prev_s, s]
            # ----------------------------------------------------------
            # x_prime[prev_a, prev_s, s] = normalise(ACTIONS[prev_a] * B_curr[s]/B_prev[prev_s])
            # shape: (N_A_prev, n_tm1, n_t, K)
            ratio_prev = (
                B_curr[np.newaxis, np.newaxis, :, :]    # (1, 1, n_t, K)
                / B_prev[np.newaxis, :, np.newaxis, :]  # (1, n_tm1, 1, K)
            )
            x_prime_raw = ACTIONS[:, np.newaxis, np.newaxis, :] * ratio_prev
            x_prime_sum = x_prime_raw.sum(axis=-1, keepdims=True)
            x_prime = x_prime_raw / np.where(x_prime_sum > 0, x_prime_sum, 1.0)
            # (N_A_prev, n_tm1, n_t, K)

            # phi[new_a, prev_a, prev_s, s] = cost_c * |ACTIONS[new_a] - x_prime[prev_a,prev_s,s]|
            diff = (
                ACTIONS[:, np.newaxis, np.newaxis, np.newaxis, :]   # (N_A_new,1,1,1,K)
                - x_prime[np.newaxis, :, :, :, :]                   # (1,N_A_prev,n_tm1,n_t,K)
            )
            phi_table = self.cost_c * np.sum(np.abs(diff), axis=-1)
            # (N_A_new, N_A_prev, n_tm1, n_t)

            # ----------------------------------------------------------
            # Step 3: value_all[s, new_a, prev_a, prev_s]
            #       = (1 - phi[new_a,prev_a,prev_s,s]) * A[s,new_a] - 1
            # ----------------------------------------------------------
            # Rearrange phi to (n_t, N_A_new, N_A_prev, n_tm1)
            phi_t = phi_table.transpose(3, 0, 1, 2)      # (n_t, N_A_new, N_A_prev, n_tm1)

            # A: (n_t, N_A_new, 1, 1)
            A_exp = A[:, :, np.newaxis, np.newaxis]

            value_all = (1.0 - phi_t) * A_exp - 1.0      # (n_t, N_A_new, N_A_prev, n_tm1)

            # J_t[s, prev_a, prev_s] = max_{new_a} value_all[s, new_a, prev_a, prev_s]
            J_t      = np.max(value_all, axis=1)          # (n_t, N_A_prev, n_tm1)
            policy_t = np.argmax(value_all, axis=1)       # (n_t, N_A_prev, n_tm1)

            self.J[t]      = J_t
            self.policy[t] = policy_t

        # ------------------------------------------------------------------
        # t = 0: no augmented state, no transaction cost
        # J_0(s_0) = max_a sum_s' p * [(1+g_0)*(1+rho*J_1) - 1]
        #          = max_a { A_0[s_0, a] - 1 }
        # ------------------------------------------------------------------
        n_0 = len(self.chain.state_lists[0])
        J_1 = self.J[1]   # shape (n_1, N_A, n_0)

        B_0 = self.prices[0]   # (n_0, K)
        B_1 = self.prices[1]   # (n_1, K)

        A_0 = np.zeros((n_0, N_ACTIONS), dtype=float)
        for s0_idx in range(n_0):
            successors = self.chain.transition_matrices[0].get(s0_idx, {})
            if not successors:
                continue
            succ_idx  = np.array(list(successors.keys()), dtype=int)
            succ_prob = np.array(list(successors.values()), dtype=float)

            ratio_0 = B_1[succ_idx] / B_0[s0_idx]        # (n_succ, K)
            g_0     = ACTIONS @ ratio_0.T - 1.0           # (N_A, n_succ)

            if T >= 2 and J_1 is not None:
                J1_vals = J_1[succ_idx, :, s0_idx]        # (n_succ, N_A)
            else:
                J1_vals = np.zeros((len(succ_idx), N_ACTIONS), dtype=float)

            # A_0[s0, a] = sum_s' p_s' * (1+g_0[a,s']) * (1+rho*J1[s',a,s0])
            factor_0 = (1.0 + g_0) * (1.0 + self.rho * J1_vals.T)  # (N_A, n_succ)
            A_0[s0_idx] = factor_0 @ succ_prob             # (N_A,)

        J_0      = A_0.max(axis=1) - 1.0
        policy_0 = A_0.argmax(axis=1)

        self.J[0]      = J_0
        self.policy[0] = policy_0

    # ------------------------------------------------------------------
    # Policy lookup
    # ------------------------------------------------------------------

    def get_action(
        self,
        t: int,
        curr_state_idx: int,
        prev_action_idx: Optional[int] = None,
        prev_state_idx: Optional[int] = None,
    ) -> int:
        """Retrieve the optimal action at time t for the given augmented state.

        Parameters
        ----------
        t : int
            Current time step.
        curr_state_idx : int
            Index of the current discrete state.
        prev_action_idx : int or None
            Index of the previous action (required for t >= 1).
        prev_state_idx : int or None
            Index of the previous state (required for t >= 1).

        Returns
        -------
        int
            Index into the ACTIONS array for the optimal action.
        """
        if self.policy[t] is None:
            raise RuntimeError("Policy not yet solved; call solve() first.")
        if t == 0:
            return int(self.policy[0][curr_state_idx])
        else:
            if prev_action_idx is None or prev_state_idx is None:
                raise ValueError("prev_action_idx and prev_state_idx required for t >= 1.")
            return int(self.policy[t][curr_state_idx, prev_action_idx, prev_state_idx])
