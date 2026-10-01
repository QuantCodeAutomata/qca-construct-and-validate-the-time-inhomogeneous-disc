"""Time-inhomogeneous Markov chain builder for the DNS bond yield model.

Transition matrices are estimated from discretized Monte Carlo paths.
Each date has its own transition matrix (time-inhomogeneous chain).
To keep the model tractable, at most *max_successors* successor states are
retained per origin state; remaining mass is redistributed by renormalisation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import numpy as np


class MarkovChainBuilder:
    """Estimate time-inhomogeneous Markov transition matrices from index paths.

    Parameters
    ----------
    max_states : int
        Maximum number of states per date (informational; actual cap is
        enforced by :class:`~state_discretizer.DatewiseDiscretizer`).
        Default: 80.
    max_successors : int
        Maximum number of successor states kept per origin state.
        Default: 25.

    Attributes (set after :meth:`build`)
    ------------------------------------
    transition_matrices : list of dict
        Length T.  Each entry is ``{origin_idx: {succ_idx: prob}}``, a
        sparse representation of the row-stochastic transition matrix at
        time t.
    state_lists : list of list
        Reference to the state_lists passed to :meth:`build`.
    state_to_idx : list of dict
        Reference to the state_to_idx passed to :meth:`build`.
    T : int
        Horizon (number of transition steps).
    """

    def __init__(self, max_states: int = 80, max_successors: int = 25) -> None:
        self.max_states:     int = max_states
        self.max_successors: int = max_successors

        # Populated by build()
        self.transition_matrices: list[dict[int, dict[int, float]]] = []
        self.state_lists:  list  = []
        self.state_to_idx: list  = []
        self.T:            int   = 0

        # Internal diagnostics bookkeeping
        self._retained_masses_per_t: list[dict[int, float]] = []
        self._n_successors_per_row:  list[dict[int, int]]   = []
        self._n_states_before_cap:   Optional[list[int]]    = None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(
        self,
        remapped_idx_paths: np.ndarray,
        T: int,
        state_lists: list,
        state_to_idx: list,
        n_states_before_cap: Optional[list[int]] = None,
    ) -> None:
        """Estimate transition matrices from discretised, index-coded paths.

        Parameters
        ----------
        remapped_idx_paths : np.ndarray of int, shape (N_paths, T+1)
            Integer state indices at each (path, date); output of
            :meth:`~state_discretizer.DatewiseDiscretizer.transform`.
        T : int
            Horizon: number of one-step transitions (= T_steps).
        state_lists : list of list
            Per-date lists of retained integer-coordinate tuples, as produced
            by :class:`~state_discretizer.DatewiseDiscretizer`.
        state_to_idx : list of dict
            Per-date mappings coord_tuple -> index, parallel to *state_lists*.
        n_states_before_cap : list of int or None
            Optional.  Number of distinct discretised states at each date
            *before* the max_states cap, as recorded by
            :class:`~state_discretizer.DatewiseDiscretizer`.  Used only by
            :meth:`compute_diagnostics`.  Length must be T+1 if provided.

        Returns
        -------
        None
            Results are stored in :attr:`transition_matrices`,
            :attr:`state_lists`, :attr:`state_to_idx`, and :attr:`T`.
        """
        self.state_lists         = state_lists
        self.state_to_idx        = state_to_idx
        self.T                   = T
        self._n_states_before_cap = n_states_before_cap

        self.transition_matrices   = []
        self._retained_masses_per_t = []
        self._n_successors_per_row  = []

        for t in range(T):
            n_t  = len(state_lists[t])
            n_t1 = len(state_lists[t + 1])

            # ----------------------------------------------------------------
            # Step 1 – count transitions s → s' at time t
            # ----------------------------------------------------------------
            origins    = remapped_idx_paths[:, t]
            successors = remapped_idx_paths[:, t + 1]

            # Count (origin, successor) pairs efficiently.
            pair_counts: Counter = Counter(
                zip(origins.tolist(), successors.tolist())
            )

            # Build per-origin successor counts: origin_idx -> {succ_idx: count}
            origin_counts: defaultdict[int, dict[int, int]] = defaultdict(dict)
            for (orig, succ), cnt in pair_counts.items():
                origin_counts[orig][succ] = cnt

            # ----------------------------------------------------------------
            # Step 2-3 – estimate p(s'|s), keep top max_successors, renormalise
            # ----------------------------------------------------------------
            trans_t:            dict[int, dict[int, float]] = {}
            retained_masses_t:  dict[int, float]            = {}
            n_successors_t:     dict[int, int]              = {}

            for orig_idx in range(n_t):
                if orig_idx not in origin_counts:
                    # --------------------------------------------------------
                    # Repair: origin state observed in state_lists but has no
                    # transitions (should not occur with training data).
                    # Assign uniform distribution over all t+1 states.
                    # --------------------------------------------------------
                    n_succ = n_t1
                    trans_t[orig_idx]           = {j: 1.0 / n_t1 for j in range(n_t1)}
                    retained_masses_t[orig_idx] = 1.0
                    n_successors_t[orig_idx]    = n_succ
                    continue

                succ_counts: dict[int, int] = origin_counts[orig_idx]
                N_s: int = sum(succ_counts.values())

                # Sort by count descending; tie-break by succ_idx ascending for
                # deterministic selection.
                sorted_succs: list[tuple[int, int]] = sorted(
                    succ_counts.items(), key=lambda x: (-x[1], x[0])
                )

                # Compute retained mass fraction before capping.
                if len(sorted_succs) <= self.max_successors:
                    retained_mass: float = 1.0
                    top_succs = sorted_succs
                else:
                    top_succs = sorted_succs[: self.max_successors]
                    top_count: int = sum(c for _, c in top_succs)
                    retained_mass = top_count / N_s

                retained_masses_t[orig_idx] = retained_mass
                n_successors_t[orig_idx]    = len(top_succs)

                # Renormalise over the retained successor set.
                top_count_total: int = sum(c for _, c in top_succs)
                trans_t[orig_idx] = {
                    s_idx: c / top_count_total for s_idx, c in top_succs
                }

            self.transition_matrices.append(trans_t)
            self._retained_masses_per_t.append(retained_masses_t)
            self._n_successors_per_row.append(n_successors_t)

    # ------------------------------------------------------------------
    # Dense matrix retrieval
    # ------------------------------------------------------------------

    def get_transition_matrix_dense(self, t: int) -> np.ndarray:
        """Return the transition matrix at time t as a dense 2-D array.

        Parameters
        ----------
        t : int
            Time index, 0 <= t < T.

        Returns
        -------
        np.ndarray, shape (n_states_t, n_states_{t+1})
            Row-stochastic transition matrix.  Entry [i, j] is the
            probability of moving from state i at date t to state j at
            date t+1.  Rows for states with no observed transitions are
            filled with a uniform distribution (repair).
        """
        n_t  = len(self.state_lists[t])
        n_t1 = len(self.state_lists[t + 1])
        M    = np.zeros((n_t, n_t1), dtype=float)

        for orig_idx, successors in self.transition_matrices[t].items():
            for succ_idx, prob in successors.items():
                M[orig_idx, succ_idx] = prob

        return M

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def retained_mass_per_row(self, t: int) -> np.ndarray:
        """Pre-renormalisation retained probability mass for each origin state.

        For each origin state at date t, this is the fraction of total
        transition counts captured by the top *max_successors* successors
        before renormalisation.  The value equals 1.0 for states with at
        most *max_successors* distinct successors.

        Parameters
        ----------
        t : int
            Time index, 0 <= t < T.

        Returns
        -------
        np.ndarray, shape (n_states_t,)
            Retained mass fractions in [0, 1].
        """
        n_t    = len(self.state_lists[t])
        masses = self._retained_masses_per_t[t]
        # Default to 1.0 for any state not explicitly recorded (uniform repair).
        return np.array([masses.get(i, 1.0) for i in range(n_t)], dtype=float)

    def compute_diagnostics(self) -> dict[int, dict]:
        """Compute per-date quality diagnostics for the estimated chain.

        Returns
        -------
        dict keyed by date index t (0 <= t < T)
            Each value is a dict with:

            ``n_states_before_cap`` : int or None
                Number of distinct discretised states at date t before the
                max_states cap (sourced from ``n_states_before_cap`` argument
                of :meth:`build`; ``None`` if not provided).
            ``n_states_after_cap`` : int
                Number of retained states at date t (= ``len(state_lists[t])``).
            ``retained_mass_stats`` : dict
                Summary statistics of retained-mass fractions across all origin
                states: ``{'mean': float, 'min': float, 'max': float}``.
            ``n_rows_lt_25_successors`` : int
                Number of origin states with fewer than *max_successors*
                successors after the successor cap.
            ``n_empty_rows`` : int
                Number of origin states with 0 successors after repair
                (should be 0).
        """
        diags: dict[int, dict] = {}

        for t in range(self.T):
            n_t    = len(self.state_lists[t])
            masses = self.retained_mass_per_row(t)                 # (n_t,)
            n_succs = [self._n_successors_per_row[t].get(i, 0) for i in range(n_t)]

            before_cap = (
                self._n_states_before_cap[t]
                if self._n_states_before_cap is not None
                else None
            )

            diags[t] = {
                "n_states_before_cap": before_cap,
                "n_states_after_cap":  n_t,
                "retained_mass_stats": {
                    "mean": float(np.mean(masses)),
                    "min":  float(np.min(masses)),
                    "max":  float(np.max(masses)),
                },
                "n_rows_lt_25_successors": int(
                    sum(1 for n in n_succs if n < self.max_successors)
                ),
                "n_empty_rows": int(
                    sum(1 for n in n_succs if n == 0)
                ),
            }

        return diags

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Verify that the estimated chain is a valid row-stochastic process.

        Checks
        ------
        * Every origin state at every date has at least one successor.
        * Every row of every transition matrix sums to 1 within tolerance 1e-12.

        Returns
        -------
        bool
            ``True`` if all checks pass, ``False`` otherwise.
        """
        for t in range(self.T):
            n_t = len(self.state_lists[t])
            for orig_idx in range(n_t):
                row = self.transition_matrices[t].get(orig_idx)
                if not row:
                    # Empty row – repair was expected to prevent this.
                    return False
                total = sum(row.values())
                if abs(total - 1.0) > 1e-12:
                    return False
        return True
