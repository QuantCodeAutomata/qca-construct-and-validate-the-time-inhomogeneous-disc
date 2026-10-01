"""State discretization utilities for the DNS Markov chain project.

Continuous DNS yield paths are mapped to a finite grid and then pruned to a
manageable number of states per date, enabling compact Markov chain estimation.

Constants
---------
GRID_BP   : float – grid spacing in decimal yields (40 bp = 0.004).
MAX_STATES : int  – maximum retained states per date.
K          : int  – dimensionality of the yield state vector.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

GRID_BP: float = 0.004   # 40 basis points in decimal
MAX_STATES: int = 80
K: int = 3               # level, slope, curvature


# ---------------------------------------------------------------------------
# Low-level discretizer
# ---------------------------------------------------------------------------

class StateDiscretizer:
    """Map continuous yield vectors to integer grid coordinates.

    Each yield dimension is independently rounded to the nearest multiple of
    *grid_size*, then represented as an integer coordinate.

    Parameters
    ----------
    grid_size : float
        Grid spacing in decimal yield units. Default: 0.004 (40 bp).
    """

    def __init__(self, grid_size: float = GRID_BP) -> None:
        self.grid_size: float = grid_size

    def discretize_yield(self, y: float) -> int:
        """Round a single decimal yield to its integer grid coordinate.

        Parameters
        ----------
        y : float
            Yield in decimal units.

        Returns
        -------
        int
            Integer grid coordinate: round(y / grid_size).
        """
        return int(round(y / self.grid_size))

    def discretize_vector(self, yield_vec: np.ndarray) -> tuple[int, int, int]:
        """Discretize a 3-element yield vector to integer coordinates.

        Parameters
        ----------
        yield_vec : np.ndarray, shape (3,)
            Yield vector [level, slope, curvature] in decimal units.

        Returns
        -------
        tuple of 3 ints
            Integer grid coordinates (coord_level, coord_slope, coord_curvature).
        """
        return tuple(int(round(v / self.grid_size)) for v in yield_vec)  # type: ignore[return-value]

    def coord_to_decimal(self, coord: tuple[int, int, int]) -> np.ndarray:
        """Convert integer grid coordinates back to decimal yield values.

        Parameters
        ----------
        coord : tuple of 3 ints
            Integer grid coordinates produced by :meth:`discretize_vector`.

        Returns
        -------
        np.ndarray, shape (3,)
            Decimal yield vector: coord * grid_size.
        """
        return np.array(coord, dtype=float) * self.grid_size


# ---------------------------------------------------------------------------
# Per-date discretizer with frequency-based pruning
# ---------------------------------------------------------------------------

class DatewiseDiscretizer:
    """Discretize DNS yield paths date-by-date with frequency-based state pruning.

    For each calendar date *t* the discretizer:

    1. Maps all path yield vectors to integer grid coordinates.
    2. Counts how many paths land on each coordinate (state frequency).
    3. If the number of distinct states exceeds *max_states*, retains only the
       *max_states* most frequent ones; ties broken lexicographically (smaller
       coordinate tuple preferred).
    4. Remaps every pruned state to the Euclidean-nearest retained state in 3-D
       integer coordinate space; ties broken lexicographically.

    After :meth:`fit`, the following attributes are available:

    Attributes
    ----------
    state_lists : list of list of tuple
        Length T+1. Each entry is the lexicographically sorted list of retained
        integer-coordinate tuples for that date.
    state_to_idx : list of dict
        Length T+1. Maps coord_tuple -> contiguous integer index (0-based).
    remap_tables : list of dict
        Length T+1. Maps pruned_coord_tuple -> nearest_retained_coord_tuple.
        Empty dict when no pruning occurred.

    Parameters
    ----------
    grid_size : float
        Grid spacing in decimal yield units. Default: 0.004 (40 bp).
    max_states : int
        Maximum number of retained states per date. Default: 80.
    """

    def __init__(self, grid_size: float = GRID_BP, max_states: int = MAX_STATES) -> None:
        self.grid_size: float = grid_size
        self.max_states: int = max_states
        self._disc: StateDiscretizer = StateDiscretizer(grid_size)

        # Populated by fit()
        self.state_lists:         list[list[tuple[int, ...]]] = []
        self.state_to_idx:        list[dict[tuple[int, ...], int]] = []
        self.remap_tables:        list[dict[tuple[int, ...], tuple[int, ...]]] = []
        self.n_states_before_cap: list[int] = []  # distinct states before pruning

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, continuous_paths: np.ndarray) -> None:
        """Build per-date state spaces from simulated yield paths.

        Parameters
        ----------
        continuous_paths : np.ndarray, shape (N_paths, T_steps+1, 3)
            Continuous DNS yield paths (decimal units).  The yield vector at
            each (path, date) is [level, slope, curvature].

        Returns
        -------
        None
            Results are stored in :attr:`state_lists`, :attr:`state_to_idx`,
            and :attr:`remap_tables`.
        """
        N_paths, T_plus_1, _ = continuous_paths.shape

        self.state_lists          = []
        self.state_to_idx         = []
        self.remap_tables         = []
        self.n_states_before_cap  = []

        for t in range(T_plus_1):
            # ----------------------------------------------------------------
            # Step 1 – discretize all paths at date t
            # ----------------------------------------------------------------
            # Vectorised integer rounding for speed.
            raw = continuous_paths[:, t, :]                     # (N_paths, 3)
            coords_int = np.round(raw / self.grid_size).astype(int)  # (N_paths, 3)
            coords: list[tuple[int, ...]] = [tuple(row) for row in coords_int]

            # ----------------------------------------------------------------
            # Step 2 – count state frequencies
            # ----------------------------------------------------------------
            freq: Counter = Counter(coords)
            n_unique = len(freq)
            self.n_states_before_cap.append(n_unique)

            # ----------------------------------------------------------------
            # Step 3 – prune to top max_states; tie-break: lex ascending
            # Sort key: (-frequency, coord_tuple) so most-frequent comes first;
            # among equal frequencies, lexicographically smaller coord wins.
            # ----------------------------------------------------------------
            ranked: list[tuple[int, ...]] = sorted(
                freq.keys(), key=lambda c: (-freq[c], c)
            )

            if n_unique > self.max_states:
                retained_set: set[tuple[int, ...]] = set(ranked[: self.max_states])
                pruned_set:   set[tuple[int, ...]] = set(ranked[self.max_states :])
            else:
                retained_set = set(ranked)
                pruned_set   = set()

            # Final lexicographic ordering of retained states → stable index map.
            retained_list: list[tuple[int, ...]] = sorted(retained_set)

            # ----------------------------------------------------------------
            # Step 4-5 – remap pruned states to nearest retained state;
            # tie-break: lexicographically smallest retained coord.
            # Because retained_list is lex-sorted, np.argmin on distance rows
            # returns the lex-smallest among distance-tied retained states.
            # ----------------------------------------------------------------
            remap: dict[tuple[int, ...], tuple[int, ...]] = {}

            if pruned_set:
                retained_arr = np.array(retained_list, dtype=float)   # (n_ret, 3)
                pruned_sorted = sorted(pruned_set)                     # lex order
                pruned_arr    = np.array(pruned_sorted, dtype=float)  # (n_pru, 3)

                # Euclidean distances in integer-coordinate space.
                dists = cdist(pruned_arr, retained_arr)                # (n_pru, n_ret)

                # argmin returns first minimum → lex smallest for ties.
                nearest_idx = np.argmin(dists, axis=1)
                for i, pruned_coord in enumerate(pruned_sorted):
                    remap[pruned_coord] = retained_list[nearest_idx[i]]

            # Build coord → contiguous index mapping.
            s2i: dict[tuple[int, ...], int] = {
                coord: idx for idx, coord in enumerate(retained_list)
            }

            self.state_lists.append(retained_list)
            self.state_to_idx.append(s2i)
            self.remap_tables.append(remap)

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform(self, continuous_paths: np.ndarray) -> np.ndarray:
        """Map continuous yield paths to per-date state indices.

        Applies the discretization, remap, and index look-up built by
        :meth:`fit`.

        Parameters
        ----------
        continuous_paths : np.ndarray, shape (N_paths, T_steps+1, 3)
            Continuous DNS yield paths in decimal units.

        Returns
        -------
        np.ndarray of int, shape (N_paths, T_steps+1)
            Integer state indices at each (path, date) pair.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        KeyError
            If a coordinate produced from *continuous_paths* is neither
            retained nor in the remap table for a given date (e.g. out-of-
            sample paths visit a novel grid cell).
        """
        if not self.state_lists:
            raise RuntimeError("fit() must be called before transform().")

        N_paths, T_plus_1, _ = continuous_paths.shape
        result = np.empty((N_paths, T_plus_1), dtype=int)

        for t in range(T_plus_1):
            raw = continuous_paths[:, t, :]                          # (N_paths, 3)
            coords_int = np.round(raw / self.grid_size).astype(int) # (N_paths, 3)

            remap  = self.remap_tables[t]
            s2i    = self.state_to_idx[t]

            for p in range(N_paths):
                coord: tuple[int, ...] = tuple(coords_int[p])
                # Apply remap if pruned; identity if already retained.
                coord = remap.get(coord, coord)
                result[p, t] = s2i[coord]

        return result
