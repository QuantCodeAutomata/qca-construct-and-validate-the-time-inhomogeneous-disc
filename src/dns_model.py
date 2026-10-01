"""Dynamic Nelson-Siegel (DNS) factor model for yield curve simulation.

Parameters follow Pericoli-Taboga (2016): monthly frequency, decimal yield units.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


class DNSModel:
    """Dynamic Nelson-Siegel model for simulating yield curve factor paths.

    The state vector F_t = (level, slope, curvature) follows a VAR(1) process:

        F_t = (I - A) @ mu + A @ F_{t-1} + eta_t,   eta_t ~ N(0, Sigma_eta)

    Yields at maturity tau are recovered via the Nelson-Siegel loadings:

        y(tau) = l + s * B1(tau) + kappa * B2(tau)

    where B1(tau) = (1 - exp(-lambda*tau)) / (lambda*tau)
    and   B2(tau) = B1(tau) - exp(-lambda*tau).

    Parameters
    ----------
    lambda_param : float
        Decay parameter (monthly). Default: 0.0609 (Pericoli-Taboga 2016).
    mu : np.ndarray, shape (3,)
        Long-run mean of factors [level, slope, curvature] in decimal yields.
    A : np.ndarray, shape (3, 3)
        Monthly AR persistence matrix.
    sigma_eta : np.ndarray, shape (3,)
        Standard deviations of factor innovations in decimal units.
    """

    #: Volatility scenario presets (sigma_eta in decimal units).
    VOLATILITY_SCENARIOS: dict[str, np.ndarray] = {
        "low":      np.array([0.0020, 0.0040, 0.0085]),
        "baseline": np.array([0.0036, 0.0062, 0.0092]),
        "high":     np.array([0.0050, 0.0080, 0.0099]),
    }

    #: Initial factor vector presets (decimal yields).
    #: 'long_run_mean' is populated in __init__ to equal mu.
    INITIAL_CONDITIONS: dict[str, Optional[np.ndarray]] = {
        "long_run_mean":   None,                              # set dynamically to mu
        "upward_sloping":  np.array([0.0400, -0.0250,  0.0200]),
        "inverted":        np.array([0.0600,  0.0250, -0.0100]),
        "original":        np.array([0.0450, -0.0100,  0.0150]),
    }

    def __init__(
        self,
        lambda_param: float = 0.0609,
        mu: Optional[np.ndarray] = None,
        A: Optional[np.ndarray] = None,
        sigma_eta: Optional[np.ndarray] = None,
    ) -> None:
        """Initialise DNS model with Pericoli-Taboga (2016) defaults."""
        self.lambda_param: float = lambda_param

        self.mu: np.ndarray = (
            np.array([0.0530, -0.0180, 0.0170]) if mu is None else np.asarray(mu, dtype=float)
        )
        self.A: np.ndarray = (
            np.diag([0.9978, 0.9811, 0.9660]) if A is None else np.asarray(A, dtype=float)
        )
        self.sigma_eta: np.ndarray = (
            np.array([0.0036, 0.0062, 0.0092])
            if sigma_eta is None
            else np.asarray(sigma_eta, dtype=float)
        )

        # Populate the long_run_mean entry with the actual mu value.
        self.initial_conditions: dict[str, np.ndarray] = {
            k: (self.mu.copy() if v is None else v.copy())
            for k, v in self.INITIAL_CONDITIONS.items()
        }

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        N_paths: int,
        T_steps: int,
        F0: Optional[Union[np.ndarray, str]] = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Simulate factor paths under the DNS VAR(1) dynamics.

        Parameters
        ----------
        N_paths : int
            Number of independent Monte Carlo paths.
        T_steps : int
            Number of monthly time steps to simulate.
        F0 : array-like of shape (3,), str, or None
            Initial factor vector. If a string, it is looked up in
            ``self.initial_conditions``. Defaults to ``'long_run_mean'`` (= mu).
        seed : int
            Random seed for reproducibility. Default: 42.

        Returns
        -------
        np.ndarray, shape (N_paths, T_steps+1, 3)
            Simulated factor paths. Axis 1 indexes time (0 = initial state),
            axis 2 indexes [level, slope, curvature].
        """
        rng = np.random.default_rng(seed)

        if F0 is None:
            F0_vec = self.initial_conditions["long_run_mean"]
        elif isinstance(F0, str):
            F0_vec = self.initial_conditions.get(F0, self.initial_conditions["long_run_mean"])
        else:
            F0_vec = np.asarray(F0, dtype=float)

        # Pre-compute the constant drift term (I - A) @ mu.
        drift = (np.eye(3) - self.A) @ self.mu          # shape (3,)
        Sigma_eta = np.diag(self.sigma_eta ** 2)         # (3, 3) covariance matrix

        # Allocate output: paths[:, 0, :] = F0.
        paths = np.empty((N_paths, T_steps + 1, 3), dtype=float)
        paths[:, 0, :] = F0_vec[np.newaxis, :]

        # Draw all innovations in one call: shape (N_paths, T_steps, 3).
        eta = rng.multivariate_normal(np.zeros(3), Sigma_eta, size=(N_paths, T_steps))

        # Recurse: F_t = drift + A @ F_{t-1} + eta_t.
        # Batched matrix multiply: (N, 3) @ A.T == A @ each row (A diagonal => symmetric).
        for t in range(T_steps):
            paths[:, t + 1, :] = drift + paths[:, t, :] @ self.A.T + eta[:, t, :]

        return paths

    # ------------------------------------------------------------------
    # Yield curve
    # ------------------------------------------------------------------

    def yields_from_factors(
        self,
        factors: np.ndarray,
        maturities: Union[np.ndarray, list],
    ) -> np.ndarray:
        """Convert DNS factors to model yields at given maturities.

        Parameters
        ----------
        factors : np.ndarray, shape (..., 3)
            Factor vectors [level, slope, curvature]. Leading dimensions are
            broadcast-compatible with the maturity dimension.
        maturities : array-like, shape (K,)
            Maturities tau in years (or months, consistent with lambda_param).

        Returns
        -------
        np.ndarray, shape (..., K)
            Fitted yields in decimal units. The last dimension corresponds to
            the K maturities.

        Notes
        -----
        The Nelson-Siegel loadings are:
            B1(tau) = (1 - exp(-lambda*tau)) / (lambda*tau)   [limit 1 as tau->0]
            B2(tau) = B1(tau) - exp(-lambda*tau)               [limit 0 as tau->0]

        y(tau) = level + slope * B1(tau) + curvature * B2(tau)
        """
        factors = np.asarray(factors, dtype=float)
        maturities = np.asarray(maturities, dtype=float)   # shape (K,)

        lam = self.lambda_param
        exp_term = np.exp(-lam * maturities)               # (K,)

        # Numerically stable B1 with the limiting value of 1 at tau = 0.
        with np.errstate(invalid="ignore", divide="ignore"):
            B1 = np.where(
                maturities == 0.0,
                1.0,
                (1.0 - exp_term) / (lam * maturities),
            )                                               # (K,)
        B2 = B1 - exp_term                                 # (K,) – limit 0 at tau=0

        # Extract factors along last axis; keep trailing dimension for broadcasting.
        level    = factors[..., 0:1]                       # (..., 1)
        slope    = factors[..., 1:2]                       # (..., 1)
        curvature = factors[..., 2:3]                      # (..., 1)

        # Broadcast: (..., 1) * (K,) → (..., K).
        return level + slope * B1 + curvature * B2

    # ------------------------------------------------------------------
    # Rolling maturities
    # ------------------------------------------------------------------

    def rolling_maturities(
        self,
        tau_initial: Union[np.ndarray, list],
        T_steps: int,
        delta: float = 1.0 / 12.0,
    ) -> np.ndarray:
        """Compute time-varying maturities for a rolling bond portfolio.

        At each step t the maturity of each bond decreases by *delta* (one
        month by default), reflecting the passage of time.

        Parameters
        ----------
        tau_initial : array-like, shape (K,)
            Initial maturities of the K bonds at t = 0.
        T_steps : int
            Number of time steps.
        delta : float
            Time elapsed per step (in years). Default: 1/12 (one month).

        Returns
        -------
        np.ndarray, shape (T_steps+1, K)
            Maturities at each date.  Row t contains tau_initial - t * delta.
        """
        tau_initial = np.asarray(tau_initial, dtype=float)      # (K,)
        t_values = np.arange(T_steps + 1, dtype=float)[:, np.newaxis]  # (T+1, 1)
        return tau_initial[np.newaxis, :] - t_values * delta    # (T+1, K)
