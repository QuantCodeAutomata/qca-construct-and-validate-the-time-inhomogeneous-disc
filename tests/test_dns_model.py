import numpy as np
import pytest
import sys
sys.path.insert(0, '/workspace/project')
from src.dns_model import DNSModel


def test_dns_model_default_init():
    model = DNSModel()
    assert model.lambda_param == pytest.approx(0.0609)
    assert len(model.mu) == 3


def test_yield_formula_at_infinity():
    """At tau -> infinity, yield should approach level factor."""
    model = DNSModel()
    # At very large tau, B1 -> 0, B2 -> 0, so y -> l
    y_large = model.yields_from_factors(np.array([0.05, -0.015, 0.01]), np.array([100.0]))
    assert y_large[0] == pytest.approx(0.05, abs=1e-3)


def test_yield_formula_at_zero():
    """At tau -> 0, yield should approach level + slope (l+s)."""
    model = DNSModel()
    # At tau->0: B1->1, B2->0, so y -> l + s
    y_short = model.yields_from_factors(np.array([0.05, -0.015, 0.01]), np.array([0.001]))
    assert y_short[0] == pytest.approx(0.05 - 0.015, abs=1e-2)


def test_simulate_paths_shape():
    model = DNSModel()
    paths = model.simulate_paths(N_paths=100, T_steps=6, seed=42)
    assert paths.shape == (100, 7, 3)


def test_simulate_paths_reproducibility():
    model = DNSModel()
    p1 = model.simulate_paths(100, 6, seed=42)
    p2 = model.simulate_paths(100, 6, seed=42)
    np.testing.assert_array_equal(p1, p2)


def test_rolling_maturities():
    model = DNSModel()
    tau_init = np.array([2.0, 5.0, 10.0])
    mats = model.rolling_maturities(tau_init, T_steps=6)
    # Shape should be (T+1, 3)
    assert mats.shape == (7, 3)
    # At t=0, should equal tau_init
    np.testing.assert_array_almost_equal(mats[0], tau_init)
    # All maturities should be > 0 for T<=12 months
    assert np.all(mats > 0)


def test_simulate_factor_mean():
    """Long-run factor mean should be close to mu after many steps."""
    model = DNSModel()
    paths = model.simulate_paths(N_paths=10000, T_steps=120, seed=0)
    # Mean of last time step should be close to mu
    final_mean = paths[:, -1, :].mean(axis=0)
    np.testing.assert_allclose(final_mean, model.mu, atol=0.005)
