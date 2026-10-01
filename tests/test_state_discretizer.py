import numpy as np
import pytest
import sys
sys.path.insert(0, '/workspace/project')
from src.state_discretizer import StateDiscretizer, DatewiseDiscretizer


def test_discretize_yield_rounding():
    sd = StateDiscretizer(grid_size=0.004)
    # Exact multiples – no ambiguity
    assert sd.discretize_yield(0.04)  == 10   # 0.04  / 0.004 = 10.0
    assert sd.discretize_yield(0.044) == 11   # 0.044 / 0.004 = 11.0
    assert sd.discretize_yield(0.036) == 9    # 0.036 / 0.004 =  9.0
    assert sd.discretize_yield(0.000) == 0
    # Near-half: closer to lower integer
    assert sd.discretize_yield(0.041) == 10   # 0.041 / 0.004 = 10.25 → 10
    # Near-half: closer to upper integer
    assert sd.discretize_yield(0.043) == 11   # 0.043 / 0.004 = 10.75 → 11


def test_coord_to_decimal_roundtrip():
    sd = StateDiscretizer(grid_size=0.004)
    original = np.array([0.036, 0.040, 0.044])
    coords = sd.discretize_vector(original)
    recovered = sd.coord_to_decimal(coords)
    np.testing.assert_allclose(recovered, original, atol=1e-10)


def test_datewise_discretizer_state_cap():
    """Number of states per date should not exceed max_states."""
    from src.dns_model import DNSModel
    model = DNSModel()
    paths_factors = model.simulate_paths(1000, 6, seed=1)
    # Convert to yields
    tau = model.rolling_maturities(np.array([2.0, 5.0, 10.0]), 6)
    N, T1, K = paths_factors.shape
    yield_paths = np.zeros((N, T1, K))
    for i in range(T1):
        yield_paths[:, i, :] = model.yields_from_factors(paths_factors[:, i, :], tau[i])
    dd = DatewiseDiscretizer(grid_size=0.004, max_states=80)
    dd.fit(yield_paths)
    for t, sl in enumerate(dd.state_lists):
        assert len(sl) <= 80, f"Date {t} has {len(sl)} states > 80"


def test_datewise_discretizer_transform_shape():
    from src.dns_model import DNSModel
    model = DNSModel()
    paths_factors = model.simulate_paths(500, 6, seed=2)
    tau = model.rolling_maturities(np.array([2.0, 5.0, 10.0]), 6)
    N, T1, K = paths_factors.shape
    yield_paths = np.zeros((N, T1, K))
    for i in range(T1):
        yield_paths[:, i, :] = model.yields_from_factors(paths_factors[:, i, :], tau[i])
    dd = DatewiseDiscretizer(grid_size=0.004, max_states=80)
    dd.fit(yield_paths)
    idx_paths = dd.transform(yield_paths)
    assert idx_paths.shape == (N, T1)
    # All indices should be valid
    for t in range(T1):
        assert np.all(idx_paths[:, t] >= 0)
        assert np.all(idx_paths[:, t] < len(dd.state_lists[t]))
