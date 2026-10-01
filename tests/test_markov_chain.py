import numpy as np
import pytest
import sys
sys.path.insert(0, '/workspace/project')
from src.dns_model import DNSModel
from src.state_discretizer import DatewiseDiscretizer
from src.markov_chain import MarkovChainBuilder


def build_chain_for_test(N=2000, T=6, seed=42):
    model = DNSModel()
    paths_f = model.simulate_paths(N, T, seed=seed)
    tau = model.rolling_maturities(np.array([2.0, 5.0, 10.0]), T)
    N_, T1, K = paths_f.shape
    yield_paths = np.zeros((N_, T1, K))
    for i in range(T1):
        yield_paths[:, i, :] = model.yields_from_factors(paths_f[:, i, :], tau[i])
    dd = DatewiseDiscretizer()
    dd.fit(yield_paths)
    idx_paths = dd.transform(yield_paths)
    builder = MarkovChainBuilder(max_states=80, max_successors=25)
    builder.build(idx_paths, T, dd.state_lists, dd.state_to_idx)
    return builder


def test_transition_rows_sum_to_one():
    builder = build_chain_for_test()
    assert builder.validate()


def test_max_successors():
    builder = build_chain_for_test()
    for t in range(builder.T):
        for origin_idx, succ_dict in builder.transition_matrices[t].items():
            assert len(succ_dict) <= 25, (
                f"Date {t}, state {origin_idx} has {len(succ_dict)} successors"
            )


def test_no_empty_rows():
    builder = build_chain_for_test()
    for t in range(builder.T):
        for origin_idx in range(len(builder.state_lists[t])):
            if origin_idx in builder.transition_matrices[t]:
                assert len(builder.transition_matrices[t][origin_idx]) >= 1


def test_diagnostics_structure():
    builder = build_chain_for_test()
    diag = builder.compute_diagnostics()
    assert isinstance(diag, dict)
    for t in range(builder.T):
        assert t in diag
        d = diag[t]
        assert 'n_states_after_cap' in d
        assert 'retained_mass_stats' in d
        mean_mass = d['retained_mass_stats']['mean']
        assert mean_mass >= 0.0
        assert mean_mass <= 1.0 + 1e-10
