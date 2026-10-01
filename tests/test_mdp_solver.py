import numpy as np
import pytest
import sys
sys.path.insert(0, '/workspace/project')
from src.mdp_solver import MDPSolver, ACTIONS


def test_actions_shape():
    assert ACTIONS.shape == (6, 3)
    # All long-only
    assert np.all(ACTIONS >= 0)
    # All fully invested
    np.testing.assert_allclose(ACTIONS.sum(axis=1), 1.0)


def test_bond_price():
    from src.mdp_solver import bond_price
    # Zero yield -> price = 1
    assert bond_price(1.0, 0.0) == pytest.approx(1.0)
    # Positive yield -> price < 1
    assert bond_price(5.0, 0.05) < 1.0
    assert bond_price(5.0, 0.05) == pytest.approx(np.exp(-0.25))


def test_transaction_cost_zero_for_same_weights():
    """No cost when action unchanged."""
    solver = MDPSolver.__new__(MDPSolver)  # bypass __init__ for unit test
    solver.cost_c = 0.001
    w = np.array([1/3, 1/3, 1/3])
    phi = solver._transaction_cost(w, w)
    assert phi == pytest.approx(0.0)


def test_transaction_cost_formula():
    solver = MDPSolver.__new__(MDPSolver)
    solver.cost_c = 0.01
    w_new = np.array([1.0, 0.0, 0.0])
    w_pre = np.array([0.0, 1.0, 0.0])
    phi = solver._transaction_cost(w_new, w_pre)
    # |1-0| + |0-1| + |0-0| = 2, * 0.01 = 0.02
    assert phi == pytest.approx(0.02)


def test_pre_rebalance_weights_sum_to_one():
    """Pre-rebalance weights should sum to 1."""
    solver = MDPSolver.__new__(MDPSolver)
    solver.cost_c = 0.001
    solver.delta = 1/12
    solver.tau_initial = np.array([2.0, 5.0, 10.0])
    solver.grid_size = 0.004
    # Mock prices
    solver.prices = [
        np.array([[np.exp(-2*0.04), np.exp(-5*0.04), np.exp(-10*0.04)]]),   # t=0
        np.array([[np.exp(-1.917*0.041), np.exp(-4.917*0.041), np.exp(-9.917*0.041)]])  # t=1
    ]
    w = np.array([1/3, 1/3, 1/3])
    w_pre = solver._pre_rebalance_weights(w, 0, 0, t=1)
    assert np.sum(w_pre) == pytest.approx(1.0)
    assert np.all(w_pre >= 0)
