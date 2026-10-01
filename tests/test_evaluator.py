import numpy as np
import pytest
import sys
sys.path.insert(0, '/workspace/project')
from src.evaluator import StrategyEvaluator


def test_compute_metrics_shape():
    # Create fake wealth paths
    N, T = 1000, 7  # T+1=7 for T=6
    wealth_paths = np.ones((N, T))
    wealth_paths[:, -1] = 1.1  # terminal wealth = 1.1

    # Mock evaluator with minimal attributes
    evaluator = StrategyEvaluator.__new__(StrategyEvaluator)
    metrics = evaluator.compute_metrics(wealth_paths)

    assert metrics['mean_terminal'] == pytest.approx(1.1)
    assert metrics['median_terminal'] == pytest.approx(1.1)
    assert 'q5_terminal' in metrics
    assert 'mean_mdd' in metrics


def test_mdd_is_nonnegative():
    N, T = 100, 7
    # Declining wealth path -> high MDD
    wealth_paths = np.ones((N, T))
    for i in range(1, T):
        wealth_paths[:, i] = wealth_paths[:, i-1] * 0.98

    evaluator = StrategyEvaluator.__new__(StrategyEvaluator)
    metrics = evaluator.compute_metrics(wealth_paths)
    assert metrics['mean_mdd'] > 0


def test_mdd_zero_for_monotone_increasing():
    N, T = 100, 7
    # Monotonically increasing wealth -> no drawdown
    wealth_paths = np.cumprod(np.ones((N, T)) * 1.01, axis=1)
    wealth_paths = np.hstack([np.ones((N, 1)), wealth_paths[:, :-1]])

    evaluator = StrategyEvaluator.__new__(StrategyEvaluator)
    metrics = evaluator.compute_metrics(wealth_paths)
    assert metrics['mean_mdd'] == pytest.approx(0.0, abs=1e-10)


def test_paired_stats():
    evaluator = StrategyEvaluator.__new__(StrategyEvaluator)
    tw_dp = np.array([1.1, 1.2, 1.0, 1.15, 1.05])
    tw_bench = np.array([1.0, 1.1, 0.95, 1.05, 0.98])
    stats = evaluator.compute_paired_stats(tw_dp, tw_bench)
    assert stats['mean_diff'] == pytest.approx(np.mean(tw_dp - tw_bench))
    assert stats['t_stat'] > 0  # DP outperforms
    assert stats['n'] == 5
