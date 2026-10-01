# DNS Markov Chain Approximation for Bond Portfolio Optimization

## Description

This project implements a **time-inhomogeneous Markov chain approximation** of the
Dynamic Nelson-Siegel (DNS) factor model for bond yields, combined with a
**dynamic programming** (DP) solution for bond portfolio optimization.

Five experiments are included:

1. **exp_1_chain.py** – Discretize the DNS factor paths into a finite-state Markov chain and inspect chain diagnostics (state counts, transition mass retained, successor distributions).
2. **exp_2_yields.py** – Validate simulated yield curves against the DNS closed-form formula across maturities and volatility scenarios.
3. **exp_3_dp.py** – Solve the DP problem for the optimal rebalancing policy over the investment horizon and report value-function convergence.
4. **exp_4_evaluate.py** – Evaluate the DP policy against benchmark strategies (equal-weight, pure bond) on out-of-sample simulated paths; report terminal wealth and drawdown metrics.
5. **exp_5_sensitivity.py** – Sensitivity analysis over volatility scenarios (low / baseline / high) and transaction-cost levels.

## Installation

```bash
pip install numpy scipy pandas matplotlib seaborn cvxpy
```

## Usage

Run each experiment from the project root:

```bash
python exp/exp_1_chain.py
python exp/exp_2_yields.py
python exp/exp_3_dp.py
python exp/exp_4_evaluate.py
python exp/exp_5_sensitivity.py
```

## Results

See [results/RESULTS.md](results/RESULTS.md) for tables and figures produced by the experiments.
