# AGENTS.md — Repository Knowledge Base

## DNS Model Parameters (Pericoli-Taboga 2016)

| Parameter | Value |
|-----------|-------|
| `lambda` | 0.0609 |
| `mu` | [0.0530, -0.0180, 0.0170] |
| `A` (diagonal) | [0.9978, 0.9811, 0.9660] |

### Volatility Scenarios

| Scenario  | σ_level | σ_slope | σ_curvature |
|-----------|---------|---------|-------------|
| Low       | 0.0020  | 0.0040  | 0.0085      |
| Baseline  | 0.0036  | 0.0062  | 0.0092      |
| High      | 0.0050  | 0.0080  | 0.0099      |

## Project Structure

```
project/
├── data/           # Simulated data (generated on-the-fly; CSV/JSON git-ignored)
├── exp/            # Experiment entry-point scripts (exp_1_chain.py … exp_5_sensitivity.py)
├── results/        # Output figures and RESULTS.md summary
├── src/
│   ├── dns_model.py          # DNS factor model: simulation and yield formula
│   ├── state_discretizer.py  # Grid discretisation: StateDiscretizer, DatewiseDiscretizer
│   ├── markov_chain.py       # MarkovChainBuilder: transition matrices and diagnostics
│   ├── mdp_solver.py         # MDPSolver: backward induction + ACTIONS definition
│   └── evaluator.py          # StrategyEvaluator: out-of-sample metrics
└── tests/          # pytest test suite
```

## Key Implementation Notes

### Units and Frequency
- **Yield units**: decimal (e.g. 4 % → 0.04). All internal computations use decimal yields.
- **Time frequency**: monthly (δ = 1/12 year per step).
- **Maturities**: roll forward by δ each period (rolling-bond assumption).

### State Grid
- **Grid spacing**: 40 bp = 0.004 in decimal units.
- **Max states per date**: 80 (states trimmed to the 80 most-visited grid points).
- **Max successors**: top 25 successor states retained per origin state (sparse transition matrices).

### DNS Yield Formula
```
y(τ) = l · B1(λ,τ) + s · B2(λ,τ) + c · B3(λ,τ)

B1(λ,τ) = 1
B2(λ,τ) = (1 - exp(-λτ)) / (λτ)
B3(λ,τ) = B2(λ,τ) - exp(-λτ)
```
- At τ → ∞: B2, B3 → 0, so y → l  (level factor)
- At τ → 0: B2 → 1, B3 → 0, so y → l + s  (level + slope)

### Factor Dynamics (VAR(1))
```
f_{t+1} = mu + A @ (f_t - mu) + Sigma @ epsilon_t,   epsilon_t ~ N(0, I)
```
`A` is diagonal; `Sigma` is diagonal with the scenario volatilities above.

### Action Set (6 Portfolios)
Weights are over three bond maturities (2 yr, 5 yr, 10 yr). All portfolios are
long-only and fully invested (weights sum to 1).

| Index | Description        | w_2yr | w_5yr | w_10yr |
|-------|--------------------|-------|-------|--------|
| 0     | Equal-weight        | 1/3   | 1/3   | 1/3    |
| 1     | Pure short (2 yr)   | 1.0   | 0.0   | 0.0    |
| 2     | Pure medium (5 yr)  | 0.0   | 1.0   | 0.0    |
| 3     | Pure long (10 yr)   | 0.0   | 0.0   | 1.0    |
| 4     | Short-medium mix    | 0.5   | 0.5   | 0.0    |
| 5     | Medium-long mix     | 0.0   | 0.5   | 0.5    |

### Transaction Costs
Cost function: `φ(w_new, w_pre) = c · Σ|w_new_i - w_pre_i|`
Default `c = 0.001` (10 bp per unit of turnover).

## Implementation Notes (src/)

### dns_model.py
- `simulate_paths`: vectorised over N_paths using `np.random.default_rng.multivariate_normal(size=(N,T))`;
  VAR update uses `paths[:, t, :] @ A.T` (correct for batched matrix-vector multiply; A is symmetric/diagonal so `.T` is a no-op here).
- `yields_from_factors`: supports arbitrary leading batch dimensions via `factors[..., 0:1]` slicing and
  numpy broadcasting `(..., 1) * (K,) → (..., K)`. Handles τ=0 limit (B1→1, B2→0) with `np.where`.
- `initial_conditions` dict: `'long_run_mean'` is populated at `__init__` time with a copy of `mu`.

### state_discretizer.py
- `DatewiseDiscretizer.fit`: states ranked by `(-frequency, coord_tuple)` so ties among equal-frequency
  states break lexicographically (smaller coord wins). Retained states are stored in lex order so that
  `np.argmin` on distance rows naturally returns the lex-smallest tied retained state.
- `remap_tables[t]` only contains pruned→retained mappings; retained coords are absent (identity is implied).
- `transform` applies `remap_tables[t].get(coord, coord)` then looks up `state_to_idx[t][coord]`.

### markov_chain.py
- Transitions counted via `Counter(zip(origins, successors))` for efficiency (avoids Python inner loop).
- Empty-row repair: if an origin state appears in `state_lists[t]` but has no observed transitions (rare),
  a uniform distribution over all t+1 states is assigned.
- `retained_mass_per_row(t)`: fraction of total transition counts captured by top-25 successors before
  renormalisation; equals 1.0 for origins with ≤25 unique successors.
- `build` accepts optional `n_states_before_cap: list[int]` (length T+1) populated by the caller using
  pre-pruning unique-coord counts from `DatewiseDiscretizer`.

## Testing
Run the full test suite from the project root:
```bash
pytest tests/ -v
```
