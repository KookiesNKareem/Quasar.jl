# Monte Carlo Simulation

SuperNova provides a comprehensive Monte Carlo engine for pricing path-dependent derivatives.

## Dynamics

### Geometric Brownian Motion

```julia
dynamics = GBMDynamics(r, σ)
```

- `r` - Risk-free rate
- `σ` - Volatility

The asset follows: `dS = rS dt + σS dW`

### Heston Stochastic Volatility

```julia
dynamics = HestonDynamics(r, v0, κ, θ, ξ, ρ)
```

- `r` - Risk-free rate
- `v0` - Initial variance
- `κ` - Mean reversion speed
- `θ` - Long-term variance
- `ξ` - Vol of vol
- `ρ` - Correlation between spot and vol

## Payoffs

| Payoff | Constructor | Description |
|--------|-------------|-------------|
| European Call | `EuropeanCall(K)` | max(S_T - K, 0) |
| European Put | `EuropeanPut(K)` | max(K - S_T, 0) |
| Asian Call | `AsianCall(K)` | max(avg(S) - K, 0) |
| Asian Put | `AsianPut(K)` | max(K - avg(S), 0) |
| Up-and-Out Call | `UpAndOutCall(K, B)` | Call that knocks out if S ≥ B |
| Down-and-Out Put | `DownAndOutPut(K, B)` | Put that knocks out if S ≤ B |
| American Put | `AmericanPut(K)` | Early exercise put |
| American Call | `AmericanCall(K)` | Early exercise call |

## Basic Pricing

```julia
S0 = 100.0  # Initial spot
T = 1.0     # Time to maturity
dynamics = GBMDynamics(0.05, 0.2)

result = mc_price(S0, T, EuropeanCall(100.0), dynamics;
    npaths=100000,    # Number of paths
    nsteps=252,       # Steps per path
    antithetic=true   # Variance reduction
)

result.price     # Monte Carlo estimate
result.stderr    # Standard error
result.ci_lower  # 95% CI lower bound
result.ci_upper  # 95% CI upper bound
```

## Variance Reduction

### Antithetic Variates

Enabled by default. Uses pairs of negatively correlated paths:

```julia
mc_price(S0, T, payoff, dynamics; antithetic=true)
```

### Quasi-Monte Carlo

Uses Sobol sequences instead of pseudo-random numbers. Better convergence: O(1/N) vs O(1/√N).

```julia
# Direct QMC pricing
result = mc_price_qmc(S0, T, payoff, dynamics; npaths=10000)

# Generate Sobol-based normals directly
Z = sobol_normals(nsteps, npaths)  # (npaths × nsteps) matrix
```

## American Options (LSM)

The Longstaff-Schwartz algorithm prices American options via regression:

```julia
result = lsm_price(S0, T, AmericanPut(100.0), dynamics;
    npaths=50000,
    nsteps=50  # Exercise dates
)

# Early exercise premium
eu_put = mc_price(S0, T, EuropeanPut(100.0), dynamics)
premium = result.price - eu_put.price
```

## Monte Carlo Greeks

Compute sensitivities via pathwise differentiation:

```julia
using Enzyme
using SuperNova

# Delta only
delta = mc_delta(S0, T, payoff, dynamics;
    npaths=10000,
    nsteps=50,
    backend=EnzymeBackend()
)

# Delta and Vega
greeks = mc_greeks(S0, T, payoff, dynamics;
    npaths=10000,
    nsteps=50,
    backend=EnzymeBackend()
)
greeks.delta
greeks.vega
```

### Backend Behavior

| Backend | Method |
|---------|--------|
| `ForwardDiffBackend()` | Pseudo-random with fixed seed |
| `PureJuliaBackend()` | Finite differences |
| `EnzymeBackend()` | QMC (Sobol sequences) |

Enzyme automatically uses QMC because it cannot differentiate through RNGs.

## Heston Monte Carlo

```julia
heston = HestonDynamics(
    0.05,   # r
    0.04,   # v0
    2.0,    # κ
    0.04,   # θ
    0.3,    # ξ
    -0.7    # ρ
)

result = mc_price(100.0, 1.0, EuropeanCall(100.0), heston;
    npaths=50000,
    nsteps=252
)

# Compare to semi-analytical Heston
params = HestonParams(0.04, 0.04, 2.0, 0.3, -0.7)
analytic = heston_price(100.0, 100.0, 1.0, 0.05, params, :call)
```

## Path Simulation

For custom payoffs or analysis, simulate paths directly:

```julia
# Single GBM path
path = SuperNova.MonteCarlo.simulate_gbm(S0, T, nsteps, dynamics)

# Antithetic pair
path1, path2 = SuperNova.MonteCarlo.simulate_gbm_antithetic(S0, T, nsteps, dynamics)

# Heston path
path = SuperNova.MonteCarlo.simulate_heston(S0, T, nsteps, heston_dynamics)

# QMC path (deterministic)
Z = sobol_normals(nsteps, 1)
path = SuperNova.MonteCarlo.simulate_gbm_qmc(S0, T, nsteps, dynamics, Z[1, :])
```
