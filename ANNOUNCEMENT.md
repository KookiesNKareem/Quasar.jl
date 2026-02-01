# SuperNova.jl - Differentiable Quantitative Finance

I'm sharing **SuperNova.jl**, a quant finance library built around two ideas: a unified API and automatic differentiation everywhere.

**GitHub:** https://github.com/KookiesNKareem/SuperNova.jl
**Docs:** https://KookiesNKareem.github.io/SuperNova.jl/dev/

## Performance

SuperNova.jl vs [QuantLib](https://www.quantlib.org/) C++ (v1.41):

| Benchmark | SuperNova.jl | QuantLib C++ | Speedup |
|-----------|---------|--------------|---------|
| European option | 0.04 μs | 5.7 μs | **139x** |
| Greeks (all 5) | 0.08 μs | 5.7 μs | **69x** |
| American option (binomial) | 8.4 μs | 67 μs | **8x** |

QuantLib builds a reusable object graph per instrument; SuperNova compiles specialized native code via Julia's JIT. The 8x speedup on American options reflects pure algorithm performance. Benchmarks on Apple M1 — full methodology in `benchmarks/comparison/`.

## Why SuperNova?

**One API, everything differentiable.** Price an option, compute Greeks, calibrate a model, optimize a portfolio—it's all the same interface, and gradients flow through automatically.

```julia
using SuperNova

# Price
black_scholes(S, K, T, r, σ, :call)

# Greeks via AD
compute_greeks(option, state)

# Calibrate SABR
calibrate_sabr(smile_data)

# Optimize portfolio
optimize(MeanVariance(μ, Σ); target_return=0.10)
```

No separate "analytical Greeks" vs "numerical Greeks". No special calibration code. The AD system handles it.

**Backend-agnostic.** Same code runs on CPU (ForwardDiff) or GPU (Reactant/Enzyme):

```julia
gradient(f, x)                              # CPU
gradient(f, x; backend=ReactantBackend())   # GPU
```

## What's included

- **Pricing:** European, American, Asian, Barrier options; Monte Carlo with variance reduction
- **Models:** Black-Scholes, SABR, Heston stochastic volatility
- **Greeks:** Delta, Gamma, Vega, Theta, Rho, Vanna, Volga, Charm
- **Optimization:** Mean-variance, risk parity, Black-Litterman, efficient frontier
- **Risk:** VaR, CVaR, Sharpe, Sortino, max drawdown
- **Rates:** Yield curves, bonds, caps/floors, swaptions, short-rate models
- **Backtesting:** Strategy simulation with execution models

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/KookiesNKareem/SuperNova.jl")
```

Feedback and contributions welcome!
