```@meta
CurrentModule = Quasar
```

# Quasar.jl

A high-performance quantitative finance library for Julia with automatic differentiation support.

## Features

- **Multi-backend AD**: ForwardDiff, Enzyme, Reactant backends with unified API
- **Options Pricing**: Black-Scholes, Heston, SABR models
- **Monte Carlo**: European, Asian, barrier, American options with Longstaff-Schwartz
- **Greeks**: Analytical and AD-based sensitivities
- **Portfolio Optimization**: Mean-variance, Sharpe maximization, CVaR
- **Risk Measures**: VaR, CVaR, volatility, drawdown

## Quick Start

```julia
using Quasar

# Price a European call option
S0, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.2
price = black_scholes(S0, K, T, r, σ, :call)

# Compute Greeks
greeks = compute_greeks(EuropeanOption(K, T, :call, :european), S0, r, σ)

# Monte Carlo pricing
dynamics = GBMDynamics(r, σ)
result = mc_price(S0, T, EuropeanCall(K), dynamics; npaths=100000)

# Portfolio optimization
weights = optimize(MeanVariance(0.5), returns, Σ)
```

## Manual

```@contents
Pages = ["backends.md"]
Depth = 2
```

## API Reference

```@autodocs
Modules = [Quasar]
```
