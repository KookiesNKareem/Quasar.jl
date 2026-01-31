---
layout: home

hero:
  name: Quasar.jl
  text: Differentiable Quantitative Finance
  tagline: High-performance derivatives pricing with automatic differentiation
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started/installation
    - theme: alt
      text: View on GitHub
      link: https://github.com/KookiesNKareem/Quasar.jl

features:
  - icon: 🔄
    title: Differentiable by Default
    details: Every computation flows through a unified AD system. Gradients are first-class outputs, not afterthoughts.
  - icon: 🚀
    title: Multi-Backend AD
    details: Same code runs on CPU (ForwardDiff) or GPU (Enzyme/Reactant). Write once, deploy anywhere.
  - icon: 📊
    title: Production Ready
    details: Pure Julia reference implementations for debugging, optimized backends for production workloads.
  - icon: 🧩
    title: Composable Design
    details: Small, focused types that combine naturally. Build complex strategies from simple primitives.
---

# What is Quasar.jl?

Quasar is a quantitative finance library for Julia that puts automatic differentiation at the center of everything. Whether you're pricing exotic derivatives, computing Greeks, calibrating models, or optimizing portfolios, Quasar provides a unified, differentiable API.

## Quick Example

```julia
using Quasar

# Price a European call option
S0, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.2
price = black_scholes(S0, K, T, r, σ, :call)

# Compute Greeks via AD
option = EuropeanOption("AAPL", K, T, :call)
state = MarketState(prices=Dict("AAPL" => S0), rates=Dict("USD" => r), volatilities=Dict("AAPL" => σ))
greeks = compute_greeks(option, state)

# Monte Carlo with pathwise Greeks
dynamics = GBMDynamics(r, σ)
delta = mc_delta(S0, T, EuropeanCall(K), dynamics; backend=EnzymeBackend())
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Options Pricing** | Black-Scholes, Heston, SABR models |
| **Monte Carlo** | European, Asian, barrier, American (LSM) |
| **Greeks** | Analytical and AD-based sensitivities |
| **Calibration** | SABR and Heston model calibration |
| **Optimization** | Mean-variance, Sharpe, CVaR objectives |
| **Risk Measures** | VaR, CVaR, volatility, max drawdown |

## AD Backends

Choose the right backend for your workload:

| Backend | Best For |
|---------|----------|
| `ForwardDiffBackend()` | Default, reliable, low-dimensional |
| `EnzymeBackend()` | Large-scale, reverse-mode, GPU |
| `ReactantBackend()` | XLA compilation, GPU acceleration |
| `PureJuliaBackend()` | Debugging, testing |

```julia
# Switch backends easily
gradient(f, x; backend=EnzymeBackend())

# Or use scoped switching
with_backend(ReactantBackend()) do
    optimize(objective, x0)
end
```
