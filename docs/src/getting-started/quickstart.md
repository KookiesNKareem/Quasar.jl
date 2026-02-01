# Quick Start

This guide covers the core concepts of SuperNova in 5 minutes.

## Market State

SuperNova separates instruments from market data. A `MarketState` holds current prices, rates, and volatilities:

```julia
using SuperNova

state = MarketState(
    prices = Dict("AAPL" => 150.0, "GOOGL" => 140.0),
    rates = Dict("USD" => 0.05),
    volatilities = Dict("AAPL" => 0.2, "GOOGL" => 0.25)
)
```

## Pricing Options

```julia
# Create an option (just a contract, no market data)
option = EuropeanOption("AAPL", 155.0, 0.5, :call)

# Price using the market state
p = price(option, state)

# Or use Black-Scholes directly
black_scholes(150.0, 155.0, 0.5, 0.05, 0.2, :call)
```

## Computing Greeks

Greeks are computed using analytical Black-Scholes formulas for European options (fast and exact), with automatic differentiation as a fallback for exotic options:

```julia
greeks = compute_greeks(option, state)

greeks.delta  # ∂V/∂S - sensitivity to spot
greeks.gamma  # ∂²V/∂S² - delta sensitivity
greeks.vega   # ∂V/∂σ - per 1% vol move (scaled by 0.01)
greeks.theta  # -∂V/∂T - time decay (negative convention)
greeks.rho    # ∂V/∂r - per 1% rate move (scaled by 0.01)
```

## Portfolio Management

```julia
# Create a portfolio
options = [
    EuropeanOption("AAPL", 155.0, 0.5, :call),
    EuropeanOption("AAPL", 145.0, 0.5, :put)
]
portfolio = Portfolio(options, [100.0, 50.0])  # positions

# Get portfolio value and Greeks
value(portfolio, state)
portfolio_greeks(portfolio, state)
```

## Risk Measures

```julia
returns = randn(1000) * 0.02  # daily returns

compute(VaR(0.95), returns)       # 95% Value at Risk
compute(CVaR(0.95), returns)      # Conditional VaR
compute(Volatility(), returns)    # Annualized volatility
compute(Sharpe(rf=0.02), returns) # Sharpe ratio
compute(MaxDrawdown(), returns)   # Maximum drawdown
```

## Monte Carlo Pricing

```julia
# Define dynamics
dynamics = GBMDynamics(0.05, 0.2)  # r=5%, σ=20%

# Price European call
result = mc_price(100.0, 1.0, EuropeanCall(100.0), dynamics; npaths=50000)
result.price   # Monte Carlo estimate
result.stderr  # Standard error

# Exotic options
mc_price(100.0, 1.0, AsianCall(100.0), dynamics)           # Asian
mc_price(100.0, 1.0, UpAndOutCall(100.0, 130.0), dynamics) # Barrier

# American options (Longstaff-Schwartz)
lsm_price(100.0, 1.0, AmericanPut(100.0), dynamics; npaths=50000)
```

## AD-Based Greeks

Compute Monte Carlo Greeks using automatic differentiation:

```julia
using Enzyme
using SuperNova

dynamics = GBMDynamics(0.05, 0.2)
payoff = EuropeanCall(100.0)

# Delta via pathwise differentiation
delta = mc_delta(100.0, 1.0, payoff, dynamics; backend=EnzymeBackend())

# Delta and Vega together
greeks = mc_greeks(100.0, 1.0, payoff, dynamics; backend=EnzymeBackend())
greeks.delta
greeks.vega
```

## Next Steps

- [AD Backends](../manual/backends.md) - Learn about different AD engines
- [Monte Carlo](../manual/montecarlo.md) - Advanced simulation techniques
- [Optimization](../manual/optimization.md) - Portfolio optimization
