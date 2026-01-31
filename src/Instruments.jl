module Instruments

using ..Core: AbstractEquity, AbstractOption, MarketState, IsPriceable, IsDifferentiable, HasGreeksTrait, priceable, differentiable, greeks_trait
using Distributions: Normal, cdf

# Import to extend
import ..Core: priceable, differentiable, greeks_trait

# Stock
"""
    Stock <: AbstractEquity

A simple equity instrument.

# Fields
- `symbol::String` - Ticker symbol
"""
struct Stock <: AbstractEquity
    symbol::String

    function Stock(symbol::String)
        isempty(symbol) && throw(ArgumentError("Stock symbol cannot be empty"))
        new(symbol)
    end
end

# Register traits
priceable(::Type{Stock}) = IsPriceable()
differentiable(::Type{Stock}) = IsDifferentiable()

# ============================================================================
# Pricing Interface
# ============================================================================

"""
    price(instrument, market_state)

Compute the current price of an instrument given market state.
"""
function price end

# Stock pricing - just lookup
function price(stock::Stock, state::MarketState)
    return state.prices[stock.symbol]
end

# ============================================================================
# European Option
# ============================================================================

"""
    EuropeanOption <: AbstractOption

A European-style option (exercise only at expiry).

# Fields
- `underlying::String` - Symbol of underlying asset
- `strike::Float64` - Strike price
- `expiry::Float64` - Time to expiration (in years)
- `optiontype::Symbol` - :call or :put
"""
struct EuropeanOption <: AbstractOption
    underlying::String
    strike::Float64
    expiry::Float64
    optiontype::Symbol

    function EuropeanOption(underlying, strike, expiry, optiontype)
        optiontype in (:call, :put) || throw(ArgumentError("optiontype must be :call or :put, got $optiontype"))
        strike > 0 || throw(ArgumentError("strike must be positive, got $strike"))
        expiry > 0 || throw(ArgumentError("expiry must be positive, got $expiry"))
        isfinite(strike) || throw(ArgumentError("strike must be finite, got $strike"))
        isfinite(expiry) || throw(ArgumentError("expiry must be finite, got $expiry"))
        new(underlying, strike, expiry, optiontype)
    end
end

# Register traits
priceable(::Type{EuropeanOption}) = IsPriceable()
differentiable(::Type{EuropeanOption}) = IsDifferentiable()
greeks_trait(::Type{EuropeanOption}) = HasGreeksTrait()

# Black-Scholes pricing
# TODO: Support multiple rates (currency-specific)
# TODO: Add dividend yield support
function price(opt::EuropeanOption, state::MarketState)
    S = state.prices[opt.underlying]
    K = opt.strike
    T = opt.expiry
    r = first(values(state.rates))  # FIXME: Assumes single rate - add currency support
    σ = state.volatilities[opt.underlying]

    return black_scholes(S, K, T, r, σ, opt.optiontype)
end

"""
    black_scholes(S, K, T, r, σ, optiontype)

Black-Scholes option pricing formula.

# Arguments
- `S` - Current price of underlying
- `K` - Strike price
- `T` - Time to expiration (years)
- `r` - Risk-free rate
- `σ` - Volatility
- `optiontype` - :call or :put
"""
function black_scholes(S, K, T, r, σ, optiontype::Symbol)
    # Handle edge cases
    if T <= 0
        # At or past expiry: return intrinsic value
        if optiontype == :call
            return max(S - K, zero(S))
        else
            return max(K - S, zero(S))
        end
    end

    if σ <= 0
        # Zero volatility: deterministic outcome
        forward = S * exp(r * T)
        df = exp(-r * T)
        if optiontype == :call
            return df * max(forward - K, zero(S))
        else
            return df * max(K - forward, zero(S))
        end
    end

    # Standard Black-Scholes formula
    d1 = (log(S/K) + (r + σ^2/2)*T) / (σ*sqrt(T))
    d2 = d1 - σ*sqrt(T)

    N = Normal()

    if optiontype == :call
        return S * cdf(N, d1) - K * exp(-r*T) * cdf(N, d2)
    else  # put
        return K * exp(-r*T) * cdf(N, -d2) - S * cdf(N, -d1)
    end
end

# ============================================================================
# American Option
# ============================================================================

"""
    AmericanOption <: AbstractOption

An American-style option (can exercise any time before expiry).

Pricing requires numerical methods (binomial tree or LSM Monte Carlo).
Use `lsm_price()` from MonteCarlo module for Monte Carlo pricing.

# Fields
- `underlying::String` - Symbol of underlying asset
- `strike::Float64` - Strike price
- `expiry::Float64` - Time to expiration (in years)
- `optiontype::Symbol` - :call or :put
"""
struct AmericanOption <: AbstractOption
    underlying::String
    strike::Float64
    expiry::Float64
    optiontype::Symbol

    function AmericanOption(underlying, strike, expiry, optiontype)
        optiontype in (:call, :put) || error("optiontype must be :call or :put")
        strike > 0 || error("strike must be positive")
        expiry > 0 || error("expiry must be positive")
        new(underlying, strike, expiry, optiontype)
    end
end

# Register traits
priceable(::Type{AmericanOption}) = IsPriceable()
differentiable(::Type{AmericanOption}) = IsDifferentiable()
greeks_trait(::Type{AmericanOption}) = HasGreeksTrait()

"""
    price(opt::AmericanOption, state::MarketState; method=:crr, nsteps=100)

Price an American option using binomial tree (CRR method).

For Monte Carlo pricing, use `lsm_price()` from the MonteCarlo module.
"""
function price(opt::AmericanOption, state::MarketState; method::Symbol=:crr, nsteps::Int=100)
    S = state.prices[opt.underlying]
    K = opt.strike
    T = opt.expiry
    r = first(values(state.rates))
    σ = state.volatilities[opt.underlying]

    return american_binomial(S, K, T, r, σ, opt.optiontype, nsteps)
end

"""
    american_binomial(S, K, T, r, σ, optiontype, nsteps)

Cox-Ross-Rubinstein binomial tree for American options.
"""
function american_binomial(S, K, T, r, σ, optiontype::Symbol, nsteps::Int)
    dt = T / nsteps
    u = exp(σ * sqrt(dt))
    d = 1 / u
    p = (exp(r * dt) - d) / (u - d)
    df = exp(-r * dt)

    # Terminal payoffs
    prices = [S * u^j * d^(nsteps - j) for j in 0:nsteps]
    if optiontype == :call
        values = [max(p - K, 0.0) for p in prices]
    else
        values = [max(K - p, 0.0) for p in prices]
    end

    # Backward induction with early exercise check
    for i in (nsteps-1):-1:0
        for j in 0:i
            stock_price = S * u^j * d^(i - j)
            hold_value = df * (p * values[j+2] + (1 - p) * values[j+1])
            if optiontype == :call
                exercise_value = max(stock_price - K, 0.0)
            else
                exercise_value = max(K - stock_price, 0.0)
            end
            values[j+1] = max(hold_value, exercise_value)
        end
    end

    return values[1]
end

# ============================================================================
# Asian Option
# ============================================================================

"""
    AsianOption <: AbstractOption

An Asian option with payoff based on average price over the life of the option.

Pricing requires Monte Carlo simulation. Use `mc_price()` with `AsianCall`/`AsianPut`
payoffs from the MonteCarlo module for accurate pricing.

# Fields
- `underlying::String` - Symbol of underlying asset
- `strike::Float64` - Strike price
- `expiry::Float64` - Time to expiration (in years)
- `optiontype::Symbol` - :call or :put
- `averaging::Symbol` - :arithmetic or :geometric
"""
struct AsianOption <: AbstractOption
    underlying::String
    strike::Float64
    expiry::Float64
    optiontype::Symbol
    averaging::Symbol

    function AsianOption(underlying, strike, expiry, optiontype; averaging::Symbol=:arithmetic)
        optiontype in (:call, :put) || error("optiontype must be :call or :put")
        averaging in (:arithmetic, :geometric) || error("averaging must be :arithmetic or :geometric")
        strike > 0 || error("strike must be positive")
        expiry > 0 || error("expiry must be positive")
        new(underlying, strike, expiry, optiontype, averaging)
    end
end

# Register traits
priceable(::Type{AsianOption}) = IsPriceable()
differentiable(::Type{AsianOption}) = IsDifferentiable()

"""
    price(opt::AsianOption, state::MarketState; nsteps=252)

Price an Asian option using closed-form geometric approximation or Turnbull-Wakeman
for arithmetic averaging.

For more accurate pricing, use `mc_price()` with `AsianCall`/`AsianPut` payoffs.
"""
function price(opt::AsianOption, state::MarketState; nsteps::Int=252)
    S = state.prices[opt.underlying]
    K = opt.strike
    T = opt.expiry
    r = first(values(state.rates))
    σ = state.volatilities[opt.underlying]

    if opt.averaging == :geometric
        return asian_geometric(S, K, T, r, σ, opt.optiontype)
    else
        # Turnbull-Wakeman approximation for arithmetic Asian
        return asian_arithmetic_approx(S, K, T, r, σ, opt.optiontype, nsteps)
    end
end

"""
    asian_geometric(S, K, T, r, σ, optiontype)

Closed-form solution for geometric Asian options.
"""
function asian_geometric(S, K, T, r, σ, optiontype::Symbol)
    # Adjusted parameters for geometric average
    σ_adj = σ / sqrt(3)
    r_adj = 0.5 * (r - σ^2 / 6)

    d1 = (log(S / K) + (r_adj + σ_adj^2 / 2) * T) / (σ_adj * sqrt(T))
    d2 = d1 - σ_adj * sqrt(T)

    N = Normal()

    if optiontype == :call
        return exp(-r * T) * (S * exp(r_adj * T) * cdf(N, d1) - K * cdf(N, d2))
    else
        return exp(-r * T) * (K * cdf(N, -d2) - S * exp(r_adj * T) * cdf(N, -d1))
    end
end

"""
    asian_arithmetic_approx(S, K, T, r, σ, optiontype, nsteps)

Turnbull-Wakeman approximation for arithmetic Asian options.
Uses moment matching to a lognormal distribution.
"""
function asian_arithmetic_approx(S, K, T, r, σ, optiontype::Symbol, nsteps::Int)
    σ2 = σ^2

    # M1: Expected value of arithmetic average under risk-neutral measure
    # E[A] = S * (exp(rT) - 1) / (rT) for continuous averaging
    if abs(r) < 1e-10
        M1 = S  # Limit as r -> 0
    else
        M1 = S * (exp(r * T) - 1) / (r * T)
    end

    # M2: Second moment E[A^2]
    # Using Turnbull-Wakeman formula for continuous arithmetic average
    if abs(r) < 1e-10
        # Limit case when r ≈ 0
        M2 = S^2 * exp(σ2 * T) * (2 / (σ2 * T^2)) * (exp(σ2 * T) - 1 - σ2 * T) / σ2^2
    else
        a = 2 * r + σ2
        b = r + σ2
        M2 = (2 * S^2 / (T^2)) * (
            (exp(a * T) - 1) / (a * b) -
            (exp(r * T) - 1) / (r * b)
        )
    end

    # Ensure M2 > M1^2 for valid lognormal matching
    M2 = max(M2, M1^2 * 1.0001)

    # Match to lognormal: log(A) ~ N(μ_adj, σ_adj^2 * T)
    # E[A] = exp(μ_adj + σ_adj^2*T/2) = M1
    # E[A^2] = exp(2μ_adj + 2σ_adj^2*T) = M2
    # => σ_adj^2 * T = log(M2/M1^2)
    var_log = log(M2 / M1^2)
    σ_adj = sqrt(max(var_log, 1e-10) / T)

    # Forward price of the average
    F_adj = M1

    # Black-76 style pricing with adjusted vol
    d1 = (log(F_adj / K) + σ_adj^2 * T / 2) / (σ_adj * sqrt(T))
    d2 = d1 - σ_adj * sqrt(T)

    N = Normal()
    df = exp(-r * T)

    if optiontype == :call
        return df * (F_adj * cdf(N, d1) - K * cdf(N, d2))
    else
        return df * (K * cdf(N, -d2) - F_adj * cdf(N, -d1))
    end
end

# ============================================================================
# Exports
# ============================================================================

export Stock, EuropeanOption, AmericanOption, AsianOption, price, black_scholes
export american_binomial, asian_geometric, asian_arithmetic_approx

# Greeks computation via AD
include("Greeks.jl")

end
