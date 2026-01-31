# Greeks computation - analytical formulas primary, AD for fallback/exotics

using ForwardDiff
using Distributions: Normal, cdf, pdf

# Import AD module for configurable backend
using ..AD: current_backend, ForwardDiffBackend, PureJuliaBackend

"""
    GreeksResult

Container for option Greeks including first-order, second-order, and cross-derivatives.

# First-order Greeks
- `delta` - dV/dS (sensitivity to underlying price)
- `vega` - dV/dσ (sensitivity to volatility, scaled per 1% move)
- `theta` - -dV/dT (time decay per year)
- `rho` - dV/dr (sensitivity to rates, scaled per 1% move)

# Second-order Greeks
- `gamma` - d²V/dS² (rate of change of delta)
- `vanna` - d²V/dSdσ (sensitivity of delta to volatility)
- `volga` - d²V/dσ² (sensitivity of vega to volatility, a.k.a. vomma)
- `charm` - d²V/dSdT (sensitivity of delta to time, a.k.a. delta decay)
"""
struct GreeksResult{T}
    delta::T    # dV/dS
    gamma::T    # d²V/dS²
    vega::T     # dV/dσ (per 1% move, scaled by 0.01)
    theta::T    # -dV/dT (time decay per year)
    rho::T      # dV/dr (per 1% move, scaled by 0.01)
    vanna::T    # d²V/dSdσ (delta sensitivity to vol)
    volga::T    # d²V/dσ² (vega sensitivity to vol, a.k.a. vomma)
    charm::T    # d²V/dSdT (delta decay, sensitivity of delta to time)
end

# Constructor with optional second-order Greeks (for backward compatibility)
function GreeksResult(delta, gamma, vega, theta, rho)
    T = promote_type(typeof(delta), typeof(gamma), typeof(vega), typeof(theta), typeof(rho))
    GreeksResult(delta, gamma, vega, theta, rho, zero(T), zero(T), zero(T))
end

# Pretty printing
function Base.show(io::IO, g::GreeksResult)
    print(io, "GreeksResult(Δ=", round(g.delta, digits=4),
          ", Γ=", round(g.gamma, digits=6),
          ", V=", round(g.vega, digits=4),
          ", Θ=", round(g.theta, digits=4),
          ", ρ=", round(g.rho, digits=4), ")")
end

function Base.show(io::IO, ::MIME"text/plain", g::GreeksResult)
    println(io, "GreeksResult:")
    println(io, "  First-order:")
    println(io, "    Delta (Δ):  ", round(g.delta, digits=6))
    println(io, "    Vega  (V):  ", round(g.vega, digits=6), " (per 1% vol)")
    println(io, "    Theta (Θ):  ", round(g.theta, digits=6), " (per year)")
    println(io, "    Rho   (ρ):  ", round(g.rho, digits=6), " (per 1% rate)")
    println(io, "  Second-order:")
    println(io, "    Gamma (Γ):  ", round(g.gamma, digits=6))
    println(io, "    Vanna:      ", round(g.vanna, digits=6))
    println(io, "    Volga:      ", round(g.volga, digits=6))
    print(io,   "    Charm:      ", round(g.charm, digits=6))
end

# ============================================================================
# Primary Interface - dispatches to analytical when available
# ============================================================================

"""
    compute_greeks(option, market_state; backend=current_backend())

Compute option Greeks. Uses analytical formulas when available (preferred),
falls back to AD for exotic options without closed-form solutions.

# Arguments
- `option` - The option to compute Greeks for
- `market_state` - Current market conditions
- `backend` - AD backend to use for fallback computation (default: current_backend())
"""
function compute_greeks(opt::EuropeanOption, state::MarketState; backend=current_backend())
    # European options have closed-form Black-Scholes Greeks
    return _analytical_greeks(opt, state)
end

# Fallback for options without analytical Greeks - use AD
function compute_greeks(opt::AbstractOption, state::MarketState; backend=current_backend())
    return _ad_greeks(opt, state, backend)
end

# ============================================================================
# Analytical Greeks - Black-Scholes (exact, fast)
# ============================================================================

"""
    _analytical_greeks(option, market_state)

Compute Greeks using analytical Black-Scholes formulas.
Exact closed-form solutions - no numerical approximation.
Includes all first-order and second-order Greeks.
"""
function _analytical_greeks(opt::EuropeanOption, state::MarketState)
    S = state.prices[opt.underlying]
    K = opt.strike
    T = opt.expiry
    r = first(values(state.rates))
    σ = state.volatilities[opt.underlying]

    sqrtT = sqrt(T)
    d1 = (log(S/K) + (r + σ^2/2)*T) / (σ*sqrtT)
    d2 = d1 - σ*sqrtT

    N = Normal()
    n_d1 = pdf(N, d1)
    N_d1 = cdf(N, d1)
    N_d2 = cdf(N, d2)

    # First-order Greeks
    if opt.optiontype == :call
        delta = N_d1
        theta = -S * n_d1 * σ / (2*sqrtT) - r * K * exp(-r*T) * N_d2
        rho = K * T * exp(-r*T) * N_d2 * 0.01
    else  # put
        delta = N_d1 - 1
        theta = -S * n_d1 * σ / (2*sqrtT) + r * K * exp(-r*T) * cdf(N, -d2)
        rho = -K * T * exp(-r*T) * cdf(N, -d2) * 0.01
    end

    # Gamma and Vega are same for calls and puts
    gamma = n_d1 / (S * σ * sqrtT)
    vega = S * n_d1 * sqrtT * 0.01

    # Second-order Greeks (same for calls and puts)
    # Vanna: d²V/dSdσ = d(delta)/dσ = -n(d1) * d2 / σ
    # Also: vanna = vega/S * (1 - d1/(σ*sqrt(T)))
    vanna = -n_d1 * d2 / σ

    # Volga (Vomma): d²V/dσ² = vega * d1 * d2 / σ
    # Note: vega here is unscaled (S * n_d1 * sqrtT)
    vega_unscaled = S * n_d1 * sqrtT
    volga = vega_unscaled * d1 * d2 / σ * 0.01  # Scale for 1% vol move

    # Charm: d²V/dSdT = d(delta)/dT
    # For call: charm = -n(d1) * (2*(r-q)*T - d2*σ*sqrt(T)) / (2*T*σ*sqrt(T))
    # Simplified (q=0): charm = -n(d1) * (2*r*T - d2*σ*sqrt(T)) / (2*T*σ*sqrt(T))
    if opt.optiontype == :call
        charm = -n_d1 * (2*r*T - d2*σ*sqrtT) / (2*T*σ*sqrtT)
    else
        charm = -n_d1 * (2*r*T - d2*σ*sqrtT) / (2*T*σ*sqrtT)
    end

    return GreeksResult(delta, gamma, vega, theta, rho, vanna, volga, charm)
end

# ============================================================================
# AD Greeks - Fallback for exotics without closed-form solutions
# ============================================================================

"""
    _ad_greeks(option, market_state, backend)

Compute Greeks using automatic differentiation.
Use for exotic options without analytical formulas (Asian, barrier, etc.).
Supports configurable AD backend.
"""
function _ad_greeks(opt::AbstractOption, state::MarketState, backend=current_backend())
    S = state.prices[opt.underlying]
    K = opt.strike
    T = opt.expiry
    r = first(values(state.rates))
    σ = state.volatilities[opt.underlying]

    # Price function for first-order Greeks
    # Pack parameters for AD: [S, σ, T, r]
    x = [S, σ, T, r]

    function price_fn(params)
        S_, σ_, T_, r_ = params
        black_scholes(S_, K, T_, r_, σ_, opt.optiontype)
    end

    # Use ForwardDiff for gradient (most reliable for these computations)
    grad = ForwardDiff.gradient(price_fn, x)
    delta = grad[1]
    vega = grad[2] * 0.01   # Scale to per-1% vol move
    theta = -grad[3]        # Negative because we want time decay
    rho = grad[4] * 0.01    # Scale to per-1% rate move

    # Second derivatives via nested ForwardDiff
    # Gamma: d²V/dS²
    gamma = ForwardDiff.derivative(s -> ForwardDiff.derivative(
        s_ -> black_scholes(s_, K, T, r, σ, opt.optiontype), s
    ), S)

    # Vanna: d²V/dSdσ
    vanna = ForwardDiff.derivative(σ_ -> ForwardDiff.derivative(
        s_ -> black_scholes(s_, K, T, r, σ_, opt.optiontype), S
    ), σ)

    # Volga: d²V/dσ²
    volga = ForwardDiff.derivative(σ_ -> ForwardDiff.derivative(
        σ__ -> black_scholes(S, K, T, r, σ__, opt.optiontype), σ_
    ), σ) * 0.01  # Scale for 1% vol

    # Charm: d²V/dSdT (negative for time decay convention)
    charm = -ForwardDiff.derivative(t_ -> ForwardDiff.derivative(
        s_ -> black_scholes(s_, K, t_, r, σ, opt.optiontype), S
    ), T)

    return GreeksResult(delta, gamma, vega, theta, rho, vanna, volga, charm)
end

export GreeksResult, compute_greeks
