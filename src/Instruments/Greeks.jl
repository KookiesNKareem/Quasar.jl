# Greeks computation via AD and analytical formulas

using ForwardDiff
using Distributions: Normal, cdf, pdf

"""
    GreeksResult

Container for option Greeks.
"""
struct GreeksResult{T}
    delta::T    # dV/dS
    gamma::T    # d²V/dS²
    vega::T     # dV/dσ (per 1% move, scaled by 0.01)
    theta::T    # -dV/dT (time decay per year)
    rho::T      # dV/dr (per 1% move, scaled by 0.01)
end

"""
    compute_greeks(option, market_state; backend=current_backend())

Compute option Greeks using automatic differentiation.
This is the AD-first approach - works for any priceable option.
"""
function compute_greeks(opt::AbstractOption, state::MarketState)
    S = state.prices[opt.underlying]
    K = opt.strike
    T = opt.expiry
    r = first(values(state.rates))
    σ = state.volatilities[opt.underlying]

    # Pack parameters for AD: [S, σ, T, r]
    x = [S, σ, T, r]

    # Price function of packed parameters
    function price_fn(params)
        S_, σ_, T_, r_ = params
        black_scholes(S_, K, T_, r_, σ_, opt.optiontype)
    end

    # First derivatives via AD
    grad = ForwardDiff.gradient(price_fn, x)
    delta = grad[1]
    vega = grad[2] * 0.01   # Scale to per-1% vol move
    theta = -grad[3]        # Negative because we want time decay
    rho = grad[4] * 0.01    # Scale to per-1% rate move

    # Second derivative (gamma) via nested dual
    gamma = ForwardDiff.derivative(s -> ForwardDiff.derivative(
        s_ -> black_scholes(s_, K, T, r, σ, opt.optiontype), s
    ), S)

    return GreeksResult(delta, gamma, vega, theta, rho)
end

"""
    analytical_greeks(option, market_state)

Compute Greeks using analytical Black-Scholes formulas.
Used as test oracle to validate AD implementation.
"""
function analytical_greeks(opt::EuropeanOption, state::MarketState)
    S = state.prices[opt.underlying]
    K = opt.strike
    T = opt.expiry
    r = first(values(state.rates))
    σ = state.volatilities[opt.underlying]

    d1 = (log(S/K) + (r + σ^2/2)*T) / (σ*sqrt(T))
    d2 = d1 - σ*sqrt(T)

    N = Normal()
    n_d1 = pdf(N, d1)
    N_d1 = cdf(N, d1)
    N_d2 = cdf(N, d2)

    if opt.optiontype == :call
        delta = N_d1
        theta = -S * n_d1 * σ / (2*sqrt(T)) - r * K * exp(-r*T) * N_d2
        rho = K * T * exp(-r*T) * N_d2 * 0.01
    else  # put
        delta = N_d1 - 1
        theta = -S * n_d1 * σ / (2*sqrt(T)) + r * K * exp(-r*T) * cdf(N, -d2)
        rho = -K * T * exp(-r*T) * cdf(N, -d2) * 0.01
    end

    # Gamma and Vega are same for calls and puts
    gamma = n_d1 / (S * σ * sqrt(T))
    vega = S * n_d1 * sqrt(T) * 0.01

    return GreeksResult(delta, gamma, vega, theta, rho)
end

export GreeksResult, compute_greeks, analytical_greeks
