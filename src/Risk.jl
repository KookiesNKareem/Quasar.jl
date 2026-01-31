module Risk

# TODO: Add historical simulation VaR
# TODO: Add Monte Carlo VaR
# TODO: Add Cornish-Fisher expansion for non-normal VaR
# TODO: Add correlation/covariance analytics
# TODO: Add stress testing framework

using ..Core: AbstractRiskMeasure
using ..PortfolioModule
using Statistics

# ============================================================================
# Risk Measure Types
# ============================================================================

"""
    VaR <: AbstractRiskMeasure

Value at Risk at specified confidence level.
"""
struct VaR <: AbstractRiskMeasure
    confidence::Float64

    function VaR(confidence::Float64=0.95)
        0 < confidence < 1 || error("confidence must be in (0, 1)")
        new(confidence)
    end
end

"""
    CVaR <: AbstractRiskMeasure

Conditional Value at Risk (Expected Shortfall) at specified confidence level.
"""
struct CVaR <: AbstractRiskMeasure
    confidence::Float64

    function CVaR(confidence::Float64=0.95)
        0 < confidence < 1 || error("confidence must be in (0, 1)")
        new(confidence)
    end
end

"""
    Volatility <: AbstractRiskMeasure

Standard deviation of returns.
"""
struct Volatility <: AbstractRiskMeasure end

"""
    Sharpe <: AbstractRiskMeasure

Sharpe ratio (excess return / volatility).
"""
struct Sharpe <: AbstractRiskMeasure
    rf::Float64  # Risk-free rate

    Sharpe(; rf::Float64=0.0) = new(rf)
end

"""
    MaxDrawdown <: AbstractRiskMeasure

Maximum peak-to-trough decline.
"""
struct MaxDrawdown <: AbstractRiskMeasure end

# ============================================================================
# Compute Interface
# ============================================================================

"""
    compute(measure::AbstractRiskMeasure, returns)

Compute the risk measure for given returns.
"""
function compute end

# VaR implementation
function compute(var::VaR, returns::AbstractVector)
    sorted = sort(returns)
    idx = ceil(Int, (1 - var.confidence) * length(sorted))
    idx = max(1, idx)
    return sorted[idx]
end

# CVaR implementation
function compute(cvar::CVaR, returns::AbstractVector)
    var_threshold = compute(VaR(cvar.confidence), returns)
    tail_returns = filter(r -> r <= var_threshold, returns)
    return mean(tail_returns)
end

# Volatility implementation
function compute(::Volatility, returns::AbstractVector)
    return std(returns)
end

# Sharpe implementation
# TODO: Make frequency configurable (daily, weekly, monthly)
# TODO: Add annualization option
function compute(sharpe::Sharpe, returns::AbstractVector)
    excess_return = mean(returns) - sharpe.rf / 252  # FIXME: Hard-coded 252 days
    vol = std(returns)
    return excess_return / vol
end

# MaxDrawdown implementation
function compute(::MaxDrawdown, returns::AbstractVector)
    # Convert returns to price series (starting at 1)
    prices = cumprod(1 .+ returns)

    # Track running maximum
    peak = prices[1]
    max_dd = 0.0

    for p in prices
        peak = max(peak, p)
        drawdown = (p - peak) / peak
        max_dd = min(max_dd, drawdown)
    end

    return max_dd
end

# ============================================================================
# Exports
# ============================================================================

export VaR, CVaR, Volatility, Sharpe, MaxDrawdown
export compute

end
