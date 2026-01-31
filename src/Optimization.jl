module Optimization

using ..Core: ADBackend
using ..AD: gradient, current_backend, ForwardDiffBackend
using LinearAlgebra

# ============================================================================
# Objective Types
# ============================================================================

"""
    MeanVariance

Mean-variance optimization objective (Markowitz).
"""
struct MeanVariance
    expected_returns::Vector{Float64}
    cov_matrix::Matrix{Float64}
end

"""
    SharpeMaximizer

Maximize Sharpe ratio (non-convex).
"""
struct SharpeMaximizer
    expected_returns::Vector{Float64}
    cov_matrix::Matrix{Float64}
    rf::Float64

    SharpeMaximizer(μ, Σ; rf=0.0) = new(μ, Σ, rf)
end

"""
    CVaRObjective

Conditional Value at Risk optimization objective.

Uses parametric (Gaussian) CVaR approximation for optimization.
For more accurate CVaR with non-normal returns, use scenario-based methods.
"""
struct CVaRObjective
    expected_returns::Vector{Float64}
    cov_matrix::Matrix{Float64}
    alpha::Float64

    CVaRObjective(μ, Σ; alpha=0.95) = new(μ, Σ, alpha)
end

"""
    KellyCriterion

Kelly criterion for optimal position sizing.

Maximizes expected log growth: E[log(1 + w'r)]
For Gaussian returns, optimal unconstrained Kelly is w* = Σ⁻¹μ
"""
struct KellyCriterion
    expected_returns::Vector{Float64}
    cov_matrix::Matrix{Float64}
end

"""
    OptimizationResult

Result of portfolio optimization.
"""
# TODO: Add Base.show() for readable output
# TODO: Add portfolio_return and portfolio_vol fields
struct OptimizationResult
    weights::Vector{Float64}
    objective::Float64
    converged::Bool
    iterations::Int
end

# ============================================================================
# Optimize Interface
# ============================================================================

"""
    optimize(objective; kwargs...)

Optimize portfolio weights for given objective.
"""
function optimize end

# Mean-Variance with target return (gradient descent with penalties)
# TODO: Validate covariance matrix is positive definite
# TODO: Check if target_return is achievable (min/max return constraints)
# TODO: Handle singular covariance matrices (use QR decomposition)
# TODO: Consider BFGS/L-BFGS for faster convergence
function optimize(mv::MeanVariance; target_return::Float64, backend=current_backend(), max_iter=5000, tol=1e-10, lr=0.005)
    μ = mv.expected_returns
    Σ = mv.cov_matrix
    n = length(μ)

    # Initialize with equal weights
    w = ones(n) / n

    # Use higher penalty and adaptive learning rate
    penalty = 10000.0

    for i in 1:max_iter
        function obj(weights)
            # Variance
            var = weights' * Σ * weights
            # Return constraint penalty (squared)
            ret_penalty = penalty * (dot(weights, μ) - target_return)^2
            # Sum to 1 penalty
            sum_penalty = penalty * (sum(weights) - 1)^2
            # Non-negativity penalty
            neg_penalty = penalty * sum(max.(-weights, 0).^2)
            return var + ret_penalty + sum_penalty + neg_penalty
        end

        g = gradient(obj, w; backend=backend)

        # Adaptive learning rate (decrease over time)
        current_lr = lr / (1 + i * 0.0001)
        w_new = w - current_lr * g

        # Project to simplex (non-negative, sum to 1)
        w_new = max.(w_new, 0.0)
        if sum(w_new) > 0
            w_new = w_new / sum(w_new)
        else
            w_new = ones(n) / n
        end

        if norm(w_new - w) < tol
            variance = w_new' * Σ * w_new
            return OptimizationResult(w_new, variance, true, i)
        end

        w = w_new
    end

    variance = w' * Σ * w
    return OptimizationResult(w, variance, false, max_iter)
end

# Sharpe Maximizer (gradient-based)
function optimize(sm::SharpeMaximizer; backend=current_backend(), max_iter=1000, tol=1e-8, lr=0.1)
    μ = sm.expected_returns
    Σ = sm.cov_matrix
    rf = sm.rf
    n = length(μ)

    # Initialize with equal weights
    w = ones(n) / n

    for i in 1:max_iter
        # Negative Sharpe (we minimize)
        function neg_sharpe(weights)
            ret = dot(weights, μ)
            vol = sqrt(weights' * Σ * weights)
            # Add small epsilon to avoid division by zero
            sharpe = (ret - rf) / (vol + 1e-10)

            # Penalties for constraints
            penalty = 100.0
            sum_penalty = penalty * (sum(weights) - 1)^2
            neg_penalty = penalty * sum(max.(-weights, 0).^2)

            return -sharpe + sum_penalty + neg_penalty
        end

        g = gradient(neg_sharpe, w; backend=backend)
        w_new = w - lr * g

        # Project to simplex
        w_new = max.(w_new, 0.0)
        if sum(w_new) > 0
            w_new = w_new / sum(w_new)
        else
            w_new = ones(n) / n
        end

        if norm(w_new - w) < tol
            ret = dot(w_new, μ)
            vol = sqrt(w_new' * Σ * w_new)
            sharpe = (ret - rf) / vol
            return OptimizationResult(w_new, sharpe, true, i)
        end

        w = w_new
    end

    ret = dot(w, μ)
    vol = sqrt(w' * Σ * w)
    sharpe = (ret - rf) / vol
    return OptimizationResult(w, sharpe, false, max_iter)
end

# CVaR Minimization (parametric Gaussian approximation)
"""
    optimize(cvar::CVaRObjective; target_return, backend, max_iter, tol, lr)

Minimize CVaR subject to a target return constraint.

Uses the parametric (Gaussian) CVaR formula:
    CVaR_α = -μ'w + σ(w) * φ(z_α) / (1-α)

where z_α = Φ⁻¹(α) is the VaR quantile and φ is the standard normal PDF.
"""
function optimize(cvar::CVaRObjective; target_return::Float64,
                  backend=current_backend(), max_iter::Int=5000,
                  tol::Float64=1e-10, lr::Float64=0.01)
    μ = cvar.expected_returns
    Σ = cvar.cov_matrix
    α = cvar.alpha
    n = length(μ)

    # Standard normal quantile and PDF at quantile
    z_α = _norminv(α)
    φ_z = exp(-z_α^2 / 2) / sqrt(2π)
    cvar_factor = φ_z / (1 - α)

    # Initialize with equal weights
    w = ones(n) / n
    penalty = 10000.0

    for i in 1:max_iter
        function obj(weights)
            port_return = dot(weights, μ)
            port_vol = sqrt(weights' * Σ * weights + 1e-12)

            # Parametric CVaR (we want to minimize, so negative return + risk term)
            cvar_val = -port_return + port_vol * cvar_factor

            # Constraints
            ret_penalty = penalty * (port_return - target_return)^2
            sum_penalty = penalty * (sum(weights) - 1)^2
            neg_penalty = penalty * sum(max.(-weights, 0).^2)

            return cvar_val + ret_penalty + sum_penalty + neg_penalty
        end

        g = gradient(obj, w; backend=backend)

        current_lr = lr / (1 + i * 0.0001)
        w_new = w - current_lr * g

        # Project to simplex
        w_new = max.(w_new, 0.0)
        if sum(w_new) > 0
            w_new = w_new / sum(w_new)
        else
            w_new = ones(n) / n
        end

        if norm(w_new - w) < tol
            port_return = dot(w_new, μ)
            port_vol = sqrt(w_new' * Σ * w_new)
            cvar_val = -port_return + port_vol * cvar_factor
            return OptimizationResult(w_new, cvar_val, true, i)
        end

        w = w_new
    end

    port_return = dot(w, μ)
    port_vol = sqrt(w' * Σ * w)
    cvar_val = -port_return + port_vol * cvar_factor
    return OptimizationResult(w, cvar_val, false, max_iter)
end

# Kelly Criterion Optimization
"""
    optimize(kelly::KellyCriterion; backend, max_iter, tol, lr, fractional)

Maximize expected log growth rate (Kelly criterion).

The unconstrained Kelly optimal is w* = Σ⁻¹μ, but this can produce extreme
leverage. The `fractional` parameter scales the result (e.g., 0.5 for half-Kelly).

For long-only portfolios, uses gradient descent with simplex projection.
"""
function optimize(kelly::KellyCriterion; backend=current_backend(),
                  max_iter::Int=2000, tol::Float64=1e-10, lr::Float64=0.05,
                  fractional::Float64=1.0, long_only::Bool=true)
    μ = kelly.expected_returns
    Σ = kelly.cov_matrix
    n = length(μ)

    if !long_only
        # Unconstrained Kelly: w* = Σ⁻¹μ (scaled by fractional)
        w_kelly = Σ \ μ
        w_kelly = w_kelly * fractional

        # Normalize to sum to 1 for comparison
        w_normalized = w_kelly / sum(w_kelly)
        growth = dot(w_kelly, μ) - 0.5 * (w_kelly' * Σ * w_kelly)
        return OptimizationResult(w_normalized, growth, true, 1)
    end

    # Long-only: gradient descent with simplex projection
    w = ones(n) / n
    penalty = 1000.0

    for i in 1:max_iter
        # Kelly objective: maximize E[log(1 + w'r)] ≈ w'μ - 0.5 w'Σw (quadratic approx)
        # We minimize negative of this
        function neg_kelly(weights)
            growth = dot(weights, μ) - 0.5 * (weights' * Σ * weights)

            # Constraints
            sum_penalty = penalty * (sum(weights) - 1)^2
            neg_penalty = penalty * sum(max.(-weights, 0).^2)

            return -growth + sum_penalty + neg_penalty
        end

        g = gradient(neg_kelly, w; backend=backend)
        w_new = w - lr * g

        # Project to simplex
        w_new = max.(w_new, 0.0)
        if sum(w_new) > 0
            w_new = w_new / sum(w_new)
        else
            w_new = ones(n) / n
        end

        # Scale by fractional Kelly
        if fractional != 1.0
            w_scaled = w_new * fractional + (1 - fractional) * ones(n) / n
            w_new = w_scaled / sum(w_scaled)
        end

        if norm(w_new - w) < tol
            growth = dot(w_new, μ) - 0.5 * (w_new' * Σ * w_new)
            return OptimizationResult(w_new, growth, true, i)
        end

        w = w_new
    end

    growth = dot(w, μ) - 0.5 * (w' * Σ * w)
    return OptimizationResult(w, growth, false, max_iter)
end

# Helper: inverse normal CDF (Beasley-Springer-Moro approximation)
function _norminv(p::Float64)
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low
        q = sqrt(-2 * log(p))
        return (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
               ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    elseif p <= p_high
        q = p - 0.5
        r = q * q
        return (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6])*q /
               (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1)
    else
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
                ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    end
end

# ============================================================================
# Exports
# ============================================================================

export MeanVariance, SharpeMaximizer, CVaRObjective, KellyCriterion, OptimizationResult
export optimize

end
