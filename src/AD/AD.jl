module AD

using ..Core: AbstractADBackend
using ForwardDiff

# ============================================================================
# Backend Types
# ============================================================================

"""
    PureJuliaBackend <: AbstractADBackend

Reference implementation using finite differences. Slow but always works.
Useful for debugging and testing.
"""
struct PureJuliaBackend <: AbstractADBackend end

"""
    ForwardDiffBackend <: AbstractADBackend

CPU-based forward-mode AD using ForwardDiff.jl.
Best for low-dimensional problems and nested derivatives (Greeks).
"""
struct ForwardDiffBackend <: AbstractADBackend end

"""
    ReactantBackend <: AbstractADBackend

GPU-accelerated AD using Reactant.jl + Enzyme.
Best for high-dimensional problems (portfolio optimization).
"""
struct ReactantBackend <: AbstractADBackend end

# ============================================================================
# Global Backend State
# ============================================================================

const CURRENT_BACKEND = Ref{AbstractADBackend}(ForwardDiffBackend())

"""
    current_backend()

Return the currently active AD backend.
"""
current_backend() = CURRENT_BACKEND[]

"""
    set_backend!(backend::AbstractADBackend)

Set the global AD backend.
"""
function set_backend!(backend::AbstractADBackend)
    CURRENT_BACKEND[] = backend
    return backend
end

# ============================================================================
# Gradient Interface
# ============================================================================

"""
    gradient(f, x; backend=current_backend())

Compute the gradient of `f` at `x` using the specified backend.
"""
function gradient(f, x; backend=current_backend())
    _gradient(backend, f, x)
end

# ForwardDiff implementation
function _gradient(::ForwardDiffBackend, f, x)
    ForwardDiff.gradient(f, x)
end

# PureJulia implementation (finite differences)
function _gradient(::PureJuliaBackend, f, x; eps=1e-7)
    n = length(x)
    g = similar(x)
    f0 = f(x)
    for i in 1:n
        x_plus = copy(x)
        x_plus[i] += eps
        g[i] = (f(x_plus) - f0) / eps
    end
    return g
end

# Reactant implementation (placeholder - overridden by extension when Reactant is loaded)
function _gradient(::ReactantBackend, f, x)
    error("""
    ReactantBackend requires Reactant.jl to be loaded.

    To use GPU acceleration:
        using Pkg
        Pkg.add("Reactant")
        using Reactant
        using Quasar

    Then set_backend!(ReactantBackend()) will work.
    """)
end

# ============================================================================
# Hessian Interface
# ============================================================================

"""
    hessian(f, x; backend=current_backend())

Compute the Hessian of `f` at `x` using the specified backend.
"""
function hessian(f, x; backend=current_backend())
    _hessian(backend, f, x)
end

function _hessian(::ForwardDiffBackend, f, x)
    ForwardDiff.hessian(f, x)
end

function _hessian(::PureJuliaBackend, f, x; eps=1e-5)
    n = length(x)
    H = zeros(eltype(x), n, n)
    for i in 1:n
        for j in 1:n
            x_pp = copy(x); x_pp[i] += eps; x_pp[j] += eps
            x_pm = copy(x); x_pm[i] += eps; x_pm[j] -= eps
            x_mp = copy(x); x_mp[i] -= eps; x_mp[j] += eps
            x_mm = copy(x); x_mm[i] -= eps; x_mm[j] -= eps
            H[i,j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4*eps^2)
        end
    end
    return H
end

function _hessian(::ReactantBackend, f, x)
    error("ReactantBackend requires Reactant.jl. See `_gradient` error for setup instructions.")
end

# ============================================================================
# Jacobian Interface
# ============================================================================

"""
    jacobian(f, x; backend=current_backend())

Compute the Jacobian of `f` at `x` using the specified backend.
"""
function jacobian(f, x; backend=current_backend())
    _jacobian(backend, f, x)
end

function _jacobian(::ForwardDiffBackend, f, x)
    ForwardDiff.jacobian(f, x)
end

function _jacobian(::PureJuliaBackend, f, x; eps=1e-7)
    f0 = f(x)
    m = length(f0)
    n = length(x)
    J = zeros(eltype(x), m, n)
    for j in 1:n
        x_plus = copy(x)
        x_plus[j] += eps
        J[:, j] = (f(x_plus) - f0) / eps
    end
    return J
end

function _jacobian(::ReactantBackend, f, x)
    error("ReactantBackend requires Reactant.jl. See `_gradient` error for setup instructions.")
end

# ============================================================================
# Exports
# ============================================================================

export PureJuliaBackend, ForwardDiffBackend, ReactantBackend
export current_backend, set_backend!
export gradient, hessian, jacobian

end
