module AD

using ..Core: ADBackend
using ForwardDiff
using DiffResults

# ============================================================================
# Backend Types
# ============================================================================

"""
    PureJuliaBackend <: ADBackend

Reference implementation using finite differences. Slow but always works.
Useful for debugging and testing.
"""
struct PureJuliaBackend <: ADBackend end

"""
    ForwardDiffBackend <: ADBackend

CPU-based forward-mode AD using ForwardDiff.jl.
Best for low-dimensional problems and nested derivatives (Greeks).
"""
struct ForwardDiffBackend <: ADBackend end

"""
    ReactantBackend <: ADBackend

GPU-accelerated AD using Reactant.jl + Enzyme.
Best for high-dimensional problems (portfolio optimization).
"""
struct ReactantBackend <: ADBackend end

"""
    EnzymeBackend <: ADBackend

CPU/GPU AD using Enzyme.jl (LLVM-based differentiation).
Supports both forward and reverse mode.
"""
struct EnzymeBackend <: ADBackend
    mode::Symbol  # :forward or :reverse
end
EnzymeBackend() = EnzymeBackend(:reverse)

# ============================================================================
# Global Backend State
# ============================================================================

const CURRENT_BACKEND = Ref{ADBackend}(ForwardDiffBackend())

"""
    current_backend()

Return the currently active AD backend.
"""
current_backend() = CURRENT_BACKEND[]

"""
    set_backend!(backend::ADBackend)

Set the global AD backend.
"""
function set_backend!(backend::ADBackend)
    CURRENT_BACKEND[] = backend
    return backend
end

"""
    with_backend(f, backend::ADBackend)

Execute `f` with `backend` as the active backend, then restore the original.

# Example
```julia
with_backend(EnzymeBackend()) do
    gradient(loss, params)
end
```
"""
function with_backend(f, backend::ADBackend)
    old = current_backend()
    set_backend!(backend)
    try
        return f()
    finally
        set_backend!(old)
    end
end

# ============================================================================
# Gradient Interface
# ============================================================================

"""
    validate_ad_input(x)

Check that input is valid for automatic differentiation.
Throws ArgumentError if NaN or Inf values are detected.
"""
function validate_ad_input(x)
    if any(isnan, x)
        throw(ArgumentError("Input contains NaN values - cannot compute gradient"))
    end
    if any(isinf, x)
        throw(ArgumentError("Input contains Inf values - cannot compute gradient"))
    end
    nothing
end

"""
    gradient(f, x; backend=current_backend())

Compute the gradient of `f` at `x` using the specified backend.
Validates input for NaN/Inf values before computing.
"""
function gradient(f, x; backend=current_backend())
    validate_ad_input(x)
    _gradient(backend, f, x)
end

# ForwardDiff implementation
function _gradient(::ForwardDiffBackend, f, x)
    ForwardDiff.gradient(f, x)
end

# PureJulia implementation (finite differences with central differences for better accuracy)
function _gradient(::PureJuliaBackend, f, x; eps=1e-7)
    if get(ENV, "QUANTNOVA_WARN_PUREJULIA", "1") != "0"
        @warn "Using PureJuliaBackend for gradient computation. This is slow and less accurate than ForwardDiff. Consider using ForwardDiffBackend." maxlog=1
    end
    n = length(x)
    g = similar(x)
    for i in 1:n
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[i] += eps
        x_minus[i] -= eps
        # Central differences: (f(x+h) - f(x-h)) / 2h
        g[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    end
    return g
end

# Fallback for unloaded backends
function _gradient(b::ADBackend, f, x)
    _throw_backend_not_loaded(b)
end

function _throw_backend_not_loaded(::ReactantBackend)
    error("""
    ReactantBackend requires Reactant.jl to be loaded.

    To use GPU acceleration:
        using Reactant
        using QuantNova

    Then set_backend!(ReactantBackend()) will work.
    """)
end

function _throw_backend_not_loaded(::EnzymeBackend)
    error("""
    EnzymeBackend requires Enzyme.jl to be loaded.

    To use Enzyme:
        using Enzyme
        using QuantNova

    Then set_backend!(EnzymeBackend()) will work.
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

# Fallback for unloaded backends
function _hessian(b::ADBackend, f, x)
    _throw_backend_not_loaded(b)
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

# Fallback for unloaded backends
function _jacobian(b::ADBackend, f, x)
    _throw_backend_not_loaded(b)
end

# ============================================================================
# Value and Gradient Interface
# ============================================================================

"""
    value_and_gradient(f, x; backend=current_backend())

Compute the value and gradient of `f` at `x` in a single pass.
Returns `(f(x), ∇f(x))`.
"""
function value_and_gradient(f, x; backend=current_backend())
    _value_and_gradient(backend, f, x)
end

function _value_and_gradient(::ForwardDiffBackend, f, x)
    result = DiffResults.GradientResult(x)
    ForwardDiff.gradient!(result, f, x)
    (DiffResults.value(result), DiffResults.gradient(result))
end

function _value_and_gradient(::PureJuliaBackend, f, x)
    val = f(x)
    grad = _gradient(PureJuliaBackend(), f, x)
    (val, grad)
end

# Fallback for unloaded backends
function _value_and_gradient(b::ADBackend, f, x)
    _throw_backend_not_loaded(b)
end

# ============================================================================
# GPU Initialization
# ============================================================================

"""
    enable_gpu!(backend::Symbol=:auto)

Initialize GPU backend. Detects available hardware and loads appropriate extension.

# Arguments
- `:auto` - Prefer Reactant if available, fall back to Enzyme
- `:enzyme` - Use Enzyme + CUDA.jl
- `:reactant` - Use Reactant + XLA

# Example
```julia
using Reactant
using QuantNova
enable_gpu!()  # auto-detects Reactant
```
"""
function enable_gpu!(backend::Symbol=:auto)
    if backend == :auto
        if isdefined(Main, :Reactant)
            return _enable_reactant_gpu()
        elseif isdefined(Main, :Enzyme)
            return _enable_enzyme_gpu()
        else
            error("No GPU backend available. Run `using Reactant` or `using Enzyme` first.")
        end
    elseif backend == :reactant
        isdefined(Main, :Reactant) || error("Reactant not loaded. Run `using Reactant` first.")
        return _enable_reactant_gpu()
    elseif backend == :enzyme
        isdefined(Main, :Enzyme) || error("Enzyme not loaded. Run `using Enzyme` first.")
        return _enable_enzyme_gpu()
    else
        error("Unknown backend: $backend. Use :auto, :enzyme, or :reactant.")
    end
end

function _enable_reactant_gpu()
    set_backend!(ReactantBackend())
    @info "GPU enabled via Reactant (XLA backend)"
    current_backend()
end

function _enable_enzyme_gpu()
    set_backend!(EnzymeBackend())
    @info "GPU enabled via Enzyme (CUDA backend)"
    current_backend()
end

# ============================================================================
# Exports
# ============================================================================

export PureJuliaBackend, ForwardDiffBackend, ReactantBackend, EnzymeBackend
export current_backend, set_backend!, with_backend, enable_gpu!
export gradient, hessian, jacobian, value_and_gradient

end
