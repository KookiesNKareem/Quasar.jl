module SuperNovaEnzymeExt

using SuperNova
using SuperNova.AD: EnzymeBackend, _gradient, _hessian, _jacobian, _value_and_gradient
using SuperNova.Core: ADBackend

using Enzyme

# ============================================================================
# Enzyme Backend Implementation
# ============================================================================

# TODO: Add in-place gradient for memory efficiency
# TODO: Add batch differentiation support
function SuperNova.AD._gradient(b::EnzymeBackend, f, x)
    if b.mode == :reverse
        dx = zero(x)
        Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Duplicated(x, dx))
        return dx
    else
        # Forward mode - compute component by component
        # FIXME: Slow for large vectors - consider chunking
        n = length(x)
        grad = similar(x)
        for i in 1:n
            dx = zeros(length(x))
            dx[i] = 1.0
            grad[i] = Enzyme.autodiff(Enzyme.Forward, f, Enzyme.Duplicated(x, dx))[1]
        end
        return grad
    end
end

function SuperNova.AD._hessian(::EnzymeBackend, f, x)
    n = length(x)
    H = zeros(eltype(x), n, n)
    for i in 1:n
        # Hessian row i = gradient of ∂f/∂x_i
        function grad_i(y)
            dy = zero(y)
            Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Duplicated(y, dy))
            return dy[i]
        end
        dH = zero(x)
        Enzyme.autodiff(Enzyme.Reverse, grad_i, Enzyme.Active, Enzyme.Duplicated(x, dH))
        H[i, :] = dH
    end
    return H
end

function SuperNova.AD._jacobian(::EnzymeBackend, f, x)
    y0 = f(x)
    m = length(y0)
    n = length(x)
    J = zeros(eltype(x), m, n)
    for j in 1:n
        dx = zeros(n)
        dx[j] = 1.0
        dy = Enzyme.autodiff(Enzyme.Forward, f, Enzyme.Duplicated(x, dx))[1]
        J[:, j] = dy
    end
    return J
end

function SuperNova.AD._value_and_gradient(::EnzymeBackend, f, x)
    dx = zero(x)
    _, val = Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(x, dx))
    return (val, dx)
end

end # module
