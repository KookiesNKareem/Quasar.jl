module QuasarReactantExt

using Quasar
using Quasar.AD: ReactantBackend, _gradient, _hessian, _jacobian, _value_and_gradient
using Quasar.Core: ADBackend

using Reactant
using Enzyme

# ============================================================================
# Reactant Backend Implementation
#
# Reactant compiles Julia functions to XLA via MLIR. For autodiff, we use
# Enzyme inside compiled functions - Reactant's EnzymeMLIR handles the
# compilation of Enzyme operations to efficient XLA code.
#
# Gradient and value_and_gradient are fully Reactant-accelerated.
# Hessian and Jacobian use Enzyme directly (Reactant nested compilation is complex).
# ============================================================================

function Quasar.AD._gradient(::ReactantBackend, f, x)
    x_react = x isa Reactant.ConcreteRArray ? x : Reactant.ConcreteRArray(x)

    # Define gradient function using Enzyme
    function grad_fn(x_in)
        dx = zero(x_in)
        Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Duplicated(x_in, dx))
        return dx
    end

    # Compile to XLA
    compiled = Reactant.@compile grad_fn(x_react)
    result = compiled(x_react)

    return Array(result)
end

function Quasar.AD._value_and_gradient(::ReactantBackend, f, x)
    x_react = x isa Reactant.ConcreteRArray ? x : Reactant.ConcreteRArray(x)

    # Define function that returns both value and gradient
    function val_grad_fn(x_in)
        dx = zero(x_in)
        _, val = Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(x_in, dx))
        return val, dx
    end

    # Compile to XLA
    compiled = Reactant.@compile val_grad_fn(x_react)
    val_result, grad_result = compiled(x_react)

    # Extract scalar value
    val_out = Reactant.@allowscalar Float64(val_result[])
    return (val_out, Array(grad_result))
end

function Quasar.AD._hessian(::ReactantBackend, f, x)
    # Use Enzyme directly for hessian (nested Reactant compilation is complex)
    n = length(x)
    H = zeros(eltype(x), n, n)

    for i in 1:n
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

function Quasar.AD._jacobian(::ReactantBackend, f, x)
    # Use Enzyme directly for jacobian
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

end # module
