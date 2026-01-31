module QuasarReactantExt

using Quasar
using Quasar.AD: ReactantBackend, _gradient, _hessian, _jacobian

# Only load if Reactant is available
using Reactant

# Placeholder implementations - to be filled in when Reactant API is finalized
function Quasar.AD._gradient(::ReactantBackend, f, x)
    # TODO: Implement using Reactant + Enzyme
    # compiled_grad = Reactant.@compile ...
    error("ReactantBackend gradient implementation coming soon. Use ForwardDiffBackend() for now.")
end

function Quasar.AD._hessian(::ReactantBackend, f, x)
    error("ReactantBackend hessian not yet implemented.")
end

function Quasar.AD._jacobian(::ReactantBackend, f, x)
    error("ReactantBackend jacobian not yet implemented.")
end

end
