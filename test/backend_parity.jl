using Test
using Enzyme
using Reactant
using SuperNova

@testset "Backend Parity" begin
    f(x) = sum(x.^2) + prod(x)
    x = [1.0, 2.0, 3.0]

    # Reference: ForwardDiff
    fd_grad = SuperNova.gradient(f, x; backend=ForwardDiffBackend())
    fd_val, fd_grad2 = SuperNova.value_and_gradient(f, x; backend=ForwardDiffBackend())

    @testset "PureJulia matches ForwardDiff" begin
        pj_grad = SuperNova.gradient(f, x; backend=PureJuliaBackend())
        @test pj_grad ≈ fd_grad atol=1e-6

        pj_val, pj_grad2 = SuperNova.value_and_gradient(f, x; backend=PureJuliaBackend())
        @test pj_val ≈ fd_val
        @test pj_grad2 ≈ fd_grad2 atol=1e-6
    end

    @testset "Enzyme matches ForwardDiff" begin
        enz_grad = SuperNova.gradient(f, x; backend=EnzymeBackend())
        @test enz_grad ≈ fd_grad atol=1e-10

        enz_val, enz_grad2 = SuperNova.value_and_gradient(f, x; backend=EnzymeBackend())
        @test enz_val ≈ fd_val
        @test enz_grad2 ≈ fd_grad2 atol=1e-10
    end

    @testset "Hessian Parity" begin
        h(x) = sum(x.^2) + x[1]*x[2]*x[3]  # Has off-diagonal terms

        fd_hess = SuperNova.hessian(h, x; backend=ForwardDiffBackend())

        pj_hess = SuperNova.hessian(h, x; backend=PureJuliaBackend())
        @test pj_hess ≈ fd_hess atol=1e-4  # Finite diff less precise

        enz_hess = SuperNova.hessian(h, x; backend=EnzymeBackend())
        @test enz_hess ≈ fd_hess atol=1e-10
    end

    @testset "Jacobian Parity" begin
        g(x) = [x[1]^2 + x[2], x[2]*x[3], x[1] + x[2] + x[3]]

        fd_jac = SuperNova.jacobian(g, x; backend=ForwardDiffBackend())

        pj_jac = SuperNova.jacobian(g, x; backend=PureJuliaBackend())
        @test pj_jac ≈ fd_jac atol=1e-6

        enz_jac = SuperNova.jacobian(g, x; backend=EnzymeBackend())
        @test enz_jac ≈ fd_jac atol=1e-10
    end

    @testset "Reactant matches ForwardDiff" begin
        # Use simpler function - Reactant doesn't support prod() reduction in reverse mode
        f_simple(x) = sum(x.^2) + sum(x)

        fd_grad_simple = SuperNova.gradient(f_simple, x; backend=ForwardDiffBackend())
        react_grad = SuperNova.gradient(f_simple, x; backend=ReactantBackend())
        @test react_grad ≈ fd_grad_simple atol=1e-10

        fd_val_simple, fd_grad2_simple = SuperNova.value_and_gradient(f_simple, x; backend=ForwardDiffBackend())
        react_val, react_grad2 = SuperNova.value_and_gradient(f_simple, x; backend=ReactantBackend())
        @test react_val ≈ fd_val_simple
        @test react_grad2 ≈ fd_grad2_simple atol=1e-10

        # Hessian (uses Enzyme directly, so prod is fine)
        h(x) = sum(x.^2) + x[1]*x[2]*x[3]
        fd_hess = SuperNova.hessian(h, x; backend=ForwardDiffBackend())
        react_hess = SuperNova.hessian(h, x; backend=ReactantBackend())
        @test react_hess ≈ fd_hess atol=1e-10

        # Jacobian (uses Enzyme directly)
        g(x) = [x[1]^2 + x[2], x[2]*x[3], x[1] + x[2] + x[3]]
        fd_jac = SuperNova.jacobian(g, x; backend=ForwardDiffBackend())
        react_jac = SuperNova.jacobian(g, x; backend=ReactantBackend())
        @test react_jac ≈ fd_jac atol=1e-10
    end
end
