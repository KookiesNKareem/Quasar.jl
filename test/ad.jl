using Test
using SuperNova

@testset "AD Backend System" begin
    @testset "Backend types exist" begin
        @test PureJuliaBackend <: ADBackend
        @test ForwardDiffBackend <: ADBackend
        @test ReactantBackend <: ADBackend
        @test EnzymeBackend <: ADBackend
        @test EnzymeBackend().mode == :reverse
        @test EnzymeBackend(:forward).mode == :forward
    end

    @testset "Backend selection" begin
        # Default backend should be ForwardDiff (most stable)
        @test current_backend() isa ForwardDiffBackend

        # Can change backend
        set_backend!(PureJuliaBackend())
        @test current_backend() isa PureJuliaBackend

        # Reset for other tests
        set_backend!(ForwardDiffBackend())
    end

    @testset "with_backend context manager" begin
        original = current_backend()
        @test original isa ForwardDiffBackend

        result = with_backend(PureJuliaBackend()) do
            @test current_backend() isa PureJuliaBackend
            42
        end

        @test result == 42
        @test current_backend() isa ForwardDiffBackend  # restored
    end

    @testset "Gradient computation" begin
        f(x) = sum(x.^2)
        x = [1.0, 2.0, 3.0]

        # ForwardDiff backend
        set_backend!(ForwardDiffBackend())
        g = gradient(f, x)
        @test g ≈ [2.0, 4.0, 6.0]

        # PureJulia backend (finite differences)
        set_backend!(PureJuliaBackend())
        g_fd = gradient(f, x)
        @test g_fd ≈ [2.0, 4.0, 6.0] atol=1e-6
    end

    @testset "value_and_gradient" begin
        f(x) = sum(x.^2)
        x = [1.0, 2.0, 3.0]

        set_backend!(ForwardDiffBackend())
        val, grad = value_and_gradient(f, x)

        @test val ≈ 14.0  # 1 + 4 + 9
        @test grad ≈ [2.0, 4.0, 6.0]

        # PureJulia backend
        set_backend!(PureJuliaBackend())
        val2, grad2 = value_and_gradient(f, x)
        @test val2 ≈ 14.0
        @test grad2 ≈ [2.0, 4.0, 6.0] atol=1e-6

        set_backend!(ForwardDiffBackend())  # reset
    end

    @testset "enable_gpu!" begin
        # Should error when no GPU backend loaded
        @test_throws ErrorException enable_gpu!()
        @test_throws ErrorException enable_gpu!(:enzyme)
        @test_throws ErrorException enable_gpu!(:reactant)

        set_backend!(ForwardDiffBackend())  # reset
    end

    @testset "Input validation" begin
        f(x) = sum(x.^2)

        # NaN input should throw
        @test_throws ArgumentError gradient(f, [1.0, NaN, 3.0])
        @test_throws ArgumentError gradient(f, [NaN])

        # Inf input should throw
        @test_throws ArgumentError gradient(f, [1.0, Inf, 3.0])
        @test_throws ArgumentError gradient(f, [-Inf])

        # Valid input should work
        @test gradient(f, [1.0, 2.0, 3.0]) ≈ [2.0, 4.0, 6.0]
    end

    @testset "Gradient accuracy (ForwardDiff vs PureJulia)" begin
        # Test multiple functions
        functions = [
            (x -> sum(x.^2), "quadratic"),
            (x -> sum(sin.(x)), "trigonometric"),
            (x -> sum(exp.(x)), "exponential"),
            (x -> sum(x.^3 .- 2x.^2 .+ x), "polynomial"),
        ]

        for (f, name) in functions
            x = [1.0, 2.0, 3.0]

            set_backend!(ForwardDiffBackend())
            g_exact = gradient(f, x)

            set_backend!(PureJuliaBackend())
            g_fd = gradient(f, x)

            @test g_fd ≈ g_exact atol=1e-5
        end

        set_backend!(ForwardDiffBackend())  # reset
    end

    @testset "Hessian computation" begin
        f(x) = sum(x.^2) + prod(x)  # Non-trivial Hessian
        x = [1.0, 2.0]

        set_backend!(ForwardDiffBackend())
        H_fd = hessian(f, x)

        # For f(x,y) = x² + y² + xy:
        # ∂²f/∂x² = 2, ∂²f/∂y² = 2, ∂²f/∂x∂y = 1
        @test H_fd[1,1] ≈ 2.0
        @test H_fd[2,2] ≈ 2.0
        @test H_fd[1,2] ≈ 1.0
        @test H_fd[2,1] ≈ 1.0  # Symmetry

        # Compare with PureJulia
        set_backend!(PureJuliaBackend())
        H_pure = hessian(f, x)
        @test H_pure ≈ H_fd atol=1e-4

        set_backend!(ForwardDiffBackend())  # reset
    end

    @testset "Jacobian computation" begin
        f(x) = [x[1]^2 + x[2], x[1] * x[2], sin(x[1])]
        x = [1.0, 2.0]

        set_backend!(ForwardDiffBackend())
        J_fd = jacobian(f, x)

        # Expected Jacobian at (1, 2):
        # [2x₁, 1] = [2, 1]
        # [x₂, x₁] = [2, 1]
        # [cos(x₁), 0] = [cos(1), 0]
        @test J_fd[1, 1] ≈ 2.0
        @test J_fd[1, 2] ≈ 1.0
        @test J_fd[2, 1] ≈ 2.0
        @test J_fd[2, 2] ≈ 1.0
        @test J_fd[3, 1] ≈ cos(1.0)
        @test J_fd[3, 2] ≈ 0.0

        # Compare with PureJulia
        set_backend!(PureJuliaBackend())
        J_pure = jacobian(f, x)
        @test J_pure ≈ J_fd atol=1e-5

        set_backend!(ForwardDiffBackend())  # reset
    end
end
