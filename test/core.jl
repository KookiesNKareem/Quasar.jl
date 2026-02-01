using Test
using SuperNova

@testset "Core Abstract Types" begin
    @testset "Type hierarchy exists" begin
        @test AbstractInstrument <: Any
        @test AbstractEquity <: AbstractInstrument
        @test AbstractDerivative <: AbstractInstrument
        @test AbstractOption <: AbstractDerivative
        @test AbstractPortfolio <: Any
        @test AbstractRiskMeasure <: Any
        @test ADBackend <: Any
    end
end

@testset "MarketState" begin
    state = MarketState(
        prices=Dict("AAPL" => 150.0, "GOOG" => 140.0),
        rates=Dict("USD" => 0.05),
        volatilities=Dict("AAPL" => 0.2, "GOOG" => 0.25),
        timestamp=0.0
    )

    @test state.prices["AAPL"] == 150.0
    @test state.rates["USD"] == 0.05
    @test state.volatilities["AAPL"] == 0.2
    @test state.timestamp == 0.0

    # Immutability - should error on modification attempt
    @test_throws MethodError state.prices["AAPL"] = 160.0

    # Input validation tests
    @testset "MarketState validation" begin
        # Negative price should throw
        @test_throws ArgumentError MarketState(
            prices=Dict("AAPL" => -150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        # Zero price should throw
        @test_throws ArgumentError MarketState(
            prices=Dict("AAPL" => 0.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        # NaN price should throw
        @test_throws ArgumentError MarketState(
            prices=Dict("AAPL" => NaN),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        # Inf price should throw
        @test_throws ArgumentError MarketState(
            prices=Dict("AAPL" => Inf),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        # Negative volatility should throw
        @test_throws ArgumentError MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => -0.2),
            timestamp=0.0
        )

        # Zero volatility should throw
        @test_throws ArgumentError MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.0),
            timestamp=0.0
        )

        # NaN rate should throw
        @test_throws ArgumentError MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => NaN),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )
    end
end

@testset "ImmutableDict" begin
    d = SuperNova.Core.ImmutableDict(Dict("a" => 1, "b" => 2))

    # Reading works
    @test d["a"] == 1
    @test d["b"] == 2
    @test haskey(d, "a")
    @test !haskey(d, "c")
    @test length(d) == 2
    @test Set(keys(d)) == Set(["a", "b"])
    @test Set(values(d)) == Set([1, 2])

    # Modification should throw
    @test_throws MethodError d["a"] = 10
    @test_throws MethodError d["c"] = 3

    # Verify original is unchanged
    @test d["a"] == 1

    # Iteration works
    items = [(k, v) for (k, v) in d]
    @test length(items) == 2
end

@testset "Traits" begin
    # Test trait types exist
    @test Priceable isa Type
    @test Differentiable isa Type
    @test HasGreeks isa Type
    @test Simulatable isa Type

    # Test trait query functions
    struct MockInstrument <: AbstractInstrument end

    # Default should be false/not-trait
    @test !ispriceable(MockInstrument())
    @test !isdifferentiable(MockInstrument())
    @test !hasgreeks(MockInstrument())
    @test !issimulatable(MockInstrument())
end
