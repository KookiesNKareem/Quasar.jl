using Test
using Quasar

@testset "Core Abstract Types" begin
    @testset "Type hierarchy exists" begin
        @test AbstractInstrument <: Any
        @test AbstractEquity <: AbstractInstrument
        @test AbstractDerivative <: AbstractInstrument
        @test AbstractOption <: AbstractDerivative
        @test AbstractPortfolio <: Any
        @test AbstractRiskMeasure <: Any
        @test AbstractADBackend <: Any
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
