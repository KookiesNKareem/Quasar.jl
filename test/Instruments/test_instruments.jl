using Test
using Quasar

@testset "Instruments" begin
    @testset "Stock" begin
        stock = Stock("AAPL")

        @test stock.symbol == "AAPL"
        @test stock isa AbstractEquity
        @test stock isa AbstractInstrument

        # Traits
        @test ispriceable(stock)
        @test isdifferentiable(stock)

        # Pricing
        state = MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )
        @test price(stock, state) == 150.0
    end

    @testset "EuropeanOption" begin
        # Call option
        call = EuropeanOption("AAPL", 150.0, 1.0, :call)

        @test call.underlying == "AAPL"
        @test call.strike == 150.0
        @test call.expiry == 1.0
        @test call.optiontype == :call
        @test call isa AbstractOption

        # Traits
        @test ispriceable(call)
        @test isdifferentiable(call)
        @test hasgreeks(call)

        # Black-Scholes pricing (ATM call, 1 year, 20% vol, 5% rate)
        state = MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        # ATM call should be worth roughly S * 0.4 * σ * √T for small σ
        # More precisely, use known BS value
        p = price(call, state)
        @test p > 0.0
        @test p < 150.0  # Can't be worth more than underlying

        # Put option
        put = EuropeanOption("AAPL", 150.0, 1.0, :put)
        p_put = price(put, state)

        # Put-call parity: C - P = S - K*exp(-rT)
        S = 150.0
        K = 150.0
        r = 0.05
        T = 1.0
        @test p - p_put ≈ S - K * exp(-r * T) atol=1e-10
    end

    @testset "Greeks via AD" begin
        call = EuropeanOption("AAPL", 150.0, 1.0, :call)

        state = MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        # Get all Greeks
        greeks = compute_greeks(call, state)

        # Delta: dV/dS, should be between 0 and 1 for call
        @test 0.0 < greeks.delta < 1.0

        # Gamma: d²V/dS², should be positive
        @test greeks.gamma > 0.0

        # Vega: dV/dσ, should be positive
        @test greeks.vega > 0.0

        # Theta: dV/dT, typically negative for long options
        @test greeks.theta < 0.0

        # Rho: dV/dr
        @test greeks.rho > 0.0  # Positive for calls

        # Validate against analytical Black-Scholes Greeks
        analytical = analytical_greeks(call, state)

        @test greeks.delta ≈ analytical.delta atol=1e-8
        @test greeks.gamma ≈ analytical.gamma atol=1e-8
        @test greeks.vega ≈ analytical.vega atol=1e-6
        @test greeks.theta ≈ analytical.theta atol=1e-6
        @test greeks.rho ≈ analytical.rho atol=1e-6
    end
end
