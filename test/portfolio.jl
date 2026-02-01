using Test
using SuperNova

@testset "Portfolio" begin
    @testset "Construction" begin
        stock1 = Stock("AAPL")
        stock2 = Stock("GOOG")
        call = EuropeanOption("AAPL", 150.0, 1.0, :call)

        # From instruments and weights
        portfolio = Portfolio(
            [stock1, stock2, call],
            [0.4, 0.4, 0.2]
        )

        @test length(portfolio) == 3
        @test portfolio.weights ≈ [0.4, 0.4, 0.2]
    end

    @testset "Portfolio value" begin
        stock = Stock("AAPL")
        call = EuropeanOption("AAPL", 150.0, 1.0, :call)

        portfolio = Portfolio([stock, call], [1.0, 10.0])  # 1 share + 10 options

        state = MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        stock_value = price(stock, state) * 1.0
        option_value = price(call, state) * 10.0

        @test value(portfolio, state) ≈ stock_value + option_value
    end

    @testset "Portfolio Greeks aggregation" begin
        call1 = EuropeanOption("AAPL", 150.0, 1.0, :call)
        call2 = EuropeanOption("AAPL", 160.0, 1.0, :call)

        portfolio = Portfolio([call1, call2], [10.0, 5.0])

        state = MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        greeks = portfolio_greeks(portfolio, state)

        # Portfolio delta = sum of position deltas
        g1 = compute_greeks(call1, state)
        g2 = compute_greeks(call2, state)

        @test greeks.delta ≈ g1.delta * 10.0 + g2.delta * 5.0
        @test greeks.gamma ≈ g1.gamma * 10.0 + g2.gamma * 5.0
    end
end
