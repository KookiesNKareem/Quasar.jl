using Test
using SuperNova
using Statistics

@testset "Full Pipeline Integration" begin
    @testset "Stock portfolio optimization" begin
        # Create stocks
        stocks = [Stock("AAPL"), Stock("GOOG"), Stock("MSFT")]

        # Historical return estimates
        expected_returns = [0.10, 0.15, 0.12]  # Annual
        cov_matrix = [
            0.04 0.01 0.02;
            0.01 0.09 0.01;
            0.02 0.01 0.05
        ]

        # Optimize for target return
        result = optimize(
            MeanVariance(expected_returns, cov_matrix),
            target_return=0.12
        )

        # Verify weights are valid (either converged or approximate solution is good)
        @test sum(result.weights) ≈ 1.0 atol=1e-4
        @test all(result.weights .>= -1e-6)  # Non-negative weights

        # Create portfolio with optimized weights
        portfolio = Portfolio(stocks, result.weights)

        # Value portfolio
        state = MarketState(
            prices=Dict("AAPL" => 150.0, "GOOG" => 140.0, "MSFT" => 300.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2, "GOOG" => 0.25, "MSFT" => 0.18),
            timestamp=0.0
        )

        portfolio_value = value(portfolio, state)
        @test portfolio_value > 0
    end

    @testset "Options portfolio with Greeks" begin
        # Create option portfolio
        call1 = EuropeanOption("AAPL", 150.0, 0.5, :call)
        call2 = EuropeanOption("AAPL", 160.0, 0.5, :call)
        put1 = EuropeanOption("AAPL", 140.0, 0.5, :put)

        # Bull call spread + protective put
        portfolio = Portfolio(
            [call1, call2, put1],
            [10.0, -10.0, 5.0]  # Long 10 lower calls, short 10 higher calls, long 5 puts
        )

        state = MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        # Get portfolio Greeks
        greeks = portfolio_greeks(portfolio, state)

        # Bull call spread has limited delta
        @test abs(greeks.delta) < 10.0  # Bounded by spread

        # Value the portfolio
        pv = value(portfolio, state)
        @test pv isa Float64
    end

    @testset "Risk measures on simulated returns" begin
        # Simulate portfolio returns
        n_days = 252
        daily_returns = 0.0005 .+ randn(n_days) * 0.015  # ~15% annual vol, slight positive drift

        # Compute risk measures
        var95 = compute(VaR(0.95), daily_returns)
        cvar95 = compute(CVaR(0.95), daily_returns)
        vol = compute(Volatility(), daily_returns)
        sharpe = compute(Sharpe(rf=0.0), daily_returns)
        mdd = compute(MaxDrawdown(), daily_returns)

        @test var95 < 0
        @test cvar95 <= var95
        @test vol > 0
        @test mdd <= 0

        # Annualized Sharpe should be reasonable
        annual_sharpe = sharpe * sqrt(252)
        @test -2 < annual_sharpe < 5
    end

    @testset "AD backend switching" begin
        f(x) = sum(x.^2)
        x = [1.0, 2.0, 3.0]

        # ForwardDiff
        set_backend!(ForwardDiffBackend())
        g1 = gradient(f, x)

        # PureJulia
        set_backend!(PureJuliaBackend())
        g2 = gradient(f, x)

        # Should give same result
        @test g1 ≈ g2 atol=1e-5

        # Reset
        set_backend!(ForwardDiffBackend())
    end
end
