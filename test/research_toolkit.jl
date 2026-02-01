using Test
using SuperNova
using Dates
using Random

@testset "Research Toolkit Integration" begin
    Random.seed!(42)

    # Generate realistic test data
    n_days = 504  # ~2 years
    timestamps = [DateTime(2022, 1, 3) + Day(i) for i in 0:n_days-1]

    # Simulated returns with realistic properties
    spy_returns = 0.0003 .+ 0.01 .* randn(n_days)  # ~7.5% annual, 16% vol
    agg_returns = 0.0001 .+ 0.003 .* randn(n_days)  # ~2.5% annual, 5% vol

    spy_prices = 400.0 .* cumprod(1 .+ spy_returns)
    agg_prices = 100.0 .* cumprod(1 .+ agg_returns)

    price_series = Dict(:SPY => spy_prices, :AGG => agg_prices)

    @testset "End-to-End Backtest" begin
        # Run 60/40 backtest
        strategy = RebalancingStrategy(
            target_weights=Dict(:SPY => 0.6, :AGG => 0.4),
            rebalance_frequency=:monthly
        )

        result = backtest(
            strategy,
            timestamps,
            price_series;
            initial_cash=100_000.0,
            execution_model=SlippageModel(spread_bps=5.0)
        )

        @test result.final_value > 0
        @test !isempty(result.trades)
        @test result.metrics[:sharpe_ratio] != 0
        @test result.metrics[:max_drawdown] <= 0
    end

    @testset "Backtest + Stress Test" begin
        # Run backtest
        strategy = BuyAndHoldStrategy(Dict(:SPY => 0.7, :AGG => 0.3))
        result = backtest(strategy, timestamps, price_series; initial_cash=100_000.0)

        # Take final state and stress test it
        final_state = SimulationState(
            timestamp=timestamps[end],
            cash=result.equity_curve[end] * 0.05,  # Assume 5% cash
            positions=result.positions_history[end],
            prices=Dict(:SPY => spy_prices[end], :AGG => agg_prices[end])
        )

        asset_classes = Dict(:SPY => :equity, :AGG => :bond)

        # How would the final portfolio fare in past crises?
        for (name, scenario) in CRISIS_SCENARIOS
            impact = scenario_impact(scenario, final_state, asset_classes)
            @test impact isa ScenarioImpact
        end

        # Worst case
        scenarios = collect(values(CRISIS_SCENARIOS))
        worst = worst_case_scenario(scenarios, final_state, asset_classes)
        @test worst.pct_change < 0
    end

    @testset "Monte Carlo Forward Projection" begin
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=10_000.0,
            positions=Dict(:SPY => 100.0, :AGG => 200.0),
            prices=Dict(:SPY => 450.0, :AGG => 100.0)
        )

        projection = monte_carlo_projection(
            state,
            Dict(:SPY => 0.08, :AGG => 0.03),
            Dict(:SPY => 0.16, :AGG => 0.05);
            correlation=0.2,
            horizon_days=252,
            n_simulations=5000
        )

        # Sanity checks
        @test projection.mean_value > state.cash + 100*450 + 200*100  # Should grow on average
        @test projection.var_95 < projection.mean_value  # VaR is below mean
        @test 0 < projection.prob_loss < 1  # Some but not all paths lose money
    end
end
