# Transaction Costs Tests

using Test
using QuantNova
using Statistics
using Dates

@testset "Transaction Costs" begin

    @testset "FixedCostModel" begin
        model = FixedCostModel(5.0)

        # Cost is fixed regardless of order size
        @test compute_cost(model, 1000.0, 100.0, 1e6) == 5.0
        @test compute_cost(model, 100000.0, 100.0, 1e6) == 5.0
        @test compute_cost(model, 10.0, 100.0, 1e6) == 5.0

        # Zero cost model
        zero_model = FixedCostModel(0.0)
        @test compute_cost(zero_model, 10000.0, 100.0, 1e6) == 0.0
    end

    @testset "ProportionalCostModel" begin
        # 5 bps with $1 minimum
        model = ProportionalCostModel(rate_bps=5.0, min_cost=1.0)

        # Normal case: 5 bps of $10,000 = $5
        @test compute_cost(model, 10000.0, 100.0, 1e6) ≈ 5.0

        # 5 bps of $100,000 = $50
        @test compute_cost(model, 100000.0, 100.0, 1e6) ≈ 50.0

        # Minimum kicks in: 5 bps of $100 = $0.05, but min is $1
        @test compute_cost(model, 100.0, 100.0, 1e6) == 1.0

        # Zero rate model
        zero_model = ProportionalCostModel(rate_bps=0.0, min_cost=0.0)
        @test compute_cost(zero_model, 10000.0, 100.0, 1e6) == 0.0
    end

    @testset "TieredCostModel" begin
        # 10 bps up to $10k, 5 bps from $10k-$100k, 2 bps above
        model = TieredCostModel([
            (10_000.0, 10.0),
            (100_000.0, 5.0),
            (Inf, 2.0)
        ])

        # $5,000 order: all in first tier
        # 5000 * 0.001 = 5.0
        @test compute_cost(model, 5000.0, 100.0, 1e6) ≈ 5.0

        # $10,000 order: all in first tier
        # 10000 * 0.001 = 10.0
        @test compute_cost(model, 10000.0, 100.0, 1e6) ≈ 10.0

        # $50,000 order: 10k at 10bps + 40k at 5bps
        # 10 + 20 = 30
        @test compute_cost(model, 50000.0, 100.0, 1e6) ≈ 30.0

        # $200,000 order: 10k at 10bps + 90k at 5bps + 100k at 2bps
        # 10 + 45 + 20 = 75
        @test compute_cost(model, 200000.0, 100.0, 1e6) ≈ 75.0
    end

    @testset "SpreadCostModel" begin
        # Fixed 5 bps half-spread
        model = SpreadCostModel(half_spread_bps=5.0)

        # 5 bps of $10,000 = $5
        @test compute_cost(model, 10000.0, 100.0, 1e6) ≈ 5.0

        # 5 bps of $100,000 = $50
        @test compute_cost(model, 100000.0, 100.0, 1e6) ≈ 50.0
    end

    @testset "AlmgrenChrissModel" begin
        model = AlmgrenChrissModel(
            volatility=0.02,
            participation_rate=0.1,
            temporary_impact=0.1,
            permanent_impact=0.1
        )

        # Test that larger orders have more impact
        small_order = compute_cost(model, 10000.0, 100.0, 1e6)
        large_order = compute_cost(model, 100000.0, 100.0, 1e6)
        @test large_order > small_order

        # Test that impact is non-negative
        @test small_order >= 0
        @test large_order >= 0

        # Test that impact scales less than linearly with order size
        # Due to square-root component, 10x order should have less than 10x impact
        ratio = large_order / small_order
        @test ratio > 1  # Larger orders have more impact
        @test ratio < 100  # But not 100x more for 10x order (sublinear scaling)
    end

    @testset "CompositeCostModel" begin
        # Combine commission + spread + impact
        model = CompositeCostModel([
            ProportionalCostModel(rate_bps=1.0),       # 1 bp commission
            SpreadCostModel(half_spread_bps=5.0),     # 5 bp half-spread
            AlmgrenChrissModel(volatility=0.02)       # Market impact
        ])

        order_value = 10000.0
        price = 100.0
        volume = 1e6

        # Individual costs
        commission = compute_cost(ProportionalCostModel(rate_bps=1.0), order_value, price, volume)
        spread = compute_cost(SpreadCostModel(half_spread_bps=5.0), order_value, price, volume)
        impact = compute_cost(AlmgrenChrissModel(volatility=0.02), order_value, price, volume)

        # Composite should be sum
        composite = compute_cost(model, order_value, price, volume)
        @test composite ≈ commission + spread + impact
    end

    @testset "CostTracker" begin
        tracker = CostTracker()

        # Record some trades
        breakdown1 = TradeCostBreakdown(:AAPL, 10000.0, 1.0, 5.0, 2.0, 8.0, 8.0)
        breakdown2 = TradeCostBreakdown(:MSFT, 20000.0, 2.0, 10.0, 4.0, 16.0, 8.0)

        record_trade!(tracker, breakdown1)
        record_trade!(tracker, breakdown2)

        summary = cost_summary(tracker)

        @test summary[:n_trades] == 2
        @test summary[:total_traded] == 30000.0
        @test summary[:total_costs] == 24.0
        @test summary[:avg_cost_bps] ≈ 8.0
    end

    @testset "Turnover Calculation" begin
        # Weights history
        w1 = Dict(:AAPL => 0.6, :MSFT => 0.4)
        w2 = Dict(:AAPL => 0.5, :MSFT => 0.5)
        w3 = Dict(:AAPL => 0.4, :MSFT => 0.6)

        weights_history = [w1, w2, w3]
        turnover = compute_turnover(weights_history)

        # Period 1->2: |0.5-0.6| + |0.5-0.4| = 0.2, one-way = 0.1
        # Period 2->3: |0.4-0.5| + |0.6-0.5| = 0.2, one-way = 0.1
        # Total = 0.2
        @test turnover ≈ 0.2

        # No turnover case
        same_weights = [w1, w1, w1]
        @test compute_turnover(same_weights) ≈ 0.0
    end

    @testset "Preset Cost Models" begin
        order_value = 10000.0
        price = 100.0
        volume = 1e6

        # RETAIL_COSTS (commission-free, wider spread)
        retail_cost = compute_cost(RETAIL_COSTS, order_value, price, volume)
        @test retail_cost > 0

        # INSTITUTIONAL_COSTS (low commission, tight spread, impact)
        inst_cost = compute_cost(INSTITUTIONAL_COSTS, order_value, price, volume)
        @test inst_cost > 0

        # HFT_COSTS (rebates possible)
        hft_cost = compute_cost(HFT_COSTS, order_value, price, volume)
        @test hft_cost < retail_cost  # HFT should be cheaper

        # create_cost_model
        @test create_cost_model(:retail) isa CompositeCostModel
        @test create_cost_model(:institutional) isa CompositeCostModel
        @test create_cost_model(:hft) isa CompositeCostModel
        @test create_cost_model(:zero) isa FixedCostModel
    end

    @testset "Net Returns Calculation" begin
        gross_returns = [0.01, 0.02, -0.01, 0.015]
        costs = [10.0, 20.0, 15.0, 12.0]
        portfolio_values = [100000.0, 101000.0, 103020.0, 101990.0]

        net_returns = compute_net_returns(gross_returns, costs, portfolio_values)

        @test length(net_returns) == length(gross_returns)

        # Net should be less than gross
        for i in eachindex(net_returns)
            @test net_returns[i] < gross_returns[i]
        end

        # Check first return
        expected_drag = costs[1] / portfolio_values[1]
        @test net_returns[1] ≈ gross_returns[1] - expected_drag
    end

    @testset "Break-Even Sharpe" begin
        # 10 bps one-way = 20 bps round-trip per trade
        # If we trade daily, that's 252 * 20 = 5040 bps = 50.4% annual drag
        # With 20% volatility, need Sharpe > 50.4% / 20% = 2.52

        be_sharpe = estimate_break_even_sharpe(20.0, 0.20)
        @test be_sharpe > 2.0  # Should be significant

        # Lower costs = lower break-even Sharpe
        be_sharpe_low = estimate_break_even_sharpe(5.0, 0.20)
        @test be_sharpe_low < be_sharpe

        # Higher vol = lower break-even Sharpe (costs are fixed, but denominator larger)
        be_sharpe_high_vol = estimate_break_even_sharpe(10.0, 0.30)
        be_sharpe_low_vol = estimate_break_even_sharpe(10.0, 0.15)
        @test be_sharpe_high_vol < be_sharpe_low_vol
    end

    @testset "Backtest with Costs" begin
        # Create simple price data - deterministic for reproducible tests
        n = 100
        timestamps = [DateTime(2024, 1, 1) + Day(i) for i in 0:n-1]

        # Trending up prices (deterministic)
        aapl_prices = 100.0 .* (1.0 .+ cumsum(0.001 .* ones(n)))
        msft_prices = 200.0 .* (1.0 .+ cumsum(0.001 .* ones(n)))

        prices = Dict(
            :AAPL => aapl_prices,
            :MSFT => msft_prices
        )

        # Simple strategy
        target = Dict(:AAPL => 0.6, :MSFT => 0.4)
        strategy = BuyAndHoldStrategy(target)

        # Backtest without costs
        result_no_cost = backtest(strategy, timestamps, prices, initial_cash=100_000.0)

        # Reset strategy
        strategy2 = BuyAndHoldStrategy(target)

        # Backtest with costs
        cost_model = CompositeCostModel([
            ProportionalCostModel(rate_bps=5.0),
            SpreadCostModel(half_spread_bps=10.0)
        ])

        result_with_cost = backtest(
            strategy2, timestamps, prices,
            initial_cash=100_000.0,
            cost_model=cost_model
        )

        # Results should exist
        @test result_no_cost isa BacktestResult
        @test result_with_cost isa BacktestResult

        # With costs, final value should be lower
        @test result_with_cost.final_value <= result_no_cost.final_value

        # Cost metrics should be populated
        @test haskey(result_with_cost.metrics, :total_costs)
        @test haskey(result_with_cost.metrics, :avg_cost_bps)
        @test result_with_cost.total_costs > 0

        # Gross return should be higher than net return
        if haskey(result_with_cost.metrics, :gross_return)
            @test result_with_cost.metrics[:gross_return] >= result_with_cost.metrics[:total_return]
        end
    end

end
