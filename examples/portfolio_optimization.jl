# Portfolio Optimization with Live Data
#
# This example demonstrates the complete workflow for portfolio optimization
# using live market data from Yahoo Finance.
#
# Run this example:
#   julia --project=. examples/portfolio_optimization.jl

using QuantNova
using Statistics
using Dates
using Printf
using LinearAlgebra

# ============================================================================
# Configuration
# ============================================================================

# Portfolio universe
const SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

# Backtest settings
const INITIAL_CAPITAL = 100_000.0
const DATA_RANGE = "2y"  # 2 years of data
const TRAIN_FRACTION = 0.5  # Use first half for optimization

# ============================================================================
# Helper Functions
# ============================================================================

function print_header(title)
    println()
    println("=" ^ 70)
    println(title)
    println("=" ^ 70)
end

function print_weights(symbols, weights)
    sorted = sort(collect(zip(symbols, weights)), by=x -> -x[2])
    for (sym, w) in sorted
        bar = repeat("█", round(Int, w * 40))
        @printf("  %-6s %5.1f%% %s\n", sym, w * 100, bar)
    end
end

function format_currency(x)
    # Format number with commas as thousands separator
    s = @sprintf("%.2f", x)
    parts = split(s, ".")
    int_part = parts[1]
    dec_part = parts[2]
    # Add commas
    n = length(int_part)
    result = ""
    for (i, c) in enumerate(int_part)
        if i > 1 && (n - i + 1) % 3 == 0
            result *= ","
        end
        result *= c
    end
    return "\$" * result * "." * dec_part
end

function print_metrics(result::BacktestResult)
    m = result.metrics
    println("  Final Value:       ", format_currency(result.final_value))
    @printf("  Total Return:      %+.2f%%\n", m[:total_return] * 100)
    @printf("  Annualized Return: %+.2f%%\n", m[:annualized_return] * 100)
    @printf("  Volatility:        %.2f%%\n", m[:volatility] * 100)
    @printf("  Sharpe Ratio:      %.3f\n", m[:sharpe_ratio])
    @printf("  Max Drawdown:      %.2f%%\n", m[:max_drawdown] * 100)
    @printf("  Calmar Ratio:      %.3f\n", m[:calmar_ratio])
    @printf("  Win Rate:          %.1f%%\n", m[:win_rate] * 100)
    @printf("  Total Trades:      %d\n", length(result.trades))
end

# ============================================================================
# Main Workflow
# ============================================================================

function main()
    print_header("PORTFOLIO OPTIMIZATION WITH LIVE DATA")
    println()
    println("Universe: ", join(SYMBOLS, ", "))
    println("Initial Capital: \$", INITIAL_CAPITAL)
    println()

    # -------------------------------------------------------------------------
    # Step 1: Fetch Historical Data
    # -------------------------------------------------------------------------
    print_header("Step 1: Fetching Market Data")

    println("Fetching $DATA_RANGE of daily data from Yahoo Finance...")
    histories = fetch_multiple(SYMBOLS, range=DATA_RANGE)

    n_days = length(histories[1])
    start_date = Date(histories[1].timestamps[1])
    end_date = Date(histories[1].timestamps[end])

    println("  Downloaded: $n_days trading days")
    println("  Date range: $start_date to $end_date")

    # -------------------------------------------------------------------------
    # Step 2: Train/Test Split
    # -------------------------------------------------------------------------
    print_header("Step 2: Train/Test Split")

    split_idx = round(Int, n_days * TRAIN_FRACTION)
    train_end = Date(histories[1].timestamps[split_idx])
    test_start = Date(histories[1].timestamps[split_idx + 1])

    println("  Training period: $start_date to $train_end ($split_idx days)")
    println("  Testing period:  $test_start to $end_date ($(n_days - split_idx) days)")

    # Compute training returns
    train_returns = Matrix{Float64}(undef, split_idx - 1, length(SYMBOLS))
    for (j, ph) in enumerate(histories)
        train_ph = PriceHistory(ph.symbol, ph.timestamps[1:split_idx], ph.close[1:split_idx])
        train_returns[:, j] = returns(train_ph)
    end

    # Prepare test data for backtesting
    test_timestamps = histories[1].timestamps[split_idx+1:end]
    test_prices = Dict{Symbol,Vector{Float64}}()
    for ph in histories
        test_prices[Symbol(ph.symbol)] = ph.close[split_idx+1:end]
    end

    # -------------------------------------------------------------------------
    # Step 3: Parameter Estimation
    # -------------------------------------------------------------------------
    print_header("Step 3: Parameter Estimation")

    μ = estimate_expected_returns(train_returns)
    Σ_sample = estimate_covariance(train_returns, SampleCovariance())
    Σ_lw = estimate_covariance(train_returns, LedoitWolfShrinkage())

    println("Expected Returns (annualized):")
    for (i, sym) in enumerate(SYMBOLS)
        @printf("  %-6s %+.1f%%\n", sym, μ[i] * 252 * 100)
    end
    println()
    println("Using Ledoit-Wolf shrinkage for covariance estimation")
    println("  (Shrinks sample covariance toward structured target for stability)")

    # -------------------------------------------------------------------------
    # Step 4: Portfolio Optimization
    # -------------------------------------------------------------------------
    print_header("Step 4: Portfolio Optimization")

    # Strategy 1: Minimum Variance
    println("\n[1] MINIMUM VARIANCE PORTFOLIO")
    mv_result = optimize(MinimumVariance(Σ_lw))
    println("  Converged: ", mv_result.converged)
    println("  Weights:")
    print_weights(SYMBOLS, mv_result.weights)

    # Strategy 2: Maximum Sharpe Ratio
    println("\n[2] MAXIMUM SHARPE RATIO PORTFOLIO")
    rf_daily = 0.05 / 252  # 5% annual risk-free rate
    sharpe_result = optimize(SharpeMaximizer(μ, Σ_lw, rf=rf_daily))
    println("  Converged: ", sharpe_result.converged)
    println("  Weights:")
    print_weights(SYMBOLS, sharpe_result.weights)

    # Strategy 3: Risk Parity
    println("\n[3] RISK PARITY PORTFOLIO")
    rp_result = optimize(RiskParity(Σ_lw))
    println("  Converged: ", rp_result.converged)
    println("  Weights:")
    print_weights(SYMBOLS, rp_result.weights)

    # Strategy 4: Equal Weight (benchmark)
    equal_weights = fill(1.0 / length(SYMBOLS), length(SYMBOLS))
    println("\n[4] EQUAL WEIGHT (Benchmark)")
    print_weights(SYMBOLS, equal_weights)

    # -------------------------------------------------------------------------
    # Step 5: Out-of-Sample Backtest
    # -------------------------------------------------------------------------
    print_header("Step 5: Out-of-Sample Backtest")

    strategies = [
        ("Minimum Variance", mv_result.weights),
        ("Max Sharpe", sharpe_result.weights),
        ("Risk Parity", rp_result.weights),
        ("Equal Weight", equal_weights),
    ]

    results = Dict{String, BacktestResult}()

    for (name, weights) in strategies
        target = Dict(Symbol(SYMBOLS[i]) => weights[i] for i in eachindex(SYMBOLS))
        strategy = BuyAndHoldStrategy(target)
        result = backtest(strategy, test_timestamps, test_prices, initial_cash=INITIAL_CAPITAL)
        results[name] = result
    end

    # Print comparison
    println("\nBuy-and-Hold Performance (Out-of-Sample):\n")
    println("-" ^ 70)
    @printf("%-18s %12s %10s %10s %10s\n", "Strategy", "Final Value", "Return", "Sharpe", "Max DD")
    println("-" ^ 70)

    for (name, _) in strategies
        r = results[name]
        m = r.metrics
        @printf("%-18s %12s %9.1f%% %10.3f %9.1f%%\n",
                name,
                format_currency(r.final_value),
                m[:total_return] * 100,
                m[:sharpe_ratio],
                m[:max_drawdown] * 100)
    end
    println("-" ^ 70)

    # -------------------------------------------------------------------------
    # Step 6: Rebalancing Comparison
    # -------------------------------------------------------------------------
    print_header("Step 6: Rebalancing Strategies")

    # Use minimum variance weights for rebalancing comparison
    mv_target = Dict(Symbol(SYMBOLS[i]) => mv_result.weights[i] for i in eachindex(SYMBOLS))

    # Monthly rebalancing
    monthly_strategy = RebalancingStrategy(
        target_weights=mv_target,
        rebalance_frequency=:monthly,
        tolerance=0.05
    )
    monthly_result = backtest(monthly_strategy, test_timestamps, test_prices, initial_cash=INITIAL_CAPITAL)

    println("\nMinimum Variance with Monthly Rebalancing (5% tolerance):\n")
    print_metrics(monthly_result)

    println("\nComparison: Buy-and-Hold vs Monthly Rebalancing")
    bh = results["Minimum Variance"]
    @printf("  Buy-and-Hold Return:  %+.2f%% (Sharpe: %.3f)\n",
            bh.metrics[:total_return] * 100, bh.metrics[:sharpe_ratio])
    @printf("  Monthly Rebalancing:  %+.2f%% (Sharpe: %.3f)\n",
            monthly_result.metrics[:total_return] * 100, monthly_result.metrics[:sharpe_ratio])

    # -------------------------------------------------------------------------
    # Step 7: Portfolio Analytics
    # -------------------------------------------------------------------------
    print_header("Step 7: Portfolio Analytics")

    println("\nRisk Contribution Analysis (Minimum Variance Portfolio):\n")
    risk_contrib = compute_risk_contributions(mv_result.weights, Σ_lw)
    total_risk = sqrt(mv_result.weights' * Σ_lw * mv_result.weights) * sqrt(252) * 100

    @printf("  Total Portfolio Volatility: %.2f%% annualized\n\n", total_risk)
    println("  Asset Risk Contributions:")
    for (i, sym) in enumerate(SYMBOLS)
        contrib_pct = risk_contrib[i] / sum(risk_contrib) * 100
        bar = repeat("█", round(Int, contrib_pct * 0.4))
        @printf("    %-6s %5.1f%% %s\n", sym, contrib_pct, bar)
    end

    # -------------------------------------------------------------------------
    # Step 8: Walk-Forward Analysis
    # -------------------------------------------------------------------------
    print_header("Step 8: Walk-Forward Backtesting")

    println("Running walk-forward with 6-month train / 1-month test windows...")
    println()

    # Full data for walk-forward
    full_timestamps, full_prices = to_backtest_format(histories)

    # Optimizer function for walk-forward
    function wf_optimizer(train_returns, syms)
        Σ = cov(train_returns)
        Σ = Σ + 1e-6 * I  # Regularization
        result = optimize(MinimumVariance(Σ))
        return Dict(Symbol(syms[i]) => result.weights[i] for i in eachindex(syms))
    end

    wf_config = WalkForwardConfig(
        train_period=126,   # 6 months
        test_period=21,     # 1 month
        expanding=false
    )

    wf_result = walk_forward_backtest(
        wf_optimizer,
        SYMBOLS,
        full_timestamps,
        full_prices,
        config=wf_config,
        initial_cash=INITIAL_CAPITAL
    )

    wf_m = wf_result.metrics
    println("Walk-Forward Results (", Int(wf_m[:n_periods]), " periods):")
    println()
    println("  Combined Performance:")
    @printf("    Total Return:       %+.2f%%\n", wf_m[:total_return] * 100)
    @printf("    Annualized Return:  %+.2f%%\n", wf_m[:annualized_return] * 100)
    @printf("    Volatility:         %.2f%%\n", wf_m[:volatility] * 100)
    @printf("    Sharpe Ratio:       %.3f\n", wf_m[:sharpe_ratio])
    @printf("    Max Drawdown:       %.2f%%\n", wf_m[:max_drawdown] * 100)
    println()
    println("  Period Statistics:")
    @printf("    Period Win Rate:    %.1f%%\n", wf_m[:period_win_rate] * 100)
    @printf("    Avg Period Return:  %+.2f%%\n", wf_m[:avg_period_return] * 100)
    @printf("    Period Return Std:  %.2f%%\n", wf_m[:period_return_std] * 100)

    # -------------------------------------------------------------------------
    # Step 9: Volatility Targeting
    # -------------------------------------------------------------------------
    print_header("Step 9: Volatility Targeting Comparison")

    # Use first half for comparison
    test_timestamps = histories[1].timestamps[split_idx+1:end]
    test_data = Dict{Symbol,Vector{Float64}}()
    for ph in histories
        test_data[Symbol(ph.symbol)] = ph.close[split_idx+1:end]
    end

    mv_target = Dict(Symbol(SYMBOLS[i]) => mv_result.weights[i] for i in eachindex(SYMBOLS))

    # Base strategy - monthly rebalancing
    base_strat = RebalancingStrategy(
        target_weights=mv_target,
        rebalance_frequency=:monthly,
        tolerance=0.05
    )
    base_result = backtest(base_strat, test_timestamps, test_data, initial_cash=INITIAL_CAPITAL)

    # Vol-targeted (15% target) with rebalancing
    vol_strat = VolatilityTargetStrategy(
        RebalancingStrategy(
            target_weights=mv_target,
            rebalance_frequency=:monthly,
            tolerance=0.05
        ),
        target_vol=0.15,
        max_leverage=1.0,
        min_leverage=0.3,
        lookback=20
    )
    vol_result = backtest(vol_strat, test_timestamps, test_data, initial_cash=INITIAL_CAPITAL)

    println("Comparison (Min-Variance weights):\n")
    println("-" ^ 50)
    @printf("%-20s %12s %12s\n", "Metric", "Base", "Vol Target")
    println("-" ^ 50)
    @printf("%-20s %11.2f%% %11.2f%%\n", "Return",
            base_result.metrics[:total_return] * 100,
            vol_result.metrics[:total_return] * 100)
    @printf("%-20s %11.2f%% %11.2f%%\n", "Realized Vol",
            base_result.metrics[:volatility] * 100,
            vol_result.metrics[:volatility] * 100)
    @printf("%-20s %12.3f %12.3f\n", "Sharpe Ratio",
            base_result.metrics[:sharpe_ratio],
            vol_result.metrics[:sharpe_ratio])
    @printf("%-20s %11.2f%% %11.2f%%\n", "Max Drawdown",
            base_result.metrics[:max_drawdown] * 100,
            vol_result.metrics[:max_drawdown] * 100)
    println("-" ^ 50)

    # -------------------------------------------------------------------------
    # Step 10: Visualization
    # -------------------------------------------------------------------------
    print_header("Step 10: Visualization")

    # Create visualization specs for all strategies
    println("Creating visualization specs...")

    viz_specs = Dict{String, Any}()
    for (name, result) in results
        viz_specs[name] = (
            equity = visualize(result, :equity; title="$name - Equity"),
            drawdown = visualize(result, :drawdown; title="$name - Drawdown"),
            returns = visualize(result, :returns; title="$name - Returns"),
        )
    end

    # Create comparison dashboard
    best_result = results["Max Sharpe"]
    mv_result_bt = results["Minimum Variance"]

    comparison_dashboard = Dashboard(
        title = "Portfolio Strategy Comparison",
        theme = :dark,
        layout = [
            Row(
                visualize(best_result, :equity; title="Max Sharpe Equity"),
                visualize(mv_result_bt, :equity; title="Min Variance Equity"),
            ),
            Row(
                visualize(best_result, :drawdown; title="Max Sharpe Drawdown"),
                visualize(mv_result_bt, :drawdown; title="Min Variance Drawdown"),
            ),
        ]
    )

    # Walk-forward dashboard
    wf_dashboard = Dashboard(
        title = "Walk-Forward Analysis",
        theme = :dark,
        layout = [
            Row(visualize(wf_result, :equity; title="Walk-Forward Equity"), weight=2),
            Row(
                visualize(wf_result, :drawdown; title="Walk-Forward Drawdown"),
                visualize(wf_result, :returns; title="Walk-Forward Returns"),
            ),
        ]
    )

    println("  ✓ Created specs for $(length(results)) strategies")
    println("  ✓ Created comparison dashboard")
    println("  ✓ Created walk-forward dashboard")
    println()

    # Check if Makie is available
    local makie_loaded = false
    try
        if isdefined(Main, :CairoMakie) || isdefined(Main, :GLMakie) || isdefined(Main, :WGLMakie)
            makie_loaded = true
        end
    catch
    end

    if makie_loaded
        println("  Makie backend detected!")
        println("  Available visualizations:")
        println("    render(viz_specs[\"Max Sharpe\"].equity)  # Single equity curve")
        println("    render(comparison_dashboard)             # Strategy comparison")
        println("    render(wf_dashboard)                     # Walk-forward analysis")
        println("    save(\"portfolio_report.png\", viz_specs[\"Max Sharpe\"].equity)")
    else
        println("  No Makie backend loaded - specs ready but not rendered")
        println("  To visualize, load a Makie backend:")
        println("    using CairoMakie  # For static exports (recommended)")
        println("    using GLMakie     # For interactive plots")
        println("    using WGLMakie    # For web/notebooks")
        println()
        println("  Then call:")
        println("    render(viz_specs[\"Max Sharpe\"].equity)")
        println("    render(comparison_dashboard)")
        println("    save(\"portfolio_report.pdf\", comparison_dashboard)")
    end

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_header("Summary")

    # Find best strategy by Sharpe ratio
    best_strategy = ""
    best_sharpe = -Inf
    for (name, result) in results
        if result.metrics[:sharpe_ratio] > best_sharpe
            best_sharpe = result.metrics[:sharpe_ratio]
            best_strategy = name
        end
    end
    best = results[best_strategy]

    println()
    println("Best performing strategy (by Sharpe ratio): $best_strategy")
    println()
    print_metrics(best)

    println()
    println("Visualization specs available:")
    println("  viz_specs       - Dict of specs per strategy")
    println("  comparison_dashboard - Side-by-side strategy comparison")
    println("  wf_dashboard    - Walk-forward analysis dashboard")

    println()
    println("=" ^ 70)
    println("Example complete! See examples/portfolio_optimization.jl for source code.")
    println("=" ^ 70)

    # Return specs for interactive use
    return (
        results = results,
        viz_specs = viz_specs,
        comparison_dashboard = comparison_dashboard,
        wf_dashboard = wf_dashboard,
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
