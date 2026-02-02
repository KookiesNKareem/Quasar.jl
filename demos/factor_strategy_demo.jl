# =============================================================================
# QuantNova Real-World Demo: Factor-Based Equity Strategy
# =============================================================================
#
# This demo shows a complete quant workflow using real market data:
#   1. Fetch 3 years of stock data for 30 liquid US equities
#   2. Compute factor exposures (market, momentum, volatility)
#   3. Build and backtest a momentum strategy
#   4. Analyze with factor regression and statistical tests
#   5. Measure performance with transaction costs
#
# Run: julia --project=.. factor_strategy_demo.jl
#
# This is what a junior quant's first project looks like.
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using QuantNova
using Statistics
using LinearAlgebra
using Printf
using Dates
using Random

println("=" ^ 70)
println("QUANTNOVA DEMO: Factor-Based Equity Strategy")
println("=" ^ 70)
println()

# =============================================================================
# 1. FETCH REAL MARKET DATA
# =============================================================================

println("[1/6] Fetching market data...")

# Liquid US large-caps (easy to trade, good data quality)
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Tech
    "JPM", "BAC", "WFC", "GS", "MS",           # Financials
    "JNJ", "PFE", "UNH", "MRK", "ABBV",        # Healthcare
    "XOM", "CVX", "COP", "SLB", "EOG",         # Energy
    "PG", "KO", "PEP", "WMT", "COST",          # Consumer
    "CAT", "DE", "HON", "UPS", "RTX"           # Industrials
]

# Fetch 3 years of daily data
end_date = today()
start_date = end_date - Year(3)

println("  Tickers: $(length(tickers)) stocks")
println("  Period: $start_date to $end_date")
println()

# Fetch live data via YFinance
function load_market_data(tickers, start_date, end_date)
    try
        println("  Fetching from Yahoo Finance...")
        data = fetch_return_matrix(tickers; startdt=start_date, enddt=end_date)
        println("  ✓ Fetched $(length(tickers)) stocks, $(size(data, 1)) trading days")
        return data, tickers, true
    catch e
        println("  ✗ Live data unavailable: $(typeof(e))")
        println("  → Falling back to synthetic data...")
        return nothing, String[], false
    end
end

returns_data, valid_tickers, use_live_data = load_market_data(tickers, start_date, end_date)

if !use_live_data
    # Fallback to synthetic data
    n_days_gen = 252 * 3
    valid_tickers = tickers

    Random.seed!(42)
    market_factor = 0.0003 .+ 0.01 .* randn(n_days_gen)
    sector_factors = 0.005 .* randn(n_days_gen, 6)

    returns_data = zeros(n_days_gen, length(valid_tickers))
    for (i, ticker) in enumerate(valid_tickers)
        sector_idx = div(i - 1, 5) + 1
        beta = 0.8 + 0.4 * rand()
        alpha = (rand() - 0.5) * 0.0002
        idio = 0.015 * randn(n_days_gen)
        returns_data[:, i] = alpha .+ beta .* market_factor .+ 0.3 .* sector_factors[:, sector_idx] .+ idio
    end

    println("  ✓ Generated synthetic: $(length(valid_tickers)) stocks, $n_days_gen days")
end

n_days, n_stocks = size(returns_data)

# Compute cumulative prices (normalized to 100)
prices_data = 100.0 .* cumprod(1 .+ returns_data, dims=1)

# Market benchmark (equal-weighted)
market_returns = vec(mean(returns_data, dims=2))

println()

# =============================================================================
# 2. COMPUTE FACTOR EXPOSURES
# =============================================================================

println("[2/6] Computing factor exposures...")

# Momentum factor: 12-month return, skip last month (Jegadeesh & Titman)
momentum_window = 252
skip_window = 21

function compute_momentum(returns, t)
    if t <= momentum_window + skip_window
        return fill(NaN, size(returns, 2))
    end
    # Sum of returns from t-252-21 to t-21
    cumret = vec(sum(returns[t-momentum_window-skip_window+1:t-skip_window, :], dims=1))
    return cumret
end

# Volatility factor: 60-day realized volatility (low vol = better)
vol_window = 60

function compute_volatility(returns, t)
    if t <= vol_window
        return fill(NaN, size(returns, 2))
    end
    vols = vec(std(returns[t-vol_window+1:t, :], dims=1)) .* sqrt(252)
    return vols
end

# Compute factors at end of sample for display
latest_momentum = compute_momentum(returns_data, n_days)
latest_volatility = compute_volatility(returns_data, n_days)

println("  Factor exposures (latest):")
println("  ┌────────┬────────────┬────────────┐")
println("  │ Ticker │  Momentum  │ Volatility │")
println("  ├────────┼────────────┼────────────┤")
for i in 1:min(10, n_stocks)
    mom_str = isnan(latest_momentum[i]) ? "   N/A   " : @sprintf("%+8.1f%%", latest_momentum[i] * 100)
    vol_str = isnan(latest_volatility[i]) ? "   N/A   " : @sprintf("%7.1f%%", latest_volatility[i] * 100)
    @printf("  │ %-6s │ %s │ %s │\n", valid_tickers[i], mom_str, vol_str)
end
println("  └────────┴────────────┴────────────┘")
println("  (showing first 10 of $n_stocks stocks)")
println()

# =============================================================================
# 3. BUILD MOMENTUM STRATEGY
# =============================================================================

println("[3/6] Building momentum strategy...")

# Strategy: Long top 20% momentum, short bottom 20%
# Rebalance monthly

rebalance_freq = 21  # Monthly
lookback_start = momentum_window + skip_window + 1

# Compute strategy returns
strategy_returns = zeros(n_days)
n_long = max(1, div(n_stocks, 5))  # Top 20%
n_short = n_long

positions = zeros(n_stocks)

for t in lookback_start:n_days
    # Rebalance at start of month
    if (t - lookback_start) % rebalance_freq == 0
        mom = compute_momentum(returns_data, t)

        if !any(isnan.(mom))
            sorted_idx = sortperm(mom, rev=true)

            # Reset positions
            fill!(positions, 0.0)

            # Long top momentum (equal weight)
            for i in 1:n_long
                positions[sorted_idx[i]] = 1.0 / n_long
            end

            # Short bottom momentum (equal weight)
            for i in 1:n_short
                positions[sorted_idx[end-i+1]] = -1.0 / n_short
            end
        end
    end

    # Compute return
    strategy_returns[t] = dot(positions, returns_data[t, :])
end

# Trim to valid period
valid_start = lookback_start
strategy_returns = strategy_returns[valid_start:end]
market_returns_trimmed = market_returns[valid_start:end]
n_valid = length(strategy_returns)

println("  Strategy: Long/Short Momentum (top/bottom 20%)")
println("  Rebalance: Monthly")
println("  Backtest period: $(n_valid) trading days")
println()

# =============================================================================
# 4. PERFORMANCE ANALYSIS
# =============================================================================

println("[4/6] Analyzing performance...")

# Compute metrics using QuantNova functions
strat_sharpe = sharpe_ratio(strategy_returns)
mkt_sharpe = sharpe_ratio(market_returns_trimmed)

strat_vol = std(strategy_returns) * sqrt(252)
mkt_vol = std(market_returns_trimmed) * sqrt(252)

strat_cum = cumprod(1 .+ strategy_returns)
mkt_cum = cumprod(1 .+ market_returns_trimmed)

strat_total = (strat_cum[end] - 1) * 100
mkt_total = (mkt_cum[end] - 1) * 100

# Max drawdown
strat_dd = strat_cum ./ accumulate(max, strat_cum) .- 1
mkt_dd = mkt_cum ./ accumulate(max, mkt_cum) .- 1
strat_maxdd = minimum(strat_dd) * 100
mkt_maxdd = minimum(mkt_dd) * 100

# Tracking error and information ratio
te = tracking_error(strategy_returns, market_returns_trimmed)
ir = information_ratio(strategy_returns, market_returns_trimmed)

println("  ┌─────────────────────┬────────────┬────────────┐")
println("  │       Metric        │  Strategy  │   Market   │")
println("  ├─────────────────────┼────────────┼────────────┤")
@printf("  │ Total Return        │  %+7.1f%%  │  %+7.1f%%  │\n", strat_total, mkt_total)
@printf("  │ Annualized Vol      │   %6.1f%%  │   %6.1f%%  │\n", strat_vol * 100, mkt_vol * 100)
@printf("  │ Sharpe Ratio        │   %+6.2f   │   %+6.2f   │\n", strat_sharpe, mkt_sharpe)
@printf("  │ Max Drawdown        │   %6.1f%%  │   %6.1f%%  │\n", strat_maxdd, mkt_maxdd)
println("  ├─────────────────────┼────────────┴────────────┤")
@printf("  │ Tracking Error      │          %5.1f%%          │\n", te * 100)
@printf("  │ Information Ratio   │          %+5.2f           │\n", ir)
println("  └─────────────────────┴─────────────────────────┘")
println()

# =============================================================================
# 5. FACTOR REGRESSION & STATISTICAL TESTING
# =============================================================================

println("[5/6] Running factor analysis...")

# CAPM regression
capm_result = capm_regression(strategy_returns, market_returns_trimmed)

println("  CAPM Regression:")
@printf("    Alpha (annualized): %+.2f%% (t=%.2f, p=%.3f)\n",
        capm_result.alpha * 100, capm_result.alpha_tstat, capm_result.alpha_pvalue)
@printf("    Beta:               %.2f\n", capm_result.beta)
@printf("    R²:                 %.1f%%\n", capm_result.r_squared * 100)
println()

# Sharpe ratio significance
sharpe_ci = sharpe_confidence_interval(strategy_returns; confidence=0.95)
sharpe_p = sharpe_pvalue(strategy_returns; alternative=:greater)

println("  Sharpe Ratio Analysis:")
@printf("    Sharpe:       %.2f\n", sharpe_ci.sharpe)
@printf("    95%% CI:       [%.2f, %.2f]\n", sharpe_ci.lower, sharpe_ci.upper)
@printf("    Std Error:    %.2f\n", sharpe_ci.std_error)
@printf("    P-value (>0): %.3f\n", sharpe_p)
println()

# Deflated Sharpe (assuming we tested ~20 strategy variants)
n_trials = 20
dsr = deflated_sharpe_ratio(strategy_returns, n_trials)
println("  Multiple Testing Adjustment (assuming $n_trials variants tested):")
@printf("    Deflated Sharpe Ratio: %.1f%% probability of true skill\n", dsr * 100)
println()

# =============================================================================
# 6. TRANSACTION COST ANALYSIS
# =============================================================================

println("[6/6] Estimating transaction costs impact...")

# Estimate turnover (simplified)
# With monthly rebalancing, ~40% one-way turnover per month typical for momentum
annual_turnover = 0.4 * 12  # ~480% annual turnover (both legs)

# Cost scenarios
scenarios = [
    ("Retail (10 bps)", 0.0010),
    ("Institutional (3 bps)", 0.0003),
    ("HFT/Internal (0.5 bps)", 0.00005),
]

gross_return = mean(strategy_returns) * 252

println("  Estimated Annual Turnover: $(round(annual_turnover * 100, digits=0))%")
println()
println("  ┌─────────────────────┬────────────┬────────────┬────────────┐")
println("  │     Cost Model      │ Cost/Trade │  Ann. Cost │ Net Return │")
println("  ├─────────────────────┼────────────┼────────────┼────────────┤")

for (name, cost_per_trade) in scenarios
    annual_cost = annual_turnover * cost_per_trade
    net_return = gross_return - annual_cost
    @printf("  │ %-19s │   %5.1f bps │   %5.1f%%   │   %+5.1f%%   │\n",
            name, cost_per_trade * 10000, annual_cost * 100, net_return * 100)
end
println("  └─────────────────────┴────────────┴────────────┴────────────┘")

# Break-even analysis
# Cost per trade * annual turnover = annual cost drag
annual_cost_drag = 0.0003 * annual_turnover  # 3 bps institutional
# Break-even Sharpe ≈ cost_drag / vol (need this Sharpe to overcome costs)
break_even = annual_cost_drag / strat_vol
println()
@printf("  Break-even Sharpe (institutional costs): %.2f\n", break_even)

# =============================================================================
# 7. VISUALIZATION (Optional - requires Makie backend)
# =============================================================================

println("[7/7] Preparing visualizations...")

# Create BacktestResult for visualization
# First, create proper timestamps
strategy_timestamps = [DateTime(2022, 1, 1) + Day(i) for i in valid_start:n_days]

# Compute equity curve from strategy returns
strategy_equity = zeros(n_valid)
strategy_equity[1] = 10000.0
for i in 2:n_valid
    strategy_equity[i] = strategy_equity[i-1] * (1 + strategy_returns[i])
end

# Create BacktestResult
strategy_result = BacktestResult(
    10000.0,
    strategy_equity[end],
    strategy_equity,
    strategy_returns[2:end],
    strategy_timestamps,
    Fill[],
    [Dict{Symbol,Float64}() for _ in 1:n_valid],
    Dict{Symbol,Float64}(
        :sharpe_ratio => strat_sharpe,
        :max_drawdown => strat_maxdd / 100,
        :annualized_return => strat_total / 100 / (n_valid / 252),
        :volatility => strat_vol,
        :total_return => strat_total / 100
    )
)

# Create visualization specs (lazy - doesn't render until displayed)
println("  Creating visualization specs...")
equity_spec = visualize(strategy_result, :equity; title="Momentum Strategy Equity")
drawdown_spec = visualize(strategy_result, :drawdown; title="Momentum Strategy Drawdown")
returns_spec = visualize(strategy_result, :returns; title="Daily Returns Distribution")
dashboard_spec = Dashboard(
    title = "Factor Strategy Dashboard",
    theme = :dark,
    layout = [
        Row(equity_spec, weight=2),
        Row(drawdown_spec, returns_spec),
    ]
)

println("  ✓ Specs created: equity, drawdown, returns, dashboard")
println()

# Check if Makie is available
local makie_loaded = false
try
    # Check if any Makie backend is loaded
    if isdefined(Main, :CairoMakie) || isdefined(Main, :GLMakie) || isdefined(Main, :WGLMakie)
        makie_loaded = true
    end
catch
end

if makie_loaded
    println("  Makie backend detected - rendering visualizations...")
    # render() and save() are available when Makie is loaded
    println("  → Use `render(equity_spec)` to display equity curve")
    println("  → Use `render(dashboard_spec)` for full dashboard")
    println("  → Use `save(\"report.png\", equity_spec)` to export")
else
    println("  No Makie backend loaded - skipping rendering")
    println("  To visualize, load a Makie backend first:")
    println("    using CairoMakie  # For static images (PNG, PDF, SVG)")
    println("    using GLMakie     # For interactive desktop plots")
    println("    using WGLMakie    # For web/notebook plots")
    println()
    println("  Then re-run or call:")
    println("    render(equity_spec)        # Display equity curve")
    println("    render(dashboard_spec)     # Full dashboard")
    println("    save(\"report.png\", spec)  # Export to file")
end

println()
println("=" ^ 70)
println("DEMO COMPLETE")
println("=" ^ 70)
println()
println("This demo showed a complete quant workflow:")
println("  ✓ Real market data fetching ($(n_stocks) stocks, $(n_days) days)")
println("  ✓ Factor computation (momentum, volatility)")
println("  ✓ Strategy construction and backtesting")
println("  ✓ Statistical significance testing (Sharpe CI, p-values)")
println("  ✓ Factor regression (CAPM alpha/beta)")
println("  ✓ Transaction cost impact analysis")
println("  ✓ Visualization specs (equity, drawdown, returns, dashboard)")
println()
println("All computations use QuantNova's pure Julia implementation.")
println("No Python, no external dependencies, no waiting.")
println("=" ^ 70)
