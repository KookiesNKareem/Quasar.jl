module TransactionCosts

using Statistics: mean, std

# =============================================================================
# Transaction Cost Models
# =============================================================================

"""
    AbstractCostModel

Base type for transaction cost models.
"""
abstract type AbstractCostModel end

"""
    compute_cost(model, order_value, price, volume; kwargs...) -> Float64

Compute transaction cost for a trade.

# Arguments
- `order_value` - Absolute dollar value of the order
- `price` - Current price per share
- `volume` - Average daily volume (shares) - used for market impact
"""
function compute_cost end

# -----------------------------------------------------------------------------
# Fixed Cost Model
# -----------------------------------------------------------------------------

"""
    FixedCostModel <: AbstractCostModel

Flat fee per trade regardless of size.

# Fields
- `cost_per_trade::Float64` - Fixed cost per trade (e.g., \$5.00)

# Example
```julia
model = FixedCostModel(5.0)  # \$5 per trade
cost = compute_cost(model, 10000.0, 150.0, 1e6)  # Returns 5.0
```
"""
struct FixedCostModel <: AbstractCostModel
    cost_per_trade::Float64

    function FixedCostModel(cost_per_trade::Float64=0.0)
        cost_per_trade >= 0 || error("cost_per_trade must be non-negative")
        new(cost_per_trade)
    end
end

function compute_cost(model::FixedCostModel, order_value::Float64, price::Float64, volume::Float64)
    return model.cost_per_trade
end

# -----------------------------------------------------------------------------
# Proportional Cost Model
# -----------------------------------------------------------------------------

"""
    ProportionalCostModel <: AbstractCostModel

Commission as a percentage of trade value.

# Fields
- `rate_bps::Float64` - Commission rate in basis points (1 bp = 0.01%)
- `min_cost::Float64` - Minimum cost per trade

# Example
```julia
model = ProportionalCostModel(rate_bps=5.0, min_cost=1.0)  # 5 bps with \$1 minimum
cost = compute_cost(model, 10000.0, 150.0, 1e6)  # Returns 5.0 (0.05% of 10000)
```
"""
struct ProportionalCostModel <: AbstractCostModel
    rate_bps::Float64
    min_cost::Float64

    function ProportionalCostModel(; rate_bps::Float64=0.0, min_cost::Float64=0.0)
        # rate_bps can be negative for maker rebates
        min_cost >= 0 || error("min_cost must be non-negative")
        new(rate_bps, min_cost)
    end
end

function compute_cost(model::ProportionalCostModel, order_value::Float64, price::Float64, volume::Float64)
    cost = order_value * (model.rate_bps / 10000)
    return max(cost, model.min_cost)
end

# -----------------------------------------------------------------------------
# Tiered Cost Model
# -----------------------------------------------------------------------------

"""
    TieredCostModel <: AbstractCostModel

Volume-based tiered pricing (common for institutional traders).

# Fields
- `tiers::Vector{Tuple{Float64,Float64}}` - (threshold, rate_bps) pairs, sorted ascending
- `min_cost::Float64` - Minimum cost per trade

Tiers are cumulative: first tier applies up to first threshold, etc.

# Example
```julia
# 10 bps up to \$10k, 5 bps from \$10k-\$100k, 2 bps above \$100k
model = TieredCostModel([
    (10_000.0, 10.0),
    (100_000.0, 5.0),
    (Inf, 2.0)
])
```
"""
struct TieredCostModel <: AbstractCostModel
    tiers::Vector{Tuple{Float64,Float64}}  # (threshold, rate_bps)
    min_cost::Float64

    function TieredCostModel(tiers::Vector{Tuple{Float64,Float64}}; min_cost::Float64=0.0)
        isempty(tiers) && error("Must provide at least one tier")
        # Sort by threshold
        sorted = sort(tiers, by=x->x[1])
        new(sorted, min_cost)
    end
end

function compute_cost(model::TieredCostModel, order_value::Float64, price::Float64, volume::Float64)
    remaining = order_value
    total_cost = 0.0
    prev_threshold = 0.0

    for (threshold, rate_bps) in model.tiers
        tier_amount = min(remaining, threshold - prev_threshold)
        if tier_amount > 0
            total_cost += tier_amount * (rate_bps / 10000)
            remaining -= tier_amount
        end
        prev_threshold = threshold
        remaining <= 0 && break
    end

    return max(total_cost, model.min_cost)
end

# -----------------------------------------------------------------------------
# Spread Cost Model
# -----------------------------------------------------------------------------

"""
    SpreadCostModel <: AbstractCostModel

Bid-ask spread cost model.

# Fields
- `half_spread_bps::Float64` - Half the bid-ask spread in basis points
- `spread_estimator::Symbol` - How to estimate spread (:fixed, :volatility_based)
- `vol_multiplier::Float64` - For volatility-based: spread = vol * multiplier

# Example
```julia
# Fixed 5 bps half-spread (10 bps round-trip)
model = SpreadCostModel(half_spread_bps=5.0)

# Volatility-based spread estimation
model = SpreadCostModel(spread_estimator=:volatility_based, vol_multiplier=0.1)
```
"""
struct SpreadCostModel <: AbstractCostModel
    half_spread_bps::Float64
    spread_estimator::Symbol
    vol_multiplier::Float64

    function SpreadCostModel(;
        half_spread_bps::Float64=5.0,
        spread_estimator::Symbol=:fixed,
        vol_multiplier::Float64=0.1
    )
        spread_estimator in (:fixed, :volatility_based) ||
            error("spread_estimator must be :fixed or :volatility_based")
        new(half_spread_bps, spread_estimator, vol_multiplier)
    end
end

function compute_cost(model::SpreadCostModel, order_value::Float64, price::Float64, volume::Float64;
                      volatility::Float64=0.0)
    if model.spread_estimator == :volatility_based && volatility > 0
        # Spread proportional to volatility
        spread_bps = volatility * model.vol_multiplier * 10000
        half_spread = spread_bps / 2
    else
        half_spread = model.half_spread_bps
    end

    return order_value * (half_spread / 10000)
end

# -----------------------------------------------------------------------------
# Market Impact Model (Almgren-Chriss)
# -----------------------------------------------------------------------------

"""
    AlmgrenChrissModel <: AbstractCostModel

Square-root market impact model based on Almgren-Chriss framework.

Market impact = σ * √(order_shares / ADV) * price * order_shares

This is the industry-standard model for estimating permanent and temporary
price impact from trading.

# Fields
- `volatility::Float64` - Daily volatility (annualized σ / √252)
- `participation_rate::Float64` - Fraction of ADV you're willing to trade
- `temporary_impact::Float64` - Temporary impact coefficient (default: 0.1)
- `permanent_impact::Float64` - Permanent impact coefficient (default: 0.1)

# Example
```julia
model = AlmgrenChrissModel(
    volatility=0.02,           # 2% daily vol
    participation_rate=0.1,    # Trade up to 10% of ADV
    temporary_impact=0.1,
    permanent_impact=0.1
)
```

# Reference
Almgren, R., & Chriss, N. (2000). Optimal execution of portfolio transactions.
"""
struct AlmgrenChrissModel <: AbstractCostModel
    volatility::Float64
    participation_rate::Float64
    temporary_impact::Float64
    permanent_impact::Float64

    function AlmgrenChrissModel(;
        volatility::Float64=0.02,
        participation_rate::Float64=0.1,
        temporary_impact::Float64=0.1,
        permanent_impact::Float64=0.1
    )
        volatility > 0 || error("volatility must be positive")
        0 < participation_rate <= 1 || error("participation_rate must be in (0, 1]")
        new(volatility, participation_rate, temporary_impact, permanent_impact)
    end
end

function compute_cost(model::AlmgrenChrissModel, order_value::Float64, price::Float64, volume::Float64)
    order_shares = order_value / price
    adv_value = volume * price

    # Participation rate
    participation = order_value / adv_value

    # Square-root impact
    # Temporary impact: η * σ * √(shares/ADV)
    temp_impact = model.temporary_impact * model.volatility * sqrt(participation)

    # Permanent impact: γ * σ * (shares/ADV)
    perm_impact = model.permanent_impact * model.volatility * participation

    # Total cost as fraction of order value
    total_impact = temp_impact + 0.5 * perm_impact  # Half of permanent counts

    return order_value * total_impact
end

# -----------------------------------------------------------------------------
# Composite Cost Model
# -----------------------------------------------------------------------------

"""
    CompositeCostModel <: AbstractCostModel

Combines multiple cost models (e.g., commission + spread + market impact).

# Example
```julia
model = CompositeCostModel([
    ProportionalCostModel(rate_bps=1.0),      # 1 bp commission
    SpreadCostModel(half_spread_bps=5.0),     # 5 bp half-spread
    AlmgrenChrissModel(volatility=0.02)       # Market impact
])
```
"""
struct CompositeCostModel <: AbstractCostModel
    models::Vector{AbstractCostModel}

    function CompositeCostModel(models::Vector{<:AbstractCostModel})
        isempty(models) && error("Must provide at least one cost model")
        new(collect(AbstractCostModel, models))
    end
end

function compute_cost(model::CompositeCostModel, order_value::Float64, price::Float64, volume::Float64;
                      kwargs...)
    total = 0.0
    for m in model.models
        total += compute_cost(m, order_value, price, volume; kwargs...)
    end
    return total
end

# =============================================================================
# Cost Tracking & Attribution
# =============================================================================

"""
    TradeCostBreakdown

Detailed breakdown of costs for a single trade.
"""
struct TradeCostBreakdown
    symbol::Symbol
    order_value::Float64
    commission::Float64
    spread_cost::Float64
    market_impact::Float64
    total_cost::Float64
    cost_bps::Float64  # Total cost in basis points
end

"""
    CostTracker

Tracks transaction costs across a backtest.
"""
mutable struct CostTracker
    total_traded::Float64
    total_costs::Float64
    total_commission::Float64
    total_spread::Float64
    total_impact::Float64
    n_trades::Int
    costs_by_symbol::Dict{Symbol,Float64}
    trades::Vector{TradeCostBreakdown}

    function CostTracker()
        new(0.0, 0.0, 0.0, 0.0, 0.0, 0, Dict{Symbol,Float64}(), TradeCostBreakdown[])
    end
end

"""
    record_trade!(tracker, breakdown)

Record a trade's costs in the tracker.
"""
function record_trade!(tracker::CostTracker, breakdown::TradeCostBreakdown)
    tracker.total_traded += breakdown.order_value
    tracker.total_costs += breakdown.total_cost
    tracker.total_commission += breakdown.commission
    tracker.total_spread += breakdown.spread_cost
    tracker.total_impact += breakdown.market_impact
    tracker.n_trades += 1

    tracker.costs_by_symbol[breakdown.symbol] =
        get(tracker.costs_by_symbol, breakdown.symbol, 0.0) + breakdown.total_cost

    push!(tracker.trades, breakdown)
end

"""
    cost_summary(tracker) -> Dict{Symbol,Float64}

Get summary statistics from cost tracker.
"""
function cost_summary(tracker::CostTracker)
    avg_cost_bps = tracker.total_traded > 0 ?
        (tracker.total_costs / tracker.total_traded) * 10000 : 0.0

    Dict{Symbol,Float64}(
        :total_traded => tracker.total_traded,
        :total_costs => tracker.total_costs,
        :total_commission => tracker.total_commission,
        :total_spread => tracker.total_spread,
        :total_impact => tracker.total_impact,
        :n_trades => Float64(tracker.n_trades),
        :avg_cost_bps => avg_cost_bps,
        :avg_cost_per_trade => tracker.n_trades > 0 ? tracker.total_costs / tracker.n_trades : 0.0
    )
end

# =============================================================================
# Turnover Calculation
# =============================================================================

"""
    compute_turnover(weights_history::Vector{Dict{Symbol,Float64}}) -> Float64

Compute total portfolio turnover from a history of weights.

Turnover = Σ |w_t - w_{t-1}| / 2

Returns annualized turnover if daily data (multiply by 252).
"""
function compute_turnover(weights_history::Vector{Dict{Symbol,Float64}})
    length(weights_history) < 2 && return 0.0

    all_symbols = Set{Symbol}()
    for w in weights_history
        union!(all_symbols, keys(w))
    end

    total_turnover = 0.0
    for i in 2:length(weights_history)
        period_turnover = 0.0
        for sym in all_symbols
            w_prev = get(weights_history[i-1], sym, 0.0)
            w_curr = get(weights_history[i], sym, 0.0)
            period_turnover += abs(w_curr - w_prev)
        end
        total_turnover += period_turnover / 2  # One-way turnover
    end

    return total_turnover
end

"""
    compute_turnover(positions_history, prices_history) -> Float64

Compute turnover from position and price histories.
"""
function compute_turnover(
    positions_history::Vector{Dict{Symbol,Float64}},
    prices_history::Vector{Dict{Symbol,Float64}}
)
    length(positions_history) < 2 && return 0.0
    length(positions_history) != length(prices_history) &&
        error("positions and prices histories must have same length")

    # Convert to weights
    weights_history = Dict{Symbol,Float64}[]
    for (positions, prices) in zip(positions_history, prices_history)
        total_value = sum(get(positions, s, 0.0) * get(prices, s, 0.0)
                          for s in keys(positions))

        weights = Dict{Symbol,Float64}()
        if total_value > 0
            for (sym, qty) in positions
                price = get(prices, sym, 0.0)
                weights[sym] = (qty * price) / total_value
            end
        end
        push!(weights_history, weights)
    end

    compute_turnover(weights_history)
end

# =============================================================================
# Cost-Aware Execution
# =============================================================================

"""
    CostAwareExecutionModel

Execution model that tracks detailed transaction costs.
"""
struct CostAwareExecutionModel
    commission_model::AbstractCostModel
    spread_model::AbstractCostModel
    impact_model::Union{AbstractCostModel,Nothing}
    tracker::CostTracker
    adv::Dict{Symbol,Float64}  # Average daily volume by symbol

    function CostAwareExecutionModel(;
        commission_model::AbstractCostModel=ProportionalCostModel(rate_bps=1.0),
        spread_model::AbstractCostModel=SpreadCostModel(half_spread_bps=5.0),
        impact_model::Union{AbstractCostModel,Nothing}=nothing,
        adv::Dict{Symbol,Float64}=Dict{Symbol,Float64}()
    )
        new(commission_model, spread_model, impact_model, CostTracker(), adv)
    end
end

"""
    execute_with_costs(model, symbol, order_value, price) -> (exec_price, breakdown)

Execute a trade and return execution price with cost breakdown.
"""
function execute_with_costs(
    model::CostAwareExecutionModel,
    symbol::Symbol,
    order_value::Float64,
    price::Float64
)
    volume = get(model.adv, symbol, 1e6)  # Default 1M shares ADV

    commission = compute_cost(model.commission_model, order_value, price, volume)
    spread_cost = compute_cost(model.spread_model, order_value, price, volume)

    impact_cost = if model.impact_model !== nothing
        compute_cost(model.impact_model, order_value, price, volume)
    else
        0.0
    end

    total_cost = commission + spread_cost + impact_cost
    cost_bps = (total_cost / order_value) * 10000

    breakdown = TradeCostBreakdown(
        symbol, order_value, commission, spread_cost, impact_cost, total_cost, cost_bps
    )

    record_trade!(model.tracker, breakdown)

    # Execution price includes spread and impact
    slippage_pct = (spread_cost + impact_cost) / order_value
    exec_price = price * (1 + slippage_pct)

    return exec_price, breakdown
end

# =============================================================================
# Preset Cost Models
# =============================================================================

"""
    RETAIL_COSTS

Typical retail investor costs (commission-free but wide spreads).
"""
const RETAIL_COSTS = CompositeCostModel([
    FixedCostModel(0.0),                    # Commission-free
    SpreadCostModel(half_spread_bps=10.0)   # Wider spreads due to PFOF
])

"""
    INSTITUTIONAL_COSTS

Typical institutional costs (low commission, tighter spreads, some impact).
"""
const INSTITUTIONAL_COSTS = CompositeCostModel([
    ProportionalCostModel(rate_bps=1.0, min_cost=0.0),
    SpreadCostModel(half_spread_bps=2.5),
    AlmgrenChrissModel(volatility=0.015, temporary_impact=0.05)
])

"""
    HFT_COSTS

High-frequency trading costs (maker rebates, minimal spread).
"""
const HFT_COSTS = CompositeCostModel([
    ProportionalCostModel(rate_bps=-0.2, min_cost=0.0),  # Maker rebate
    SpreadCostModel(half_spread_bps=0.5)
])

"""
    create_cost_model(profile::Symbol) -> AbstractCostModel

Create a cost model from a preset profile.

Available profiles: :retail, :institutional, :hft, :zero
"""
function create_cost_model(profile::Symbol)
    if profile == :retail
        return RETAIL_COSTS
    elseif profile == :institutional
        return INSTITUTIONAL_COSTS
    elseif profile == :hft
        return HFT_COSTS
    elseif profile == :zero
        return FixedCostModel(0.0)
    else
        error("Unknown cost profile: $profile. Use :retail, :institutional, :hft, or :zero")
    end
end

# =============================================================================
# Net Returns Calculation
# =============================================================================

"""
    compute_net_returns(gross_returns, costs, portfolio_values) -> Vector{Float64}

Compute net returns after transaction costs.

# Arguments
- `gross_returns` - Returns before costs
- `costs` - Transaction costs per period
- `portfolio_values` - Portfolio value at start of each period
"""
function compute_net_returns(
    gross_returns::Vector{Float64},
    costs::Vector{Float64},
    portfolio_values::Vector{Float64}
)
    length(gross_returns) == length(costs) ||
        error("gross_returns and costs must have same length")

    net_returns = similar(gross_returns)
    for i in eachindex(gross_returns)
        cost_drag = costs[i] / portfolio_values[i]
        net_returns[i] = gross_returns[i] - cost_drag
    end
    return net_returns
end

"""
    estimate_break_even_sharpe(cost_bps, volatility; periods_per_year=252) -> Float64

Estimate minimum Sharpe ratio needed to break even after costs.

A strategy with Sharpe below this threshold will likely lose money after costs.
"""
function estimate_break_even_sharpe(cost_bps::Float64, volatility::Float64;
                                     periods_per_year::Int=252)
    # Annual cost drag as fraction
    annual_cost = cost_bps / 10000 * periods_per_year

    # Sharpe = Return / Vol, so Return = Sharpe * Vol
    # Need Return > Costs, so Sharpe * Vol > annual_cost
    # Therefore Sharpe > annual_cost / Vol

    return annual_cost / volatility
end

# =============================================================================
# Exports
# =============================================================================

export AbstractCostModel, compute_cost
export FixedCostModel, ProportionalCostModel, TieredCostModel
export SpreadCostModel, AlmgrenChrissModel, CompositeCostModel
export TradeCostBreakdown, CostTracker, record_trade!, cost_summary
export compute_turnover
export CostAwareExecutionModel, execute_with_costs
export RETAIL_COSTS, INSTITUTIONAL_COSTS, HFT_COSTS, create_cost_model
export compute_net_returns, estimate_break_even_sharpe

end
