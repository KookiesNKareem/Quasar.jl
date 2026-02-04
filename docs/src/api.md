# API Reference

```@meta
CurrentModule = QuantNova
```

## Pricing

```@docs
black_scholes
price
```

## Greeks

```@docs
compute_greeks
GreeksResult
```

## Instruments

```@docs
Stock
EuropeanOption
Portfolio
value
portfolio_greeks
MarketState
```

## AD System

```@docs
gradient
hessian
jacobian
value_and_gradient
current_backend
set_backend!
with_backend
```

## Backends

```@docs
ForwardDiffBackend
PureJuliaBackend
EnzymeBackend
ReactantBackend
```

## Monte Carlo

```@docs
GBMDynamics
HestonDynamics
mc_price
mc_price_qmc
mc_delta
mc_greeks
lsm_price
```

## Payoffs

```@docs
EuropeanCall
EuropeanPut
AsianCall
AsianPut
UpAndOutCall
DownAndOutPut
AmericanPut
AmericanCall
```

## Risk Measures

```@docs
VaR
CVaR
Volatility
Sharpe
MaxDrawdown
compute
```

## Optimization

```@docs
MeanVariance
SharpeMaximizer
optimize
OptimizationResult
```

!!! note "Planned Features"
    `CVaRObjective` and `KellyCriterion` types are defined but `optimize()` methods are not yet implemented.

## Stochastic Volatility Models

```@docs
SABRParams
sabr_implied_vol
sabr_price
HestonParams
heston_price
```

## Calibration

```@docs
OptionQuote
SmileData
VolSurface
calibrate_sabr
calibrate_heston
CalibrationResult
```

## Interest Rates

### Yield Curves

```@docs
RateCurve
DiscountCurve
ZeroCurve
ForwardCurve
NelsonSiegelCurve
SvenssonCurve
discount
zero_rate
forward_rate
instantaneous_forward
fit_nelson_siegel
fit_svensson
```

### Interpolation

```@docs
LinearInterp
LogLinearInterp
CubicSplineInterp
```

### Bootstrapping

```@docs
DepositRate
FuturesRate
SwapRate
bootstrap
```

### Bonds

```@docs
Bond
ZeroCouponBond
FixedRateBond
FloatingRateBond
yield_to_maturity
duration
modified_duration
convexity
dv01
accrued_interest
clean_price
dirty_price
```

### Short-Rate Models

```@docs
ShortRateModel
Vasicek
CIR
HullWhite
bond_price
short_rate
simulate_short_rate
```

### Interest Rate Derivatives

```@docs
Caplet
Floorlet
Cap
Floor
Swaption
black_caplet
black_floorlet
black_cap
black_floor
```

## Market Data

```@docs
AbstractMarketData
AbstractPriceHistory
AbstractDataAdapter
PriceHistory
returns
resample
align
fetch_prices
fetch_multiple
fetch_returns
fetch_return_matrix
to_backtest_format
```

## Simulation

```@docs
SimulationState
MarketSnapshot
AbstractDriver
HistoricalDriver
Order
Fill
AbstractExecutionModel
InstantFill
SlippageModel
MarketImpactModel
execute
SimulationResult
simulate
```

## Backtesting

```@docs
AbstractStrategy
BuyAndHoldStrategy
RebalancingStrategy
VolatilityTargetStrategy
SignalStrategy
MomentumStrategy
MeanReversionStrategy
CompositeStrategy
generate_orders
should_rebalance
BacktestResult
backtest
compute_backtest_metrics
WalkForwardConfig
WalkForwardPeriod
WalkForwardResult
walk_forward_backtest
compute_extended_metrics
```

## Transaction Costs

```@docs
AbstractCostModel
FixedCostModel
ProportionalCostModel
TieredCostModel
SpreadCostModel
AlmgrenChrissModel
CompositeCostModel
CostAwareExecutionModel
compute_cost
execute_with_costs
TradeCostBreakdown
CostTracker
record_trade!
cost_summary
compute_turnover
compute_net_returns
estimate_break_even_sharpe
```

## Scenario Analysis

```@docs
StressScenario
ScenarioImpact
CRISIS_SCENARIOS
apply_scenario
scenario_impact
compare_scenarios
worst_case_scenario
SensitivityResult
sensitivity_analysis
ProjectionResult
monte_carlo_projection
```

## Factor Models

```@docs
RegressionResult
factor_regression
capm_regression
FamaFrenchResult
fama_french_regression
construct_market_factor
construct_long_short_factor
AttributionResult
return_attribution
rolling_beta
rolling_alpha
StyleAnalysisResult
style_analysis
tracking_error
information_ratio
up_capture_ratio
down_capture_ratio
capture_ratio
```

## Visualization

```@docs
AbstractVisualization
VisualizationSpec
LinkedContext
visualize
available_views
set_theme!
get_theme
LIGHT_THEME
DARK_THEME
COLORS
render
Row
Dashboard
serve
save
```

## Statistics

```@docs
sharpe_ratio
sharpe_std_error
sharpe_confidence_interval
sharpe_t_stat
sharpe_pvalue
probabilistic_sharpe_ratio
deflated_sharpe_ratio
compare_sharpe_ratios
minimum_backtest_length
probability_of_backtest_overfitting
combinatorial_purged_cv_pbo
information_coefficient
hit_rate
hit_rate_significance
```
