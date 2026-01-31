module Quasar

# TODO: Add precompilation workloads for faster load times
# TODO: Consider SnoopCompile for latency analysis

# Core must come first - defines abstract types
include("Core.jl")
using .Core
export AbstractInstrument, AbstractEquity, AbstractDerivative, AbstractOption, AbstractFuture
export AbstractPortfolio, AbstractRiskMeasure, ADBackend
export MarketState
export Priceable, IsPriceable, NotPriceable
export Differentiable, IsDifferentiable, NotDifferentiable
export HasGreeks, HasGreeksTrait, NoGreeksTrait
export Simulatable, IsSimulatable, NotSimulatable
export priceable, ispriceable, differentiable, isdifferentiable
export greeks_trait, hasgreeks, simulatable, issimulatable

# AD backend system
include("AD.jl")
using .AD
export PureJuliaBackend, ForwardDiffBackend, ReactantBackend, EnzymeBackend
export gradient, hessian, jacobian, value_and_gradient, current_backend, set_backend!, with_backend, enable_gpu!

# Instruments
include("Instruments.jl")
using .Instruments
export Stock, EuropeanOption, AmericanOption, AsianOption, price, black_scholes
export american_binomial, asian_geometric, asian_arithmetic_approx
export GreeksResult, compute_greeks

# Portfolio
include("Portfolio.jl")
using .PortfolioModule
export Portfolio, value, portfolio_greeks

# Risk measures
include("Risk.jl")
using .Risk
export VaR, CVaR, Volatility, Sharpe, MaxDrawdown, compute

# Optimization
include("Optimization.jl")
using .Optimization
export optimize, MeanVariance, SharpeMaximizer, CVaRObjective, KellyCriterion, OptimizationResult

# Stochastic volatility models (SABR, Heston)
include("Models.jl")
using .Models
export SABRParams, sabr_implied_vol, sabr_price, black76
export HestonParams, heston_price, heston_characteristic

# Batch pricing and GPU-optimized calibration
include("BatchPricing.jl")
using .BatchPricing
export sabr_vols_batch, sabr_prices_batch
export PrecompiledSABRCalibrator, compile_gpu!, calibrate!
export price_surface_batch

# Model calibration
include("Calibration.jl")
using .Calibration
export OptionQuote, SmileData, VolSurface, CalibrationResult
export calibrate_sabr, calibrate_heston

# Monte Carlo simulation
include("MonteCarlo.jl")
using .MonteCarlo
export GBMDynamics, HestonDynamics
export EuropeanCall, EuropeanPut, AsianCall, AsianPut, UpAndOutCall, DownAndOutPut
export AmericanPut, AmericanCall
export MCResult, mc_price, mc_delta, mc_greeks, lsm_price
export mc_price_qmc, sobol_normals, simulate_gbm_qmc

# Market data
include("MarketData.jl")
using .MarketData
export AbstractMarketData, AbstractPriceHistory, AbstractDataAdapter
export PriceHistory, returns, resample, align
export CSVAdapter, ParquetAdapter, YAHOO_ADAPTER
import .MarketData: load, save

# Interest rates
include("InterestRates.jl")
using .InterestRates
# TODO: Export black_swaption (Swaption type exists but black_swaption not exported)
export RateCurve, DiscountCurve, ZeroCurve, ForwardCurve
export discount, zero_rate, forward_rate, instantaneous_forward
export LinearInterp, LogLinearInterp, CubicSplineInterp
export DepositRate, FuturesRate, SwapRate, bootstrap
export Bond, ZeroCouponBond, FixedRateBond, FloatingRateBond
export yield_to_maturity, duration, modified_duration, convexity, dv01
export accrued_interest, clean_price, dirty_price
export ShortRateModel, Vasicek, CIR, HullWhite
export bond_price, short_rate, simulate_short_rate
export Caplet, Floorlet, Cap, Floor, Swaption
export black_caplet, black_floorlet, black_cap, black_floor

end
