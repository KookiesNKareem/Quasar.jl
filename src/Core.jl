module Core

# ============================================================================
# Abstract Type Hierarchy
# ============================================================================

"""
    AbstractInstrument

Root type for all financial instruments.
"""
abstract type AbstractInstrument end

"""
    AbstractEquity <: AbstractInstrument

Abstract type for equity instruments (stocks, ETFs).
"""
abstract type AbstractEquity <: AbstractInstrument end

"""
    AbstractDerivative <: AbstractInstrument

Abstract type for derivative instruments.
"""
abstract type AbstractDerivative <: AbstractInstrument end

"""
    AbstractOption <: AbstractDerivative

Abstract type for option contracts.
"""
abstract type AbstractOption <: AbstractDerivative end

"""
    AbstractFuture <: AbstractDerivative

Abstract type for futures contracts.
"""
abstract type AbstractFuture <: AbstractDerivative end

"""
    AbstractPortfolio

Abstract type for portfolio containers.
"""
abstract type AbstractPortfolio end

"""
    AbstractRiskMeasure

Abstract type for risk measures (VaR, CVaR, etc.).
"""
abstract type AbstractRiskMeasure end

"""
    ADBackend

Abstract type for automatic differentiation backends.
"""
abstract type ADBackend end

# ============================================================================
# Traits (Holy Traits Pattern)
# ============================================================================

# Trait types
abstract type Priceable end
struct IsPriceable <: Priceable end
struct NotPriceable <: Priceable end

abstract type Differentiable end
struct IsDifferentiable <: Differentiable end
struct NotDifferentiable <: Differentiable end

abstract type HasGreeks end
struct HasGreeksTrait <: HasGreeks end
struct NoGreeksTrait <: HasGreeks end

abstract type Simulatable end
struct IsSimulatable <: Simulatable end
struct NotSimulatable <: Simulatable end

# Trait query functions - default to negative
priceable(::Type{<:Any}) = NotPriceable()
priceable(x) = priceable(typeof(x))
ispriceable(x) = priceable(x) isa IsPriceable

differentiable(::Type{<:Any}) = NotDifferentiable()
differentiable(x) = differentiable(typeof(x))
isdifferentiable(x) = differentiable(x) isa IsDifferentiable

greeks_trait(::Type{<:Any}) = NoGreeksTrait()
greeks_trait(x) = greeks_trait(typeof(x))
hasgreeks(x) = greeks_trait(x) isa HasGreeksTrait

simulatable(::Type{<:Any}) = NotSimulatable()
simulatable(x) = simulatable(typeof(x))
issimulatable(x) = simulatable(x) isa IsSimulatable

# ============================================================================
# Core Types
# ============================================================================

# Simple immutable dict wrapper
struct ImmutableDict{K,V} <: AbstractDict{K,V}
    data::Dict{K,V}

    # Inner constructor that copies to ensure immutability
    function ImmutableDict{K,V}(d::Dict{K,V}) where {K,V}
        new{K,V}(copy(d))
    end
end
# Outer constructor for type inference
ImmutableDict(d::Dict{K,V}) where {K,V} = ImmutableDict{K,V}(d)

@inline Base.getindex(d::ImmutableDict, k) = d.data[k]
@inline Base.haskey(d::ImmutableDict, k) = haskey(d.data, k)
Base.keys(d::ImmutableDict) = keys(d.data)
Base.values(d::ImmutableDict) = values(d.data)
Base.length(d::ImmutableDict) = length(d.data)
Base.iterate(d::ImmutableDict, args...) = iterate(d.data, args...)
Base.setindex!(::ImmutableDict, v, k) = throw(MethodError(setindex!, (ImmutableDict, v, k)))

"""
    MarketState{P,R,V,T}

Immutable snapshot of market conditions.

# Fields
- `prices::P` - Current prices by symbol
- `rates::R` - Interest rates by currency
- `volatilities::V` - Implied volatilities by symbol
- `timestamp::T` - Time of snapshot
"""
struct MarketState{P,R,V,T}
    prices::P
    rates::R
    volatilities::V
    timestamp::T
end

# Keyword constructor for convenience with validation
function MarketState(; prices, rates, volatilities, timestamp)
    # Validate prices > 0
    for (sym, p) in prices
        p > 0 || throw(ArgumentError("Price for $sym must be positive, got $p"))
        isfinite(p) || throw(ArgumentError("Price for $sym must be finite, got $p"))
    end

    # Validate rates are reasonable (-1 to 1, i.e., -100% to 100%)
    for (ccy, r) in rates
        -1 <= r <= 1 || @warn "Rate for $ccy seems unusual: $r (expected between -100% and 100%)"
        isfinite(r) || throw(ArgumentError("Rate for $ccy must be finite, got $r"))
    end

    # Validate volatilities > 0
    for (sym, v) in volatilities
        v > 0 || throw(ArgumentError("Volatility for $sym must be positive, got $v"))
        isfinite(v) || throw(ArgumentError("Volatility for $sym must be finite, got $v"))
    end

    # Convert to immutable dictionaries for safety
    MarketState(
        ImmutableDict(prices),
        ImmutableDict(rates),
        ImmutableDict(volatilities),
        timestamp
    )
end

# Pretty printing
function Base.show(io::IO, state::MarketState)
    print(io, "MarketState(")
    print(io, length(state.prices), " prices, ")
    print(io, length(state.rates), " rates, ")
    print(io, length(state.volatilities), " vols, ")
    print(io, "t=", state.timestamp, ")")
end

function Base.show(io::IO, ::MIME"text/plain", state::MarketState)
    println(io, "MarketState @ ", state.timestamp)
    println(io, "  Prices:")
    for (k, v) in state.prices
        println(io, "    $k: $v")
    end
    println(io, "  Rates:")
    for (k, v) in state.rates
        println(io, "    $k: $(round(v * 100, digits=2))%")
    end
    println(io, "  Volatilities:")
    for (k, v) in state.volatilities
        println(io, "    $k: $(round(v * 100, digits=2))%")
    end
end

# ============================================================================
# Exports
# ============================================================================

export AbstractInstrument, AbstractEquity, AbstractDerivative, AbstractOption, AbstractFuture
export AbstractPortfolio, AbstractRiskMeasure, ADBackend
export MarketState, ImmutableDict
export Priceable, IsPriceable, NotPriceable
export Differentiable, IsDifferentiable, NotDifferentiable
export HasGreeks, HasGreeksTrait, NoGreeksTrait
export Simulatable, IsSimulatable, NotSimulatable
export priceable, ispriceable, differentiable, isdifferentiable
export greeks_trait, hasgreeks, simulatable, issimulatable

end
