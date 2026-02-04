module InterestRates

using LinearAlgebra
using Dates
using Distributions: Normal, cdf, pdf

export
    # Day count conventions
    DayCountConvention, ACT360, ACT365, Thirty360, ACTACT,
    year_fraction,
    # Scheduling and conventions
    Calendar, WeekendCalendar,
    BusinessDayConvention, Following, ModifiedFollowing, Preceding, ModifiedPreceding,
    RollRule, NoRoll, EndOfMonth,
    StubRule, StubNone, ShortFront, ShortBack,
    Schedule, schedule_periods, accrual_factors, adjust_date, is_business_day,
    # Multi-curve framework
    RateIndex, CurveSet, fixing_date, payment_date,
    # Curve types
    RateCurve, DiscountCurve, ZeroCurve, ForwardCurve,
    # Parametric curves
    NelsonSiegelCurve, SvenssonCurve, fit_nelson_siegel, fit_svensson,
    # Curve operations
    discount, zero_rate, forward_rate, instantaneous_forward,
    # Interpolation
    LinearInterp, LogLinearInterp, CubicSplineInterp,
    # Bootstrapping
    DepositRate, FuturesRate, SwapRate, bootstrap,
    OISDepositQuote, OISSwapQuote, FRAQuote, IRSwapQuote,
    bootstrap_ois_curve, bootstrap_forward_curve, bootstrap_curveset,
    # Bonds
    Bond, ZeroCouponBond, FixedRateBond, FloatingRateBond,
    # Note: price() not exported to avoid collision with Instruments.price
    # Use QuantNova.InterestRates.price() for bond pricing
    yield_to_maturity, duration, modified_duration, convexity, dv01,
    accrued_interest, clean_price, dirty_price,
    # Short rate models
    ShortRateModel, Vasicek, CIR, HullWhite,
    bond_price, short_rate, simulate_short_rate,
    # IR Derivatives
    FRA, FixedLeg, FloatLeg, Swap, par_swap_rate,
    Caplet, Floorlet, Cap, Floor, Swaption,
    black_caplet, black_cap,
    cashflows,
    # Curve risk
    bucketed_pv01

# ============================================================================
# Day Count Conventions
# ============================================================================

"""
    DayCountConvention

Abstract type for day count conventions used to calculate year fractions
between two dates.
"""
abstract type DayCountConvention end

"""
    ACT360 <: DayCountConvention

Actual/360 day count: actual days divided by 360.
Common for money market instruments (LIBOR, EURIBOR).
"""
struct ACT360 <: DayCountConvention end

"""
    ACT365 <: DayCountConvention

Actual/365 (Fixed) day count: actual days divided by 365.
Common for GBP and JPY markets.
"""
struct ACT365 <: DayCountConvention end

"""
    Thirty360 <: DayCountConvention

30/360 (Bond Basis) day count: assumes 30-day months and 360-day years.
Common for US corporate bonds.
"""
struct Thirty360 <: DayCountConvention end

"""
    ACTACT <: DayCountConvention

Actual/Actual (ISDA) day count: actual days in each period.
Standard for government bonds in many markets.
"""
struct ACTACT <: DayCountConvention end

"""
    year_fraction(start_date, end_date, convention::DayCountConvention) -> Float64

Calculate the year fraction between two dates using the specified day count convention.

# Arguments
- `start_date` - Start date (as Date or day count from reference)
- `end_date` - End date (as Date or day count from reference)
- `convention` - Day count convention to use

# Example
```julia
using Dates
year_fraction(Date(2024,1,1), Date(2024,7,1), ACT360())  # ≈ 0.5028
year_fraction(Date(2024,1,1), Date(2024,7,1), ACT365())  # ≈ 0.4959
year_fraction(Date(2024,1,1), Date(2024,7,1), Thirty360())  # = 0.5
```
"""
function year_fraction end

# For simple numeric time (already in years), just return the difference
# These take priority over the date-based implementations when inputs are numbers
year_fraction(t1::Real, t2::Real, ::ACT360) = Float64(t2 - t1)
year_fraction(t1::Real, t2::Real, ::ACT365) = Float64(t2 - t1)
year_fraction(t1::Real, t2::Real, ::Thirty360) = Float64(t2 - t1)
year_fraction(t1::Real, t2::Real, ::ACTACT) = Float64(t2 - t1)

# Date-based implementations
year_fraction(d1::Date, d2::Date, ::ACT360) = _year_fraction_act360(d1, d2)
year_fraction(d1::Date, d2::Date, ::ACT365) = _year_fraction_act365(d1, d2)
year_fraction(d1::Date, d2::Date, ::Thirty360) = _year_fraction_30360(d1, d2)
year_fraction(d1::Date, d2::Date, ::ACTACT) = _year_fraction_actact(d1, d2)

year_fraction(d1::DateTime, d2::DateTime, conv::DayCountConvention) =
    year_fraction(Date(d1), Date(d2), conv)

# These are for actual Date objects, not numeric types
function _year_fraction_act360(d1, d2)
    days = _day_count(d1, d2)
    days / 360.0
end

function _year_fraction_act365(d1, d2)
    days = _day_count(d1, d2)
    days / 365.0
end

function _year_fraction_30360(d1, d2)
    y1, m1, day1 = _year_month_day(d1)
    y2, m2, day2 = _year_month_day(d2)

    # Adjust day counts per 30/360 convention
    day1 = min(day1, 30)
    if day1 == 30
        day2 = min(day2, 30)
    end

    (360 * (y2 - y1) + 30 * (m2 - m1) + (day2 - day1)) / 360.0
end

function _year_fraction_actact(d1, d2)
    # Simplified ISDA implementation: actual days / average year length
    days = _day_count(d1, d2)
    y1, _, _ = _year_month_day(d1)
    y2, _, _ = _year_month_day(d2)

    # Handle same year case
    if y1 == y2
        return days / (_is_leap_year(y1) ? 366.0 : 365.0)
    end

    # For multi-year periods, weight by actual days in each year
    total = 0.0
    for y in y1:y2
        year_days = _is_leap_year(y) ? 366.0 : 365.0
        if y == y1
            # Days remaining in first year
            year_end = _make_date(y + 1, 1, 1)
            total += _day_count(d1, year_end) / year_days
        elseif y == y2
            # Days in final year
            year_start = _make_date(y, 1, 1)
            total += _day_count(year_start, d2) / year_days
        else
            # Full year
            total += 1.0
        end
    end
    total
end

# Helper functions for date handling
# These work with both Date objects and simple (year, month, day) tuples

_day_count(d1::Date, d2::Date) = Dates.value(d2 - d1)
_day_count(d1::DateTime, d2::DateTime) = Dates.value(Date(d2) - Date(d1))
_day_count(d1, d2) = Int(d2 - d1)

_year_month_day(d::Date) = (year(d), month(d), day(d))
_year_month_day(d::DateTime) = (year(d), month(d), day(d))
_year_month_day(d) = (d[1], d[2], d[3])

_make_date(y, m, d) = Date(y, m, d)

function _is_leap_year(y)
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
end

# ============================================================================
# Scheduling and Conventions
# ============================================================================

"""
    Calendar

Simple business-day calendar with weekend and holiday rules.
"""
struct Calendar
    name::Symbol
    holidays::Set{Date}
    weekend::Set{Int}  # 1=Mon ... 7=Sun
end

Calendar(; name::Symbol=:Generic, holidays::Vector{Date}=Date[], weekend::Set{Int}=Set([6, 7])) =
    Calendar(name, Set(holidays), weekend)

WeekendCalendar() = Calendar(name=:Weekend)

"""
    is_business_day(calendar, date) -> Bool

Return true if `date` is a business day under the calendar.
"""
function is_business_day(cal::Calendar, d::Date)
    !(dayofweek(d) in cal.weekend) && !(d in cal.holidays)
end

abstract type BusinessDayConvention end
struct Following <: BusinessDayConvention end
struct ModifiedFollowing <: BusinessDayConvention end
struct Preceding <: BusinessDayConvention end
struct ModifiedPreceding <: BusinessDayConvention end

"""
    adjust_date(date, calendar, bdc) -> Date

Adjust a date using a business-day convention.
"""
function adjust_date(d::Date, cal::Calendar, ::Following)
    dd = d
    while !is_business_day(cal, dd)
        dd += Day(1)
    end
    dd
end

function adjust_date(d::Date, cal::Calendar, ::ModifiedFollowing)
    dd = adjust_date(d, cal, Following())
    if month(dd) != month(d)
        dd = adjust_date(d, cal, Preceding())
    end
    dd
end

function adjust_date(d::Date, cal::Calendar, ::Preceding)
    dd = d
    while !is_business_day(cal, dd)
        dd -= Day(1)
    end
    dd
end

function adjust_date(d::Date, cal::Calendar, ::ModifiedPreceding)
    dd = adjust_date(d, cal, Preceding())
    if month(dd) != month(d)
        dd = adjust_date(d, cal, Following())
    end
    dd
end

abstract type RollRule end
struct NoRoll <: RollRule end
struct EndOfMonth <: RollRule end

abstract type StubRule end
struct StubNone <: StubRule end
struct ShortFront <: StubRule end
struct ShortBack <: StubRule end

is_eom(d::Date) = d == Dates.lastdayofmonth(d)

function _apply_roll(d::Date, roll::RollRule, anchor_eom::Bool)
    if roll isa EndOfMonth && anchor_eom
        return Dates.lastdayofmonth(d)
    end
    d
end

"""
    Schedule

Accrual/payment schedule generated from start/end dates and conventions.
"""
struct Schedule
    dates::Vector{Date}
    calendar::Calendar
    bdc::BusinessDayConvention
    roll::RollRule
    stub::StubRule
end

Base.length(s::Schedule) = length(s.dates)
Base.getindex(s::Schedule, i::Int) = s.dates[i]
Base.iterate(s::Schedule, args...) = iterate(s.dates, args...)

"""
    Schedule(start_date, end_date; tenor=Month(3), calendar=WeekendCalendar(),
             bdc=ModifiedFollowing(), roll=NoRoll(), stub=ShortBack(),
             include_start=true, include_end=true)

Generate an adjusted schedule between two dates.
"""
function Schedule(
    start_date::Date,
    end_date::Date;
    tenor::Period=Month(3),
    calendar::Calendar=WeekendCalendar(),
    bdc::BusinessDayConvention=ModifiedFollowing(),
    roll::RollRule=NoRoll(),
    stub::StubRule=ShortBack(),
    include_start::Bool=true,
    include_end::Bool=true
)
    start_date < end_date || throw(ArgumentError("start_date must be before end_date"))

    anchor_eom = roll isa EndOfMonth && is_eom(start_date)
    dates = Date[]

    if stub isa ShortBack || stub isa StubNone
        d = start_date
        push!(dates, d)
        while true
            d_next = _apply_roll(d + tenor, roll, anchor_eom)
            if d_next >= end_date
                break
            end
            push!(dates, d_next)
            d = d_next
        end
        if stub isa StubNone && d_next != end_date
            throw(ArgumentError("Schedule not aligned to tenor; use ShortBack or ShortFront"))
        end
        push!(dates, end_date)
    elseif stub isa ShortFront
        d = end_date
        push!(dates, d)
        while true
            d_prev = _apply_roll(d - tenor, roll, anchor_eom)
            if d_prev <= start_date
                break
            end
            push!(dates, d_prev)
            d = d_prev
        end
        if stub isa StubNone && d_prev != start_date
            throw(ArgumentError("Schedule not aligned to tenor; use ShortBack or ShortFront"))
        end
        push!(dates, start_date)
        reverse!(dates)
    else
        throw(ArgumentError("Unsupported stub rule: $(typeof(stub))"))
    end

    # Adjust and de-duplicate
    adjusted = [adjust_date(d, calendar, bdc) for d in dates]
    adjusted = unique(adjusted)

    if !include_start && !isempty(adjusted)
        adjusted = adjusted[2:end]
    end
    if !include_end && !isempty(adjusted)
        adjusted = adjusted[1:end-1]
    end

    Schedule(adjusted, calendar, bdc, roll, stub)
end

"""
    schedule_periods(schedule) -> Vector{Tuple{Date,Date}}

Return consecutive accrual periods from the schedule dates.
"""
function schedule_periods(s::Schedule)
    periods = Tuple{Date,Date}[]
    for i in 1:(length(s.dates) - 1)
        push!(periods, (s.dates[i], s.dates[i+1]))
    end
    periods
end

"""
    accrual_factors(schedule, day_count) -> Vector{Float64}

Year-fraction accruals for each period in the schedule.
"""
function accrual_factors(s::Schedule, day_count::DayCountConvention)
    [year_fraction(d1, d2, day_count) for (d1, d2) in schedule_periods(s)]
end

# ============================================================================
# Interpolation Methods
# ============================================================================

"""
    InterpolationMethod

Abstract base type for interpolation methods used in yield curve construction.

Subtypes: [`LinearInterp`](@ref), [`LogLinearInterp`](@ref), [`CubicSplineInterp`](@ref)
"""
abstract type InterpolationMethod end

"""
    LinearInterp <: InterpolationMethod

Linear interpolation between data points.

Simple and stable, but can produce kinks in forward rates.
Best for zero rate curves where smoothness is less critical.

# Example
```julia
curve = ZeroCurve(times, rates; interp=LinearInterp())
```
"""
struct LinearInterp <: InterpolationMethod end

"""
    LogLinearInterp <: InterpolationMethod

Log-linear interpolation (linear in log-space).

The default for discount curves. Ensures discount factors remain positive
and produces smoother forward rates than linear interpolation.

# Example
```julia
curve = DiscountCurve(times, dfs; interp=LogLinearInterp())
```
"""
struct LogLinearInterp <: InterpolationMethod end

"""
    CubicSplineInterp <: InterpolationMethod

Natural cubic spline interpolation.

Produces the smoothest curves with continuous first and second derivatives.
Best for applications requiring smooth forward rates (e.g., HJM models).

# Fields
- `coeffs::Vector{NTuple{4,Float64}}` - Spline coefficients (a, b, c, d) per segment

# Example
```julia
curve = ZeroCurve(times, rates; interp=CubicSplineInterp())
```
"""
struct CubicSplineInterp <: InterpolationMethod
    coeffs::Vector{NTuple{4,Float64}}  # (a, b, c, d) for each segment
end

function CubicSplineInterp()
    CubicSplineInterp(NTuple{4,Float64}[])
end

# Cubic spline coefficient computation
function compute_spline_coeffs(x::Vector{Float64}, y::Vector{Float64})
    n = length(x) - 1
    h = diff(x)

    # Set up tridiagonal system for second derivatives
    A = zeros(n + 1, n + 1)
    b = zeros(n + 1)

    A[1, 1] = 1.0
    A[end, end] = 1.0

    for i in 2:n
        A[i, i-1] = h[i-1]
        A[i, i] = 2(h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])
    end

    c = A \ b

    coeffs = NTuple{4,Float64}[]
    for i in 1:n
        a = y[i]
        b_i = (y[i+1] - y[i])/h[i] - h[i]*(2c[i] + c[i+1])/3
        d = (c[i+1] - c[i])/(3h[i])
        push!(coeffs, (a, b_i, c[i], d))
    end

    coeffs
end

function interp_value(::LinearInterp, x::Vector{Float64}, y::Vector{Float64}, t::Float64)
    t <= x[1] && return y[1]
    t >= x[end] && return y[end]

    i = searchsortedlast(x, t)
    i = clamp(i, 1, length(x) - 1)

    w = (t - x[i]) / (x[i+1] - x[i])
    y[i] * (1 - w) + y[i+1] * w
end

function interp_value(::LogLinearInterp, x::Vector{Float64}, y::Vector{Float64}, t::Float64)
    t <= x[1] && return y[1]
    t >= x[end] && return y[end]

    i = searchsortedlast(x, t)
    i = clamp(i, 1, length(x) - 1)

    w = (t - x[i]) / (x[i+1] - x[i])
    exp(log(y[i]) * (1 - w) + log(y[i+1]) * w)
end

function interp_value(interp::CubicSplineInterp, x::Vector{Float64}, y::Vector{Float64}, t::Float64)
    t <= x[1] && return y[1]
    t >= x[end] && return y[end]

    i = searchsortedlast(x, t)
    i = clamp(i, 1, length(x) - 1)

    if isempty(interp.coeffs)
        # Fall back to linear if coeffs not computed
        return interp_value(LinearInterp(), x, y, t)
    end

    a, b, c, d = interp.coeffs[i]
    dx = t - x[i]
    a + b*dx + c*dx^2 + d*dx^3
end

# ============================================================================
# Yield Curves
# ============================================================================

"""
    RateCurve

Abstract base type for interest rate curves.

All rate curves support the following operations:
- `discount(curve, T)` - Get discount factor to time T
- `zero_rate(curve, T)` - Get zero rate to time T
- `forward_rate(curve, T1, T2)` - Get forward rate between T1 and T2
- `instantaneous_forward(curve, T)` - Get instantaneous forward rate at T

# Subtypes
- [`DiscountCurve`](@ref) - Curve of discount factors
- [`ZeroCurve`](@ref) - Curve of zero rates
- [`ForwardCurve`](@ref) - Curve of instantaneous forward rates
- [`NelsonSiegelCurve`](@ref) - Parametric Nelson-Siegel curve
- [`SvenssonCurve`](@ref) - Parametric Svensson curve

# Example
```julia
# Create a flat 5% curve
curve = ZeroCurve(0.05)

# Get discount factor and zero rate at 2 years
df = discount(curve, 2.0)      # ≈ 0.9048
r = zero_rate(curve, 2.0)      # = 0.05
```
"""
abstract type RateCurve end

"""
    DiscountCurve(times, discount_factors; interp=LogLinearInterp())

Curve of discount factors P(0,T). Interpolates in log-space by default
to ensure discount factors remain positive.

# Arguments
- `times::Vector{Float64}` - Maturities in years
- `discount_factors::Vector{Float64}` - Discount factors (must be positive)
- `interp::InterpolationMethod` - Interpolation method (default: LogLinearInterp)

# Constructors
- `DiscountCurve(times, dfs)` - From vectors
- `DiscountCurve(rate)` - Flat curve at given rate

# Example
```julia
curve = DiscountCurve([0.0, 1.0, 5.0], [1.0, 0.95, 0.78])
df = discount(curve, 2.5)  # Interpolated discount factor
```
"""
struct DiscountCurve <: RateCurve
    times::Vector{Float64}
    values::Vector{Float64}
    interp::InterpolationMethod

    # TODO: Check that discount factors are decreasing with time
    function DiscountCurve(times, values; interp::InterpolationMethod=LogLinearInterp())
        @assert length(times) == length(values)
        @assert all(values .> 0) "Discount factors must be positive"
        idx = sortperm(times)
        new(times[idx], values[idx], interp)
    end
end

"""
    ZeroCurve(times, zero_rates; interp=LinearInterp())

Curve of continuously compounded zero rates.
"""
struct ZeroCurve <: RateCurve
    times::Vector{Float64}
    values::Vector{Float64}
    interp::InterpolationMethod

    function ZeroCurve(times, values; interp::InterpolationMethod=LinearInterp())
        @assert length(times) == length(values)
        idx = sortperm(times)
        new(times[idx], values[idx], interp)
    end
end

"""
    ForwardCurve(times, forward_rates; interp=LinearInterp())

Curve of instantaneous forward rates.
"""
struct ForwardCurve <: RateCurve
    times::Vector{Float64}
    values::Vector{Float64}
    interp::InterpolationMethod

    function ForwardCurve(times, values; interp::InterpolationMethod=LinearInterp())
        @assert length(times) == length(values)
        idx = sortperm(times)
        new(times[idx], values[idx], interp)
    end
end

# Flat curve constructor
DiscountCurve(rate::Float64, max_t::Float64=30.0) =
    DiscountCurve([0.0, max_t], [1.0, exp(-rate * max_t)])
ZeroCurve(rate::Float64, max_t::Float64=30.0) =
    ZeroCurve([0.0, max_t], [rate, rate])

# Core curve operations
"""
    discount(curve, T) -> Float64

Discount factor from time 0 to time T.
"""
function discount(curve::DiscountCurve, T::Float64)
    T <= 0 && return 1.0
    interp_value(curve.interp, curve.times, curve.values, T)
end

function discount(curve::ZeroCurve, T::Float64)
    T <= 0 && return 1.0
    r = interp_value(curve.interp, curve.times, curve.values, T)
    exp(-r * T)
end

function discount(curve::ForwardCurve, T::Float64)
    T <= 0 && return 1.0
    # Integrate forward rates: DF = exp(-∫f(s)ds)
    n = 100
    dt = T / n
    integral = sum(interp_value(curve.interp, curve.times, curve.values, i*dt) * dt for i in 0:n-1)
    exp(-integral)
end

"""
    zero_rate(curve, T) -> Float64

Continuously compounded zero rate to time T.
"""
function zero_rate(curve::DiscountCurve, T::Float64)
    T <= 0 && return curve.values[1] > 0 ? -log(curve.values[1]) : 0.0
    df = discount(curve, T)
    -log(df) / T
end

function zero_rate(curve::ZeroCurve, T::Float64)
    T <= 0 && return curve.values[1]
    interp_value(curve.interp, curve.times, curve.values, T)
end

function zero_rate(curve::ForwardCurve, T::Float64)
    T <= 0 && return curve.values[1]
    df = discount(curve, T)
    -log(df) / T
end

"""
    forward_rate(curve, T1, T2) -> Float64

Simply compounded forward rate between T1 and T2.
"""
function forward_rate(curve::RateCurve, T1::Float64, T2::Float64)
    @assert T2 > T1 "T2 must be greater than T1"
    df1 = discount(curve, T1)
    df2 = discount(curve, T2)
    (df1 / df2 - 1) / (T2 - T1)
end

"""
    instantaneous_forward(curve, T) -> Float64

Instantaneous forward rate at time T: f(T) = -d/dT ln(P(0,T))
"""
function instantaneous_forward(curve::RateCurve, T::Float64)
    ε = 1e-6
    df_plus = discount(curve, T + ε)
    df_minus = discount(curve, T - ε)
    -log(df_plus / df_minus) / (2ε)
end

function instantaneous_forward(curve::ForwardCurve, T::Float64)
    interp_value(curve.interp, curve.times, curve.values, T)
end

# Curve conversions
# Sample many points to preserve interpolation accuracy across methods
function DiscountCurve(zc::ZeroCurve)
    t_max = zc.times[end]
    n_points = max(100, length(zc.times) * 20)
    times = vcat([0.0], collect(range(1e-6, t_max, length=n_points)))
    dfs = [discount(zc, t) for t in times]
    DiscountCurve(times, dfs; interp=LogLinearInterp())
end

function ZeroCurve(dc::DiscountCurve)
    t_max = dc.times[end]
    n_points = max(100, length(dc.times) * 20)
    times = vcat([0.0], collect(range(1e-6, t_max, length=n_points)))
    rates = [zero_rate(dc, t) for t in times]
    ZeroCurve(times, rates; interp=LinearInterp())
end

# ============================================================================
# Multi-Curve Framework
# ============================================================================

"""
    RateIndex

Market index conventions for forwarding (e.g., SOFR3M, EURIBOR6M).
"""
struct RateIndex
    name::Symbol
    tenor::Period
    day_count::DayCountConvention
    calendar::Calendar
    bdc::BusinessDayConvention
    fixing_lag::Period
    payment_lag::Period
    compounding::Symbol
end

function RateIndex(
    name::Symbol;
    tenor::Period=Month(3),
    day_count::DayCountConvention=ACT360(),
    calendar::Calendar=WeekendCalendar(),
    bdc::BusinessDayConvention=ModifiedFollowing(),
    fixing_lag::Period=Day(2),
    payment_lag::Period=Day(0),
    compounding::Symbol=:simple
)
    RateIndex(name, tenor, day_count, calendar, bdc, fixing_lag, payment_lag, compounding)
end

"""
    fixing_date(index, accrual_start) -> Date
"""
fixing_date(idx::RateIndex, accrual_start::Date) =
    adjust_date(accrual_start - idx.fixing_lag, idx.calendar, idx.bdc)

"""
    payment_date(index, accrual_end) -> Date
"""
payment_date(idx::RateIndex, accrual_end::Date) =
    adjust_date(accrual_end + idx.payment_lag, idx.calendar, idx.bdc)

"""
    CurveSet

Container for discount curve and multiple forwarding curves keyed by index name.
"""
struct CurveSet
    asof::Date
    discount::RateCurve
    forwards::Dict{Symbol,RateCurve}
    day_count::DayCountConvention
end

function CurveSet(
    asof::Date,
    discount::RateCurve;
    forwards::AbstractDict{Symbol,<:RateCurve}=Dict{Symbol,RateCurve}(),
    day_count::DayCountConvention=ACT365()
)
    CurveSet(asof, discount, Dict{Symbol,RateCurve}(k => v for (k, v) in forwards), day_count)
end

discount(cs::CurveSet, t::Real) = discount(cs.discount, Float64(t))

function discount(cs::CurveSet, d::Date; day_count::DayCountConvention=cs.day_count)
    t = year_fraction(cs.asof, d, day_count)
    discount(cs.discount, t)
end

function forward_rate(cs::CurveSet, idx::Symbol, t1::Real, t2::Real)
    curve = get(cs.forwards, idx, cs.discount)
    forward_rate(curve, Float64(t1), Float64(t2))
end

function forward_rate(cs::CurveSet, idx::RateIndex, d1::Date, d2::Date)
    t1 = year_fraction(cs.asof, d1, idx.day_count)
    t2 = year_fraction(cs.asof, d2, idx.day_count)
    curve = get(cs.forwards, idx.name, cs.discount)
    forward_rate(curve, t1, t2)
end

function _with_discount(cs::CurveSet, curve::RateCurve)
    CurveSet(cs.asof, curve; forwards=cs.forwards, day_count=cs.day_count)
end

function _with_forward(cs::CurveSet, name::Symbol, curve::RateCurve)
    forwards = copy(cs.forwards)
    forwards[name] = curve
    CurveSet(cs.asof, cs.discount; forwards=forwards, day_count=cs.day_count)
end

# ============================================================================
# Parametric Yield Curves (Nelson-Siegel, Svensson)
# ============================================================================

"""
    NelsonSiegelCurve(β0, β1, β2, τ)

Nelson-Siegel parametric yield curve model.

The zero rate at maturity T is given by:
```
r(T) = β₀ + β₁ * (1 - exp(-T/τ)) / (T/τ) + β₂ * ((1 - exp(-T/τ)) / (T/τ) - exp(-T/τ))
```

# Parameters
- `β0` - Long-term rate level (asymptotic rate as T → ∞)
- `β1` - Short-term component (controls slope at origin)
- `β2` - Medium-term component (controls curvature/hump)
- `τ` - Decay factor (time at which medium-term component reaches maximum)

# Example
```julia
curve = NelsonSiegelCurve(0.05, -0.02, 0.01, 2.0)
zero_rate(curve, 5.0)  # Get 5-year zero rate
```
"""
struct NelsonSiegelCurve <: RateCurve
    β0::Float64
    β1::Float64
    β2::Float64
    τ::Float64

    function NelsonSiegelCurve(β0, β1, β2, τ)
        τ > 0 || throw(ArgumentError("τ must be positive, got $τ"))
        new(β0, β1, β2, τ)
    end
end

"""
    SvenssonCurve(β0, β1, β2, β3, τ1, τ2)

Svensson extension of Nelson-Siegel with an additional hump component.

The zero rate at maturity T is given by:
```
r(T) = β₀ + β₁ * g1(T/τ1) + β₂ * h1(T/τ1) + β₃ * h2(T/τ2)
```
where:
- g1(x) = (1 - exp(-x)) / x
- h1(x) = g1(x) - exp(-x)
- h2(x) = g1(x, τ2) - exp(-T/τ2)

# Parameters
- `β0` - Long-term rate level
- `β1` - Short-term component
- `β2` - First medium-term component (hump at τ1)
- `β3` - Second medium-term component (hump at τ2)
- `τ1` - First decay factor
- `τ2` - Second decay factor

# Example
```julia
curve = SvenssonCurve(0.05, -0.02, 0.01, 0.005, 2.0, 5.0)
zero_rate(curve, 10.0)  # Get 10-year zero rate
```
"""
struct SvenssonCurve <: RateCurve
    β0::Float64
    β1::Float64
    β2::Float64
    β3::Float64
    τ1::Float64
    τ2::Float64

    function SvenssonCurve(β0, β1, β2, β3, τ1, τ2)
        τ1 > 0 || throw(ArgumentError("τ1 must be positive, got $τ1"))
        τ2 > 0 || throw(ArgumentError("τ2 must be positive, got $τ2"))
        new(β0, β1, β2, β3, τ1, τ2)
    end
end

# Nelson-Siegel loading functions
function _ns_g1(x::Float64)
    # g1(x) = (1 - exp(-x)) / x, with limit g1(0) = 1
    abs(x) < 1e-10 ? 1.0 - x/2 + x^2/6 : (1 - exp(-x)) / x
end

function _ns_h1(x::Float64)
    # h1(x) = g1(x) - exp(-x)
    _ns_g1(x) - exp(-x)
end

# Zero rate implementations
function zero_rate(curve::NelsonSiegelCurve, T::Float64)
    T <= 0 && return curve.β0 + curve.β1  # Instantaneous rate
    x = T / curve.τ
    curve.β0 + curve.β1 * _ns_g1(x) + curve.β2 * _ns_h1(x)
end

function zero_rate(curve::SvenssonCurve, T::Float64)
    T <= 0 && return curve.β0 + curve.β1  # Instantaneous rate
    x1 = T / curve.τ1
    x2 = T / curve.τ2
    curve.β0 + curve.β1 * _ns_g1(x1) + curve.β2 * _ns_h1(x1) + curve.β3 * _ns_h1(x2)
end

# Discount factors
function discount(curve::NelsonSiegelCurve, T::Float64)
    T <= 0 && return 1.0
    exp(-zero_rate(curve, T) * T)
end

function discount(curve::SvenssonCurve, T::Float64)
    T <= 0 && return 1.0
    exp(-zero_rate(curve, T) * T)
end

# Instantaneous forward rate: f(T) = -d/dT ln(P(0,T)) = r(T) + T * dr/dT
function instantaneous_forward(curve::NelsonSiegelCurve, T::Float64)
    T <= 0 && return curve.β0 + curve.β1
    x = T / curve.τ
    ex = exp(-x)

    # r(T) = β0 + β1 * g1 + β2 * h1
    # f(T) = r(T) + T * dr/dT
    # dr/dT = (β1/τ) * dg1/dx + (β2/τ) * dh1/dx
    # where dg1/dx = (x*exp(-x) - (1-exp(-x)))/x² = (exp(-x) - g1)/x
    # and dh1/dx = dg1/dx + exp(-x) = (exp(-x) - g1)/x + exp(-x)

    g1 = _ns_g1(x)
    dg1_dx = abs(x) < 1e-10 ? -0.5 + x/3 : (ex - g1) / x
    dh1_dx = dg1_dx + ex

    r = curve.β0 + curve.β1 * g1 + curve.β2 * _ns_h1(x)
    dr_dT = (curve.β1 * dg1_dx + curve.β2 * dh1_dx) / curve.τ

    r + T * dr_dT
end

function instantaneous_forward(curve::SvenssonCurve, T::Float64)
    # Use numerical differentiation for Svensson
    ε = 1e-6
    df_plus = discount(curve, T + ε)
    df_minus = discount(curve, max(0.0, T - ε))
    -log(df_plus / df_minus) / (2ε)
end

"""
    fit_nelson_siegel(maturities, rates; initial_guess=nothing) -> NelsonSiegelCurve

Fit a Nelson-Siegel curve to observed zero rates using least squares.

# Arguments
- `maturities` - Vector of maturities (in years)
- `rates` - Vector of observed zero rates
- `initial_guess` - Optional (β0, β1, β2, τ) starting point

# Returns
A fitted NelsonSiegelCurve

# Example
```julia
mats = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
rates = [0.02, 0.022, 0.025, 0.028, 0.032, 0.035, 0.037]
curve = fit_nelson_siegel(mats, rates)
```
"""
function fit_nelson_siegel(maturities::Vector{Float64}, rates::Vector{Float64};
                           initial_guess::Union{Nothing, NTuple{4,Float64}}=nothing)
    @assert length(maturities) == length(rates)
    @assert length(maturities) >= 4 "Need at least 4 data points to fit Nelson-Siegel"

    # Default initial guess
    if initial_guess === nothing
        β0_init = rates[end]  # Long rate
        β1_init = rates[1] - rates[end]  # Slope
        β2_init = 0.0  # Curvature
        τ_init = 2.0  # Typical decay
    else
        β0_init, β1_init, β2_init, τ_init = initial_guess
    end

    # Simple gradient descent optimization
    params = [β0_init, β1_init, β2_init, τ_init]

    function loss(p)
        β0, β1, β2, τ = p
        τ = max(τ, 0.01)  # Keep τ positive
        curve = NelsonSiegelCurve(β0, β1, β2, τ)
        sum((zero_rate(curve, t) - r)^2 for (t, r) in zip(maturities, rates))
    end

    # Gradient-free Nelder-Mead simplex optimization
    params = _nelder_mead(loss, params, 1000, 1e-10)

    NelsonSiegelCurve(params[1], params[2], params[3], max(params[4], 0.01))
end

"""
    fit_svensson(maturities, rates; initial_guess=nothing) -> SvenssonCurve

Fit a Svensson curve to observed zero rates using least squares.

# Arguments
- `maturities` - Vector of maturities (in years)
- `rates` - Vector of observed zero rates
- `initial_guess` - Optional (β0, β1, β2, β3, τ1, τ2) starting point

# Returns
A fitted SvenssonCurve
"""
function fit_svensson(maturities::Vector{Float64}, rates::Vector{Float64};
                      initial_guess::Union{Nothing, NTuple{6,Float64}}=nothing)
    @assert length(maturities) == length(rates)
    @assert length(maturities) >= 6 "Need at least 6 data points to fit Svensson"

    # Default initial guess
    if initial_guess === nothing
        β0_init = rates[end]
        β1_init = rates[1] - rates[end]
        β2_init = 0.0
        β3_init = 0.0
        τ1_init = 2.0
        τ2_init = 5.0
    else
        β0_init, β1_init, β2_init, β3_init, τ1_init, τ2_init = initial_guess
    end

    params = [β0_init, β1_init, β2_init, β3_init, τ1_init, τ2_init]

    function loss(p)
        β0, β1, β2, β3, τ1, τ2 = p
        τ1 = max(τ1, 0.01)
        τ2 = max(τ2, 0.01)
        curve = SvenssonCurve(β0, β1, β2, β3, τ1, τ2)
        sum((zero_rate(curve, t) - r)^2 for (t, r) in zip(maturities, rates))
    end

    params = _nelder_mead(loss, params, 2000, 1e-10)

    SvenssonCurve(params[1], params[2], params[3], params[4],
                  max(params[5], 0.01), max(params[6], 0.01))
end

# Simple Nelder-Mead optimizer (no external dependencies)
function _nelder_mead(f, x0::Vector{Float64}, max_iter::Int, tol::Float64)
    n = length(x0)

    # Initialize simplex
    simplex = Vector{Vector{Float64}}(undef, n + 1)
    simplex[1] = copy(x0)
    for i in 2:n+1
        simplex[i] = copy(x0)
        simplex[i][i-1] += 0.1 * max(abs(x0[i-1]), 1.0)
    end

    # Evaluate function at all vertices
    fvals = [f(v) for v in simplex]

    α, γ, ρ, σ = 1.0, 2.0, 0.5, 0.5  # Standard parameters

    for _ in 1:max_iter
        # Sort vertices by function value
        order = sortperm(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        # Check convergence
        if maximum(abs(fvals[i] - fvals[1]) for i in 2:n+1) < tol
            break
        end

        # Centroid of best n points
        x0_new = sum(simplex[1:n]) / n

        # Reflection
        xr = x0_new + α * (x0_new - simplex[end])
        fr = f(xr)

        if fvals[1] <= fr < fvals[n]
            simplex[end] = xr
            fvals[end] = fr
        elseif fr < fvals[1]
            # Expansion
            xe = x0_new + γ * (xr - x0_new)
            fe = f(xe)
            if fe < fr
                simplex[end] = xe
                fvals[end] = fe
            else
                simplex[end] = xr
                fvals[end] = fr
            end
        else
            # Contraction
            xc = x0_new + ρ * (simplex[end] - x0_new)
            fc = f(xc)
            if fc < fvals[end]
                simplex[end] = xc
                fvals[end] = fc
            else
                # Shrink
                for i in 2:n+1
                    simplex[i] = simplex[1] + σ * (simplex[i] - simplex[1])
                    fvals[i] = f(simplex[i])
                end
            end
        end
    end

    simplex[argmin(fvals)]
end

# ============================================================================
# Curve Bootstrapping
# ============================================================================

abstract type MarketInstrument end

"""Deposit rate: simple rate for short maturities"""
struct DepositRate <: MarketInstrument
    maturity::Float64
    rate::Float64
end

"""Futures rate: convexity-adjusted forward rate"""
struct FuturesRate <: MarketInstrument
    start::Float64
    maturity::Float64
    rate::Float64
    convexity_adj::Float64
end
FuturesRate(start, mat, rate) = FuturesRate(start, mat, rate, 0.0)

"""Swap rate: par swap rate"""
struct SwapRate <: MarketInstrument
    maturity::Float64
    rate::Float64
    frequency::Int  # payments per year
end
SwapRate(mat, rate) = SwapRate(mat, rate, 2)  # semi-annual default

"""
    bootstrap(instruments; interp=LogLinearInterp()) -> DiscountCurve

Bootstrap a discount curve from market instruments using sequential stripping.

The function iteratively solves for discount factors that reprice each
instrument, starting from the shortest maturity. Instruments should be
provided in order of increasing maturity.

# Arguments
- `instruments::Vector{<:MarketInstrument}` - Market quotes (sorted by maturity)
- `interp::InterpolationMethod` - Interpolation for intermediate points

# Supported Instruments
- [`DepositRate`](@ref) - Money market deposits (short end)
- [`FuturesRate`](@ref) - Interest rate futures (middle)
- [`SwapRate`](@ref) - Par swap rates (long end)

# Returns
A [`DiscountCurve`](@ref) that reprices all input instruments.

# Example
```julia
instruments = [
    DepositRate(0.25, 0.02),   # 3-month deposit at 2%
    DepositRate(0.5, 0.022),   # 6-month deposit at 2.2%
    SwapRate(2.0, 0.028),      # 2-year swap at 2.8%
    SwapRate(5.0, 0.032),      # 5-year swap at 3.2%
]
curve = bootstrap(instruments)
```

See also: [`DepositRate`](@ref), [`FuturesRate`](@ref), [`SwapRate`](@ref)
"""
function bootstrap(instruments::Vector{<:MarketInstrument}; interp::InterpolationMethod=LogLinearInterp())
    times = Float64[0.0]
    dfs = Float64[1.0]

    for inst in instruments
        df = bootstrap_instrument(inst, times, dfs)
        push!(times, get_maturity(inst))
        push!(dfs, df)
    end

    DiscountCurve(times, dfs; interp=interp)
end

get_maturity(d::DepositRate) = d.maturity
get_maturity(f::FuturesRate) = f.maturity
get_maturity(s::SwapRate) = s.maturity

function bootstrap_instrument(d::DepositRate, times, dfs)
    # DF = 1 / (1 + r * T) for simple rate
    1.0 / (1.0 + d.rate * d.maturity)
end

function bootstrap_instrument(f::FuturesRate, times, dfs)
    # Need DF to start date
    df_start = interp_value(LogLinearInterp(), times, dfs, f.start)
    # Forward rate (adjusted)
    fwd = f.rate - f.convexity_adj
    tau = f.maturity - f.start
    df_start / (1.0 + fwd * tau)
end

function bootstrap_instrument(s::SwapRate, times, dfs)
    # Par swap: sum of DF * tau = (1 - DF_N) / rate
    # Solve for DF_N
    tau = 1.0 / s.frequency
    n_periods = Int(s.maturity * s.frequency)

    pv_fixed = 0.0
    for i in 1:(n_periods - 1)
        t = i * tau
        df = interp_value(LogLinearInterp(), times, dfs, t)
        pv_fixed += df * tau
    end

    # DF_N = (1 - rate * pv_fixed) / (1 + rate * tau)
    (1.0 - s.rate * pv_fixed) / (1.0 + s.rate * tau)
end

# ============================================================================
# Multi-Curve Bootstrapping (OIS + FRA/IRS)
# ============================================================================

abstract type CurveQuote end

"""
    OISDepositQuote(start_date, end_date, rate; day_count=ACT360())
"""
struct OISDepositQuote <: CurveQuote
    start_date::Date
    end_date::Date
    rate::Float64
    day_count::DayCountConvention
end

OISDepositQuote(start_date, end_date, rate; day_count::DayCountConvention=ACT360()) =
    OISDepositQuote(start_date, end_date, rate, day_count)

"""
    OISSwapQuote(start_date, maturity_date, fixed_rate;
                 fixed_frequency=1, fixed_day_count=ACT360(),
                 calendar=WeekendCalendar(), bdc=ModifiedFollowing())
"""
struct OISSwapQuote <: CurveQuote
    start_date::Date
    maturity_date::Date
    fixed_rate::Float64
    fixed_frequency::Int
    fixed_day_count::DayCountConvention
    calendar::Calendar
    bdc::BusinessDayConvention
end

OISSwapQuote(start_date, maturity_date, fixed_rate;
             fixed_frequency::Int=1,
             fixed_day_count::DayCountConvention=ACT360(),
             calendar::Calendar=WeekendCalendar(),
             bdc::BusinessDayConvention=ModifiedFollowing()) =
    OISSwapQuote(start_date, maturity_date, fixed_rate, fixed_frequency, fixed_day_count, calendar, bdc)

"""
    FRAQuote(start_date, end_date, rate, index)
"""
struct FRAQuote <: CurveQuote
    start_date::Date
    end_date::Date
    rate::Float64
    index::RateIndex
end

"""
    IRSwapQuote(start_date, maturity_date, fixed_rate, index;
                fixed_frequency=2, fixed_day_count=Thirty360(),
                calendar=WeekendCalendar(), bdc=ModifiedFollowing())
"""
struct IRSwapQuote <: CurveQuote
    start_date::Date
    maturity_date::Date
    fixed_rate::Float64
    fixed_frequency::Int
    fixed_day_count::DayCountConvention
    index::RateIndex
    calendar::Calendar
    bdc::BusinessDayConvention
end

IRSwapQuote(start_date, maturity_date, fixed_rate, index;
            fixed_frequency::Int=2,
            fixed_day_count::DayCountConvention=Thirty360(),
            calendar::Calendar=WeekendCalendar(),
            bdc::BusinessDayConvention=ModifiedFollowing()) =
    IRSwapQuote(start_date, maturity_date, fixed_rate, fixed_frequency, fixed_day_count, index, calendar, bdc)

quote_maturity(q::OISDepositQuote) = q.end_date
quote_maturity(q::OISSwapQuote) = q.maturity_date
quote_maturity(q::FRAQuote) = q.end_date
quote_maturity(q::IRSwapQuote) = q.maturity_date

function _tenor_from_frequency(freq::Int)
    (12 % freq == 0) || throw(ArgumentError("fixed_frequency must divide 12, got $freq"))
    Month(Int(12 ÷ freq))
end

function _discount_at(curve::RateCurve, asof::Date, d::Date, day_count::DayCountConvention)
    t = year_fraction(asof, d, day_count)
    discount(curve, t)
end

function _temp_curve(times, dfs; interp::InterpolationMethod=LogLinearInterp())
    DiscountCurve(times, dfs; interp=interp)
end

"""
    bootstrap_ois_curve(quotes; asof, curve_day_count=ACT365(), interp=LogLinearInterp())

Bootstrap an OIS discount curve from OIS deposits and OIS swaps.
"""
function bootstrap_ois_curve(
    quotes::Vector{<:CurveQuote};
    asof::Date,
    curve_day_count::DayCountConvention=ACT365(),
    interp::InterpolationMethod=LogLinearInterp()
)
    filtered = [q for q in quotes if q isa OISDepositQuote || q isa OISSwapQuote]
    isempty(filtered) && throw(ArgumentError("No OIS quotes provided"))
    sorted = sort(filtered; by=quote_maturity)

    times = Float64[0.0]
    dfs = Float64[1.0]

    for q in sorted
        curve = _temp_curve(times, dfs; interp=interp)
        if q isa OISDepositQuote
            qq = q::OISDepositQuote
            accrual = year_fraction(qq.start_date, qq.end_date, qq.day_count)
            df_start = _discount_at(curve, asof, qq.start_date, curve_day_count)
            df_end = df_start / (1.0 + qq.rate * accrual)
            t_end = year_fraction(asof, qq.end_date, curve_day_count)
            push!(times, t_end)
            push!(dfs, df_end)
        elseif q isa OISSwapQuote
            qq = q::OISSwapQuote
            tenor = _tenor_from_frequency(qq.fixed_frequency)
            sched = Schedule(qq.start_date, qq.maturity_date; tenor=tenor, calendar=qq.calendar, bdc=qq.bdc)
            periods = schedule_periods(sched)
            n = length(periods)
            n == 0 && throw(ArgumentError("OIS swap schedule is empty"))

            sum_prev = 0.0
            for i in 1:(n - 1)
                d1, d2 = periods[i]
                accrual = year_fraction(d1, d2, qq.fixed_day_count)
                df = _discount_at(curve, asof, d2, curve_day_count)
                sum_prev += accrual * df
            end

            d1_last, d2_last = periods[end]
            accrual_last = year_fraction(d1_last, d2_last, qq.fixed_day_count)
            df_end = (1.0 - qq.fixed_rate * sum_prev) / (1.0 + qq.fixed_rate * accrual_last)
            t_end = year_fraction(asof, d2_last, curve_day_count)
            push!(times, t_end)
            push!(dfs, df_end)
        end
    end

    DiscountCurve(times, dfs; interp=interp)
end

"""
    bootstrap_forward_curve(quotes; asof, discount_curve, interp=LogLinearInterp())

Bootstrap a forwarding curve from FRA and IRS quotes using a discount curve.
"""
function bootstrap_forward_curve(
    quotes::Vector{<:CurveQuote};
    asof::Date,
    discount_curve::RateCurve,
    curve_day_count::DayCountConvention=ACT365(),
    interp::InterpolationMethod=LogLinearInterp()
)
    filtered = [q for q in quotes if q isa FRAQuote || q isa IRSwapQuote]
    isempty(filtered) && throw(ArgumentError("No FRA/IRS quotes provided"))
    sorted = sort(filtered; by=quote_maturity)

    idx = nothing
    for q in sorted
        if q isa FRAQuote
            idx = (q::FRAQuote).index
            break
        elseif q isa IRSwapQuote
            idx = (q::IRSwapQuote).index
            break
        end
    end
    idx === nothing && throw(ArgumentError("Forward quotes require an index"))

    times = Float64[0.0]
    dfs = Float64[1.0]

    for q in sorted
        curve = _temp_curve(times, dfs; interp=interp)
        if q isa FRAQuote
            qq = q::FRAQuote
            accrual = year_fraction(qq.start_date, qq.end_date, qq.index.day_count)
            t_start = year_fraction(asof, qq.start_date, qq.index.day_count)
            t_end = year_fraction(asof, qq.end_date, qq.index.day_count)
            df_start = discount(curve, t_start)
            df_end = df_start / (1.0 + qq.rate * accrual)
            push!(times, t_end)
            push!(dfs, df_end)
        elseif q isa IRSwapQuote
            qq = q::IRSwapQuote
            tenor_fixed = _tenor_from_frequency(qq.fixed_frequency)
            fixed_sched = Schedule(qq.start_date, qq.maturity_date; tenor=tenor_fixed, calendar=qq.calendar, bdc=qq.bdc)
            float_sched = Schedule(qq.start_date, qq.maturity_date; tenor=qq.index.tenor, calendar=qq.calendar, bdc=qq.bdc)

            fixed_annuity = 0.0
            for (d1, d2) in schedule_periods(fixed_sched)
                accrual = year_fraction(d1, d2, qq.fixed_day_count)
                df = _discount_at(discount_curve, asof, adjust_date(d2, qq.calendar, qq.bdc), curve_day_count)
                fixed_annuity += accrual * df
            end

            float_periods = schedule_periods(float_sched)
            n = length(float_periods)
            n == 0 && throw(ArgumentError("IRS float schedule is empty"))

            float_pv = 0.0
            for i in 1:(n - 1)
                d1, d2 = float_periods[i]
                accrual = year_fraction(d1, d2, qq.index.day_count)
                t1 = year_fraction(asof, d1, qq.index.day_count)
                t2 = year_fraction(asof, d2, qq.index.day_count)
                fwd = forward_rate(curve, t1, t2)
                df = _discount_at(discount_curve, asof, payment_date(qq.index, d2), curve_day_count)
                float_pv += fwd * accrual * df
            end

            d1_last, d2_last = float_periods[end]
            t1_last = year_fraction(asof, d1_last, qq.index.day_count)
            t2_last = year_fraction(asof, d2_last, qq.index.day_count)
            df_start = discount(curve, t1_last)
            df_disc = _discount_at(discount_curve, asof, payment_date(qq.index, d2_last), curve_day_count)
            k = qq.fixed_rate * fixed_annuity - float_pv
            df_end = df_start / (1.0 + k / df_disc)

            push!(times, t2_last)
            push!(dfs, df_end)
        end
    end

    DiscountCurve(times, dfs; interp=interp)
end

"""
    bootstrap_curveset(ois_quotes, fwd_quotes; asof, curve_day_count=ACT365())

Build a CurveSet with an OIS discount curve and a single forwarding curve.
"""
function bootstrap_curveset(
    ois_quotes::Vector{<:CurveQuote},
    fwd_quotes::Vector{<:CurveQuote};
    asof::Date,
    curve_day_count::DayCountConvention=ACT365()
)
    dc = bootstrap_ois_curve(ois_quotes; asof=asof, curve_day_count=curve_day_count)
    fc = bootstrap_forward_curve(fwd_quotes; asof=asof, discount_curve=dc, curve_day_count=curve_day_count)

    idx = nothing
    for q in fwd_quotes
        if q isa FRAQuote
            idx = (q::FRAQuote).index
            break
        elseif q isa IRSwapQuote
            idx = (q::IRSwapQuote).index
            break
        end
    end
    idx === nothing && throw(ArgumentError("Forward quotes require an index"))

    CurveSet(asof, dc; forwards=Dict(idx.name => fc), day_count=curve_day_count)
end

# ============================================================================
# Bonds
# ============================================================================

"""
    Bond

Abstract base type for fixed income instruments.

All bonds support the following operations:
- `price(bond, curve)` - Present value using a discount curve
- `price(bond, yield)` - Present value at a given yield
- `yield_to_maturity(bond, price)` - Solve for yield given price
- `duration(bond, yield)` - Macaulay duration
- `modified_duration(bond, yield)` - Modified duration
- `convexity(bond, yield)` - Convexity
- `dv01(bond, yield)` - Dollar value of 1 basis point

# Subtypes
- [`ZeroCouponBond`](@ref) - Zero-coupon (discount) bond
- [`FixedRateBond`](@ref) - Fixed-rate coupon bond
- [`FloatingRateBond`](@ref) - Floating-rate bond

# Example
```julia
bond = FixedRateBond(5.0, 0.04, 2)  # 5-year, 4% semi-annual
curve = ZeroCurve(0.05)
pv = price(bond, curve)
ytm = yield_to_maturity(bond, 95.0)
dur = duration(bond, ytm)
```
"""
abstract type Bond end

"""
    ZeroCouponBond(maturity, face_value=100.0)

Zero-coupon bond paying face value at maturity.

# Arguments
- `maturity::Float64` - Time to maturity in years
- `face_value::Float64` - Face (par) value (default: 100.0)

# Example
```julia
zcb = ZeroCouponBond(5.0)  # 5-year zero
price(zcb, 0.05)  # ≈ 78.12 at 5% yield
```
"""
struct ZeroCouponBond <: Bond
    maturity::Float64
    face_value::Float64
end
ZeroCouponBond(maturity) = ZeroCouponBond(maturity, 100.0)

"""
    FixedRateBond(maturity, coupon_rate, frequency=2, face_value=100.0)

Fixed-rate coupon bond. Coupon rate is annual, paid at given frequency.
"""
struct FixedRateBond <: Bond
    maturity::Float64
    coupon_rate::Float64
    frequency::Int
    face_value::Float64
end
FixedRateBond(mat, cpn) = FixedRateBond(mat, cpn, 2, 100.0)
FixedRateBond(mat, cpn, freq) = FixedRateBond(mat, cpn, freq, 100.0)

"""
    FloatingRateBond(maturity, spread, frequency=4, face_value=100.0)

Floating-rate bond paying reference rate + spread.
"""
struct FloatingRateBond <: Bond
    maturity::Float64
    spread::Float64
    frequency::Int
    face_value::Float64
end
FloatingRateBond(mat, spread) = FloatingRateBond(mat, spread, 4, 100.0)

# Cash flows
function cash_flows(bond::ZeroCouponBond)
    [(bond.maturity, bond.face_value)]
end

function cash_flows(bond::FixedRateBond)
    tau = 1.0 / bond.frequency
    n = Int(ceil(bond.maturity * bond.frequency))
    coupon = bond.face_value * bond.coupon_rate * tau

    flows = Tuple{Float64,Float64}[]
    for i in 1:n
        t = i * tau
        cf = i == n ? coupon + bond.face_value : coupon
        push!(flows, (t, cf))
    end
    flows
end

"""
    price(bond, curve) -> Float64

Present value of bond using discount curve.
"""
function price(bond::Bond, curve::RateCurve)
    sum(cf * discount(curve, t) for (t, cf) in cash_flows(bond))
end

"""
    price(bond, yield) -> Float64

Present value of bond at given yield (continuously compounded).
"""
function price(bond::Bond, yield::Float64)
    sum(cf * exp(-yield * t) for (t, cf) in cash_flows(bond))
end

"""
    yield_to_maturity(bond, market_price; tol=1e-10) -> Float64

Solve for yield given market price using Newton-Raphson.
"""
function yield_to_maturity(bond::Bond, market_price::Float64; tol::Float64=1e-10, max_iter::Int=100)
    # Initial guess from simple yield
    y = bond isa ZeroCouponBond ?
        -log(market_price / bond.face_value) / bond.maturity :
        (bond.coupon_rate * bond.face_value + (bond.face_value - market_price) / bond.maturity) / market_price

    for _ in 1:max_iter
        p = price(bond, y)
        if abs(p - market_price) < tol
            return y
        end
        # dp/dy = -sum(t * cf * exp(-y*t))
        dp = -sum(t * cf * exp(-y * t) for (t, cf) in cash_flows(bond))
        y -= (p - market_price) / dp
    end

    y
end

"""
    duration(bond, yield) -> Float64

Macaulay duration: weighted average time to cash flows.
"""
function duration(bond::Bond, yield::Float64)
    p = price(bond, yield)
    sum(t * cf * exp(-yield * t) for (t, cf) in cash_flows(bond)) / p
end

"""
    modified_duration(bond, yield) -> Float64

Modified duration: -1/P * dP/dy
"""
function modified_duration(bond::Bond, yield::Float64)
    duration(bond, yield) / (1 + yield)
end

"""
    convexity(bond, yield) -> Float64

Convexity: 1/P * d²P/dy²
"""
function convexity(bond::Bond, yield::Float64)
    p = price(bond, yield)
    sum(t^2 * cf * exp(-yield * t) for (t, cf) in cash_flows(bond)) / p
end

"""
    dv01(bond, yield) -> Float64

Dollar value of 1 basis point: price change for 1bp yield move.
"""
function dv01(bond::Bond, yield::Float64)
    p = price(bond, yield)
    modified_duration(bond, yield) * p * 0.0001
end

"""
    accrued_interest(bond, settlement_time) -> Float64

Accrued interest from last coupon to settlement.
"""
function accrued_interest(bond::FixedRateBond, settlement_time::Float64)
    tau = 1.0 / bond.frequency
    last_coupon = floor(settlement_time / tau) * tau
    accrual_fraction = (settlement_time - last_coupon) / tau
    bond.face_value * bond.coupon_rate * tau * accrual_fraction
end
accrued_interest(::ZeroCouponBond, ::Float64) = 0.0

"""Clean price = dirty price - accrued interest"""
clean_price(bond::Bond, curve::RateCurve, settlement::Float64=0.0) =
    price(bond, curve) - accrued_interest(bond, settlement)

"""Dirty price = full price including accrued"""
dirty_price(bond::Bond, curve::RateCurve) = price(bond, curve)

# ============================================================================
# Short Rate Models
# ============================================================================

"""
    ShortRateModel

Base type for short-rate interest rate models.
"""
abstract type ShortRateModel end

"""
    Vasicek(κ, θ, σ, r0)

Vasicek model: dr = κ(θ - r)dt + σdW

Parameters:
- κ: mean reversion speed
- θ: long-term mean rate
- σ: volatility
- r0: initial short rate
"""
struct Vasicek <: ShortRateModel
    κ::Float64
    θ::Float64
    σ::Float64
    r0::Float64
end

"""
    CIR(κ, θ, σ, r0)

Cox-Ingersoll-Ross model: dr = κ(θ - r)dt + σ√r dW

Feller condition: 2κθ > σ² ensures r stays positive.
"""
struct CIR <: ShortRateModel
    κ::Float64
    θ::Float64
    σ::Float64
    r0::Float64

    function CIR(κ, θ, σ, r0)
        if 2κ * θ <= σ^2
            @warn "Feller condition violated: 2κθ = $(2κ*θ) ≤ σ² = $(σ^2). Rate may hit zero."
        end
        new(κ, θ, σ, r0)
    end
end

"""
    HullWhite(κ, σ, curve)

Hull-White model: dr = (θ(t) - κr)dt + σdW

Time-dependent θ(t) calibrated to fit initial term structure.
"""
struct HullWhite <: ShortRateModel
    κ::Float64
    σ::Float64
    curve::RateCurve  # Initial term structure to fit
end

# Analytical bond prices under short rate models

"""
    bond_price(model, T) -> Float64

Zero-coupon bond price P(0,T) under the short rate model.
"""
function bond_price(m::Vasicek, T::Float64)
    κ, θ, σ, r0 = m.κ, m.θ, m.σ, m.r0

    B = (1 - exp(-κ * T)) / κ
    A = exp((θ - σ^2 / (2κ^2)) * (B - T) - σ^2 * B^2 / (4κ))

    A * exp(-B * r0)
end

function bond_price(m::CIR, T::Float64)
    κ, θ, σ, r0 = m.κ, m.θ, m.σ, m.r0

    γ = sqrt(κ^2 + 2σ^2)

    num = 2γ * exp((κ + γ) * T / 2)
    den = (κ + γ) * (exp(γ * T) - 1) + 2γ

    A = (num / den)^(2κ * θ / σ^2)
    B = 2(exp(γ * T) - 1) / den

    A * exp(-B * r0)
end

function bond_price(m::HullWhite, T::Float64)
    κ, σ = m.κ, m.σ

    # Use market discount factor and adjust
    P_market = discount(m.curve, T)
    f0 = instantaneous_forward(m.curve, 0.0)

    B = (1 - exp(-κ * T)) / κ

    # Under HW, P(0,T) = P_market(0,T) * exp(-B*r0 + B*f0 + σ²B²(1-exp(-2κT))/(4κ))
    # Simplified: we fit to market, so return market price
    P_market
end

"""
    short_rate(model, t) -> (mean, variance)

Expected short rate and variance at time t.
"""
function short_rate(m::Vasicek, t::Float64)
    κ, θ, σ, r0 = m.κ, m.θ, m.σ, m.r0

    mean = θ + (r0 - θ) * exp(-κ * t)
    var = σ^2 * (1 - exp(-2κ * t)) / (2κ)

    (mean, var)
end

function short_rate(m::CIR, t::Float64)
    κ, θ, σ, r0 = m.κ, m.θ, m.σ, m.r0

    mean = θ + (r0 - θ) * exp(-κ * t)
    var = r0 * σ^2 * exp(-κ * t) * (1 - exp(-κ * t)) / κ +
          θ * σ^2 * (1 - exp(-κ * t))^2 / (2κ)

    (mean, var)
end

"""
    simulate_short_rate(model, T, n_steps, n_paths) -> Matrix

Simulate short rate paths. Returns [n_steps+1 × n_paths] matrix.
"""
function simulate_short_rate(m::Vasicek, T::Float64, n_steps::Int, n_paths::Int)
    κ, θ, σ, r0 = m.κ, m.θ, m.σ, m.r0
    dt = T / n_steps

    paths = zeros(n_steps + 1, n_paths)
    paths[1, :] .= r0

    sqrt_dt = sqrt(dt)
    for i in 1:n_steps
        dW = randn(n_paths) * sqrt_dt
        paths[i+1, :] = paths[i, :] + κ * (θ .- paths[i, :]) * dt + σ * dW
    end

    paths
end

function simulate_short_rate(m::CIR, T::Float64, n_steps::Int, n_paths::Int)
    κ, θ, σ, r0 = m.κ, m.θ, m.σ, m.r0
    dt = T / n_steps

    paths = zeros(n_steps + 1, n_paths)
    paths[1, :] .= r0

    sqrt_dt = sqrt(dt)
    for i in 1:n_steps
        dW = randn(n_paths) * sqrt_dt
        r = paths[i, :]
        # Full truncation scheme
        r_pos = max.(r, 0.0)
        paths[i+1, :] = r + κ * (θ .- r_pos) * dt + σ * sqrt.(r_pos) .* dW
        paths[i+1, :] = max.(paths[i+1, :], 0.0)
    end

    paths
end

function simulate_short_rate(m::HullWhite, T::Float64, n_steps::Int, n_paths::Int)
    κ, σ = m.κ, m.σ
    dt = T / n_steps

    # θ(t) chosen to fit initial curve
    function theta(t)
        f = instantaneous_forward(m.curve, t)
        df_dt = (instantaneous_forward(m.curve, t + 1e-6) - f) / 1e-6
        df_dt + κ * f + σ^2 * (1 - exp(-2κ * t)) / (2κ)
    end

    r0 = instantaneous_forward(m.curve, 0.0)
    paths = zeros(n_steps + 1, n_paths)
    paths[1, :] .= r0

    sqrt_dt = sqrt(dt)
    for i in 1:n_steps
        t = (i - 1) * dt
        dW = randn(n_paths) * sqrt_dt
        paths[i+1, :] = paths[i, :] + (theta(t) .- κ * paths[i, :]) * dt + σ * dW
    end

    paths
end

# ============================================================================
# Interest Rate Derivatives
# ============================================================================

"""
    FRA(start_date, end_date, fixed_rate, index; notional=1.0, pay_fixed=true)

Forward rate agreement on a single period.
"""
struct FRA
    start_date::Date
    end_date::Date
    fixed_rate::Float64
    index::RateIndex
    notional::Float64
    pay_fixed::Bool
end

FRA(start_date, end_date, fixed_rate, index; notional=1.0, pay_fixed=true) =
    FRA(start_date, end_date, fixed_rate, index, notional, pay_fixed)

"""
    FixedLeg(schedule, rate; day_count=Thirty360(), notional=1.0, pay=true, payment_lag=Day(0))
"""
struct FixedLeg
    schedule::Schedule
    rate::Float64
    day_count::DayCountConvention
    notional::Float64
    pay::Bool
    payment_lag::Period
end

FixedLeg(schedule, rate; day_count=Thirty360(), notional=1.0, pay=true, payment_lag=Day(0)) =
    FixedLeg(schedule, rate, day_count, notional, pay, payment_lag)

"""
    FloatLeg(schedule, index; spread=0.0, notional=1.0, pay=false)
"""
struct FloatLeg
    schedule::Schedule
    index::RateIndex
    spread::Float64
    notional::Float64
    pay::Bool
end

FloatLeg(schedule, index; spread=0.0, notional=1.0, pay=false) =
    FloatLeg(schedule, index, spread, notional, pay)

"""
    Swap(fixed_leg, float_leg)

Plain-vanilla fixed/float interest rate swap.
"""
struct Swap
    fixed_leg::FixedLeg
    float_leg::FloatLeg
end

"""
    cashflows(...)

Return schedule-based cashflows for swap legs and FRAs.
"""
function cashflows end

"""
    Swap(start_date, end_date, fixed_rate, index; tenor=Month(6), calendar=WeekendCalendar(),
         bdc=ModifiedFollowing(), fixed_day_count=Thirty360(), float_spread=0.0,
         notional=1.0, pay_fixed=true)

Convenience constructor for a standard fixed/float swap sharing one schedule.
"""
function Swap(
    start_date::Date,
    end_date::Date,
    fixed_rate::Float64,
    index::RateIndex;
    tenor::Period=Month(6),
    calendar::Calendar=WeekendCalendar(),
    bdc::BusinessDayConvention=ModifiedFollowing(),
    fixed_day_count::DayCountConvention=Thirty360(),
    float_spread::Float64=0.0,
    notional::Float64=1.0,
    pay_fixed::Bool=true
)
    sched = Schedule(start_date, end_date; tenor=tenor, calendar=calendar, bdc=bdc)
    fixed_leg = FixedLeg(sched, fixed_rate; day_count=fixed_day_count, notional=notional, pay=pay_fixed)
    float_leg = FloatLeg(sched, index; spread=float_spread, notional=notional, pay=!pay_fixed)
    Swap(fixed_leg, float_leg)
end

function _fixed_leg_cashflows(leg::FixedLeg)
    flows = Tuple{Date,Float64}[]
    sign = leg.pay ? -1.0 : 1.0
    for (d1, d2) in schedule_periods(leg.schedule)
        accrual = year_fraction(d1, d2, leg.day_count)
        pay_date = adjust_date(d2 + leg.payment_lag, leg.schedule.calendar, leg.schedule.bdc)
        amount = sign * leg.notional * leg.rate * accrual
        push!(flows, (pay_date, amount))
    end
    flows
end

cashflows(leg::FixedLeg) = _fixed_leg_cashflows(leg)

function _float_leg_cashflows(leg::FloatLeg, cs::CurveSet)
    flows = Tuple{Date,Float64}[]
    sign = leg.pay ? -1.0 : 1.0
    for (d1, d2) in schedule_periods(leg.schedule)
        accrual = year_fraction(d1, d2, leg.index.day_count)
        rate = forward_rate(cs, leg.index, d1, d2) + leg.spread
        pay_date = payment_date(leg.index, d2)
        amount = sign * leg.notional * rate * accrual
        push!(flows, (pay_date, amount))
    end
    flows
end

cashflows(leg::FloatLeg, cs::CurveSet) = _float_leg_cashflows(leg, cs)

function cashflows(s::Swap, cs::CurveSet)
    (fixed=cashflows(s.fixed_leg), float=cashflows(s.float_leg, cs))
end

"""
    price(fra, curveset) -> Float64

Present value of a FRA using a CurveSet.
"""
function price(fra::FRA, cs::CurveSet)
    accrual = year_fraction(fra.start_date, fra.end_date, fra.index.day_count)
    fwd = forward_rate(cs, fra.index, fra.start_date, fra.end_date)
    pay_date = payment_date(fra.index, fra.end_date)
    df = discount(cs, pay_date; day_count=cs.day_count)
    sign = fra.pay_fixed ? 1.0 : -1.0
    sign * (fwd - fra.fixed_rate) * accrual * fra.notional * df
end

function cashflows(fra::FRA, cs::CurveSet)
    accrual = year_fraction(fra.start_date, fra.end_date, fra.index.day_count)
    fwd = forward_rate(cs, fra.index, fra.start_date, fra.end_date)
    pay_date = payment_date(fra.index, fra.end_date)
    sign = fra.pay_fixed ? 1.0 : -1.0
    amount = sign * (fwd - fra.fixed_rate) * accrual * fra.notional
    (pay_date, amount)
end

"""
    price(fixed_leg, curveset) -> Float64
"""
function price(leg::FixedLeg, cs::CurveSet)
    sum(amount * discount(cs, d; day_count=cs.day_count) for (d, amount) in _fixed_leg_cashflows(leg))
end

"""
    price(float_leg, curveset) -> Float64
"""
function price(leg::FloatLeg, cs::CurveSet)
    sum(amount * discount(cs, d; day_count=cs.day_count) for (d, amount) in _float_leg_cashflows(leg, cs))
end

"""
    price(swap, curveset) -> Float64
"""
price(s::Swap, cs::CurveSet) = price(s.fixed_leg, cs) + price(s.float_leg, cs)

"""
    par_swap_rate(swap, curveset) -> Float64

Fixed rate that makes the swap PV zero (ignores fixed leg sign).
"""
function par_swap_rate(s::Swap, cs::CurveSet)
    annuity = sum(
        year_fraction(d1, d2, s.fixed_leg.day_count) *
        discount(cs, adjust_date(d2 + s.fixed_leg.payment_lag, s.fixed_leg.schedule.calendar, s.fixed_leg.schedule.bdc);
                 day_count=cs.day_count)
        for (d1, d2) in schedule_periods(s.fixed_leg.schedule)
    )

    float_pv = sum(
        (forward_rate(cs, s.float_leg.index, d1, d2) + s.float_leg.spread) *
        year_fraction(d1, d2, s.float_leg.index.day_count) *
        discount(cs, payment_date(s.float_leg.index, d2); day_count=cs.day_count)
        for (d1, d2) in schedule_periods(s.float_leg.schedule)
    )

    float_pv / annuity
end

"""Caplet: call option on forward rate"""
struct Caplet
    start::Float64      # Start of rate period
    maturity::Float64   # End of rate period (payment date)
    strike::Float64     # Strike rate
    notional::Float64
end
Caplet(start, mat, strike) = Caplet(start, mat, strike, 1.0)

"""Floorlet: put option on forward rate"""
struct Floorlet
    start::Float64
    maturity::Float64
    strike::Float64
    notional::Float64
end
Floorlet(start, mat, strike) = Floorlet(start, mat, strike, 1.0)

"""Cap: portfolio of caplets"""
struct Cap
    maturity::Float64
    strike::Float64
    frequency::Int
    notional::Float64
end
Cap(mat, strike) = Cap(mat, strike, 4, 1.0)

"""Floor: portfolio of floorlets"""
struct Floor
    maturity::Float64
    strike::Float64
    frequency::Int
    notional::Float64
end
Floor(mat, strike) = Floor(mat, strike, 4, 1.0)

"""
    Swaption(expiry, swap_maturity, strike, is_payer, notional)

European swaption - option to enter a swap.
"""
struct Swaption
    expiry::Float64         # Option expiry
    swap_maturity::Float64  # Underlying swap maturity
    strike::Float64         # Strike swap rate
    is_payer::Bool          # true = payer swaption
    frequency::Int
    notional::Float64
end
Swaption(exp, mat, strike, is_payer) = Swaption(exp, mat, strike, is_payer, 2, 1.0)

# TODO: Export black_swaption function (currently only price() method exists)

"""
    black_caplet(caplet, curve, volatility) -> Float64

Price a caplet using Black's formula.
"""
function black_caplet(c::Caplet, curve::RateCurve, σ::Float64)
    τ = c.maturity - c.start
    F = forward_rate(curve, c.start, c.maturity)
    df = discount(curve, c.maturity)

    if c.start <= 0
        # Already started, intrinsic value only
        return c.notional * τ * df * max(F - c.strike, 0)
    end

    d1 = (log(F / c.strike) + 0.5 * σ^2 * c.start) / (σ * sqrt(c.start))
    d2 = d1 - σ * sqrt(c.start)

    N = Normal()
    c.notional * τ * df * (F * cdf(N, d1) - c.strike * cdf(N, d2))
end

"""
    black_floorlet(floorlet, curve, volatility) -> Float64

Price a floorlet using Black's formula.
"""
function black_floorlet(f::Floorlet, curve::RateCurve, σ::Float64)
    τ = f.maturity - f.start
    F = forward_rate(curve, f.start, f.maturity)
    df = discount(curve, f.maturity)

    if f.start <= 0
        return f.notional * τ * df * max(f.strike - F, 0)
    end

    d1 = (log(F / f.strike) + 0.5 * σ^2 * f.start) / (σ * sqrt(f.start))
    d2 = d1 - σ * sqrt(f.start)

    N = Normal()
    f.notional * τ * df * (f.strike * cdf(N, -d2) - F * cdf(N, -d1))
end

"""
    black_cap(cap, curve, volatilities) -> Float64

Price a cap as sum of caplets.
volatilities can be scalar (flat) or vector (per caplet).
"""
function black_cap(cap::Cap, curve::RateCurve, σ)
    τ = 1.0 / cap.frequency
    n = Int(cap.maturity * cap.frequency)

    total = 0.0
    for i in 1:n
        start = (i - 1) * τ
        mat = i * τ
        vol = σ isa Vector ? σ[i] : σ
        caplet = Caplet(start, mat, cap.strike, cap.notional)
        total += black_caplet(caplet, curve, vol)
    end
    total
end

"""
    black_floor(floor, curve, volatilities) -> Float64

Price a floor as sum of floorlets.
"""
function black_floor(floor::Floor, curve::RateCurve, σ)
    τ = 1.0 / floor.frequency
    n = Int(floor.maturity * floor.frequency)

    total = 0.0
    for i in 1:n
        start = (i - 1) * τ
        mat = i * τ
        vol = σ isa Vector ? σ[i] : σ
        floorlet = Floorlet(start, mat, floor.strike, floor.notional)
        total += black_floorlet(floorlet, curve, vol)
    end
    total
end

"""
    price(swaption, curve, volatility) -> Float64

Price a European swaption using Black's formula.
"""
function price(s::Swaption, curve::RateCurve, σ::Float64)
    # Annuity factor
    τ = 1.0 / s.frequency
    n = Int((s.swap_maturity - s.expiry) * s.frequency)

    A = sum(τ * discount(curve, s.expiry + i * τ) for i in 1:n)

    # Forward swap rate
    df_start = discount(curve, s.expiry)
    df_end = discount(curve, s.swap_maturity)
    S = (df_start - df_end) / A

    # Black's formula
    d1 = (log(S / s.strike) + 0.5 * σ^2 * s.expiry) / (σ * sqrt(s.expiry))
    d2 = d1 - σ * sqrt(s.expiry)

    N = Normal()
    if s.is_payer
        s.notional * A * (S * cdf(N, d1) - s.strike * cdf(N, d2))
    else
        s.notional * A * (s.strike * cdf(N, -d2) - S * cdf(N, -d1))
    end
end

# ============================================================================
# Curve Risk (Bucketed PV01)
# ============================================================================

_curve_nodes(curve::DiscountCurve) = curve.times, curve.values
_curve_nodes(curve::ZeroCurve) = curve.times, curve.values
_curve_nodes(curve::ForwardCurve) = curve.times, curve.values
_curve_nodes(::RateCurve) = throw(ArgumentError("Bucketed risk only supported for node-based curves"))

function _bump_curve(curve::DiscountCurve, idx::Int, bump::Float64; method::Symbol=:zero)
    times = copy(curve.times)
    values = copy(curve.values)
    t = times[idx]
    if t <= 0
        return curve
    end

    if method == :df
        # Bump discount factor via equivalent rate bump
        values[idx] *= exp(-bump * t)
    elseif method == :zero
        zr = -log(values[idx]) / t
        zr += bump
        values[idx] = exp(-zr * t)
    else
        throw(ArgumentError("Unknown bump method: $method"))
    end

    DiscountCurve(times, values; interp=curve.interp)
end

function _bump_curve(curve::ZeroCurve, idx::Int, bump::Float64; method::Symbol=:zero)
    times = copy(curve.times)
    values = copy(curve.values)
    values[idx] += bump
    ZeroCurve(times, values; interp=curve.interp)
end

function _bump_curve(curve::ForwardCurve, idx::Int, bump::Float64; method::Symbol=:forward)
    times = copy(curve.times)
    values = copy(curve.values)
    values[idx] += bump
    ForwardCurve(times, values; interp=curve.interp)
end

"""
    bucketed_pv01(instrument, curve; bump=1e-4, method=:zero, price_fn=price)

Compute bucketed PV01 by bumping each curve node by `bump` and repricing.
Returns a named tuple `(base, bump, buckets)` where buckets map time->ΔPV.
"""
function bucketed_pv01(
    instrument,
    curve::RateCurve;
    bump::Float64=1e-4,
    method::Symbol=:zero,
    price_fn::Function=price
)
    bucketed_pv01(c -> price_fn(instrument, c), curve; bump=bump, method=method)
end

"""
    bucketed_pv01(price_fn, curve; bump=1e-4, method=:zero)

Bucketed PV01 for any pricing function of a curve.
"""
function bucketed_pv01(
    price_fn::Function,
    curve::RateCurve;
    bump::Float64=1e-4,
    method::Symbol=:zero
)
    times, _ = _curve_nodes(curve)
    base = price_fn(curve)
    buckets = Dict{Float64,Float64}()

    for i in eachindex(times)
        t = times[i]
        t <= 0 && continue
        bumped_curve = _bump_curve(curve, i, bump; method=method)
        pv = price_fn(bumped_curve)
        buckets[t] = pv - base
    end

    (base=base, bump=bump, buckets=buckets)
end

"""
    bucketed_pv01(price_fn, curveset; bump=1e-4, method=:zero, curve_ids=nothing)

Bucketed PV01 across multiple curves in a CurveSet.
Returns a named tuple `(base, bump, curves)` where curves map curve_id->buckets.
"""
function bucketed_pv01(
    price_fn::Function,
    cs::CurveSet;
    bump::Float64=1e-4,
    method::Symbol=:zero,
    curve_ids=nothing
)
    base = price_fn(cs)
    ids = curve_ids === nothing ? vcat([:discount], collect(keys(cs.forwards))) : curve_ids
    curves = Dict{Symbol,Dict{Float64,Float64}}()

    for id in ids
        curve = id == :discount ? cs.discount : cs.forwards[id]
        times, _ = _curve_nodes(curve)
        buckets = Dict{Float64,Float64}()
        for i in eachindex(times)
            t = times[i]
            t <= 0 && continue
            bumped_curve = _bump_curve(curve, i, bump; method=method)
            cs_bumped = id == :discount ? _with_discount(cs, bumped_curve) : _with_forward(cs, id, bumped_curve)
            pv = price_fn(cs_bumped)
            buckets[t] = pv - base
        end
        curves[id] = buckets
    end

    (base=base, bump=bump, curves=curves)
end

function bucketed_pv01(
    instrument,
    cs::CurveSet;
    bump::Float64=1e-4,
    method::Symbol=:zero,
    curve_ids=nothing,
    price_fn::Function=price
)
    bucketed_pv01(c -> price_fn(instrument, c), cs; bump=bump, method=method, curve_ids=curve_ids)
end

end # module
