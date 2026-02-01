# Interest Rates

SuperNova provides a comprehensive interest rate modeling framework including yield curves, bonds, short-rate models, and interest rate derivatives.

## Yield Curves

### Curve Types

Three curve representations are available, all convertible to each other:

```julia
using SuperNova

# Discount curve - stores discount factors directly
dc = DiscountCurve([0.0, 1.0, 2.0, 5.0], [1.0, 0.95, 0.90, 0.78])

# Zero curve - stores continuously compounded zero rates
zc = ZeroCurve([0.0, 1.0, 2.0, 5.0], [0.05, 0.051, 0.052, 0.05])

# Forward curve - stores instantaneous forward rates
fc = ForwardCurve([0.0, 1.0, 2.0, 5.0], [0.05, 0.052, 0.054, 0.05])

# Flat curve shortcuts
flat_dc = DiscountCurve(0.05)  # 5% flat rate
flat_zc = ZeroCurve(0.05)
```

### Curve Operations

All curve types support the same interface:

```julia
# Discount factor from 0 to T
discount(curve, 2.0)  # P(0, 2)

# Zero rate to time T
zero_rate(curve, 2.0)  # Continuously compounded

# Forward rate between T1 and T2 (simply compounded)
forward_rate(curve, 1.0, 2.0)  # F(1, 2)

# Instantaneous forward rate at T
instantaneous_forward(curve, 1.0)  # f(1)
```

### Interpolation Methods

Curves support different interpolation methods:

```julia
# Linear interpolation (default for zero/forward curves)
zc = ZeroCurve(times, rates; interp=LinearInterp())

# Log-linear interpolation (default for discount curves)
dc = DiscountCurve(times, dfs; interp=LogLinearInterp())

# Cubic spline interpolation (smooth forwards)
zc = ZeroCurve(times, rates; interp=CubicSplineInterp())
```

### Curve Bootstrapping

Build curves from market instruments:

```julia
# Market instruments
instruments = [
    DepositRate(0.25, 0.05),           # 3M deposit at 5%
    DepositRate(0.5, 0.051),           # 6M deposit at 5.1%
    FuturesRate(0.5, 0.75, 0.052),     # Futures: 6M-9M at 5.2%
    SwapRate(2.0, 0.053),              # 2Y swap at 5.3%
    SwapRate(5.0, 0.055),              # 5Y swap at 5.5%
    SwapRate(10.0, 0.056),             # 10Y swap at 5.6%
]

# Bootstrap discount curve
curve = bootstrap(instruments)

# With specific interpolation
curve = bootstrap(instruments; interp=CubicSplineInterp())
```

## Bond Pricing

### Bond Types

```julia
# Zero-coupon bond
zcb = ZeroCouponBond(5.0)              # 5-year, face value 100
zcb = ZeroCouponBond(5.0, 1000.0)      # Custom face value

# Fixed-rate coupon bond
frb = FixedRateBond(5.0, 0.06)         # 5-year, 6% coupon, semi-annual
frb = FixedRateBond(5.0, 0.06, 4)      # Quarterly coupons
frb = FixedRateBond(5.0, 0.06, 2, 1000.0)  # Custom face value

# Floating-rate bond
flb = FloatingRateBond(5.0, 0.01)      # 5-year, LIBOR + 100bp spread
```

### Pricing and Analytics

```julia
bond = FixedRateBond(5.0, 0.06)
curve = ZeroCurve(0.05)

# Present value
pv = price(bond, curve)

# Price at a specific yield
pv = price(bond, 0.055)

# Yield to maturity from market price
ytm = yield_to_maturity(bond, 102.5)

# Duration measures
mac_dur = duration(bond, 0.05)          # Macaulay duration
mod_dur = modified_duration(bond, 0.05) # Modified duration

# Convexity
conv = convexity(bond, 0.05)

# DV01 (dollar value of 1 basis point)
dv = dv01(bond, 0.05)

# Accrued interest and clean/dirty prices
ai = accrued_interest(bond, 0.25)       # Settlement 3 months in
clean = clean_price(bond, curve, 0.25)
dirty = dirty_price(bond, curve)
```

## Short-Rate Models

### Vasicek Model

Mean-reverting Gaussian short rate: `dr = κ(θ - r)dt + σdW`

```julia
# Parameters: mean reversion (κ), long-term mean (θ), volatility (σ), initial rate (r0)
vasicek = Vasicek(0.5, 0.05, 0.01, 0.03)

# Analytical zero-coupon bond price
P = bond_price(vasicek, 5.0)  # P(0, 5)

# Expected rate and variance at time t
mean_r, var_r = short_rate(vasicek, 2.0)

# Simulate paths
paths = simulate_short_rate(vasicek, 5.0, 252, 1000)  # T=5, daily steps, 1000 paths
# Returns [253 × 1000] matrix
```

### CIR Model

Mean-reverting with square-root volatility: `dr = κ(θ - r)dt + σ√r dW`

Ensures non-negative rates when Feller condition holds: `2κθ > σ²`

```julia
cir = CIR(0.5, 0.05, 0.1, 0.03)

# Same interface as Vasicek
P = bond_price(cir, 5.0)
mean_r, var_r = short_rate(cir, 2.0)
paths = simulate_short_rate(cir, 5.0, 252, 1000)
```

### Hull-White Model

Time-dependent mean reversion calibrated to market curve: `dr = (θ(t) - κr)dt + σdW`

```julia
# Fit to market term structure
market_curve = ZeroCurve([0.0, 1.0, 2.0, 5.0, 10.0], [0.03, 0.035, 0.04, 0.045, 0.05])
hw = HullWhite(0.1, 0.01, market_curve)

# Bond prices match market exactly
P = bond_price(hw, 5.0)  # Equals discount(market_curve, 5.0)

# Simulate paths consistent with initial curve
paths = simulate_short_rate(hw, 5.0, 252, 1000)
```

## Interest Rate Derivatives

### Caps and Floors

Options on forward rates:

```julia
curve = ZeroCurve(0.05)
cap_vol = 0.20  # 20% Black vol

# Individual caplet/floorlet
caplet = Caplet(1.0, 1.25, 0.05)  # Start=1Y, End=1.25Y, Strike=5%
price_caplet = black_caplet(caplet, curve, cap_vol)

floorlet = Floorlet(1.0, 1.25, 0.04)  # Strike=4%
price_floorlet = black_floorlet(floorlet, curve, cap_vol)

# Full cap (portfolio of caplets)
cap = Cap(5.0, 0.05)              # 5-year cap, strike 5%, quarterly
price_cap = black_cap(cap, curve, cap_vol)

# With term structure of vols (one per caplet)
vols = [0.18, 0.19, 0.20, 0.21, ...]  # Per-period vols
price_cap = black_cap(cap, curve, vols)

# Floor
floor = Floor(5.0, 0.04)
price_floor = black_floor(floor, curve, cap_vol)
```

### Swaptions

Options to enter interest rate swaps:

```julia
curve = ZeroCurve(0.05)
swaption_vol = 0.15  # 15% Black vol

# Payer swaption: option to pay fixed, receive floating
payer = Swaption(1.0, 6.0, 0.05, true)  # 1Y expiry into 5Y swap, strike 5%
price_payer = price(payer, curve, swaption_vol)

# Receiver swaption: option to receive fixed, pay floating
receiver = Swaption(1.0, 6.0, 0.05, false)
price_receiver = price(receiver, curve, swaption_vol)

# With custom frequency and notional
swaption = Swaption(1.0, 6.0, 0.05, true, 4, 1_000_000.0)  # Quarterly, $1M notional
```

## Example: Building a Rate Curve and Pricing

```julia
using SuperNova

# Bootstrap curve from market data
instruments = [
    DepositRate(0.25, 0.048),
    DepositRate(0.5, 0.049),
    SwapRate(1.0, 0.050),
    SwapRate(2.0, 0.052),
    SwapRate(5.0, 0.055),
    SwapRate(10.0, 0.058),
]
curve = bootstrap(instruments)

# Price a corporate bond
bond = FixedRateBond(5.0, 0.06, 2)  # 5Y, 6% semi-annual
pv = price(bond, curve)
ytm = yield_to_maturity(bond, pv)
dur = modified_duration(bond, ytm)

println("Bond PV: \$$(round(pv, digits=2))")
println("YTM: $(round(ytm*100, digits=2))%")
println("Modified Duration: $(round(dur, digits=2)) years")

# Price an interest rate cap
cap = Cap(5.0, 0.055)  # 5Y cap at 5.5%
cap_price = black_cap(cap, curve, 0.20)
println("Cap Price: \$$(round(cap_price*10000, digits=2)) per \$10K notional")

# Simulate rates under Vasicek
model = Vasicek(0.3, 0.05, 0.015, curve.values[1])
paths = simulate_short_rate(model, 5.0, 60, 10000)  # Monthly for 5 years
terminal_rates = paths[end, :]
println("5Y rate distribution: mean=$(round(mean(terminal_rates)*100, digits=2))%, std=$(round(std(terminal_rates)*100, digits=2))%")
```
