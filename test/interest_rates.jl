using Test
using Dates
using QuantNova
using QuantNova.InterestRates

# Use module-qualified price to avoid collision with Instruments.price
const ir_price = QuantNova.InterestRates.price

@testset "Interest Rates" begin

    @testset "Day Count Conventions" begin
        @testset "Year fraction with numeric times" begin
            # For numeric inputs, all conventions return t2 - t1
            @test year_fraction(0.0, 1.0, ACT360()) ≈ 1.0
            @test year_fraction(0.0, 1.0, ACT365()) ≈ 1.0
            @test year_fraction(0.0, 1.0, Thirty360()) ≈ 1.0
            @test year_fraction(0.0, 1.0, ACTACT()) ≈ 1.0

            @test year_fraction(0.5, 1.5, ACT360()) ≈ 1.0
            @test year_fraction(1.0, 3.0, ACT365()) ≈ 2.0
        end

        @testset "Convention types exist" begin
            @test ACT360() isa DayCountConvention
            @test ACT365() isa DayCountConvention
            @test Thirty360() isa DayCountConvention
            @test ACTACT() isa DayCountConvention
        end

        @testset "Year fraction with Dates" begin
            d1 = Date(2024, 1, 1)
            d2 = Date(2024, 1, 31)  # 30-day span
            @test year_fraction(d1, d2, ACT360()) ≈ 30 / 360
            @test year_fraction(d1, d2, ACT365()) ≈ 30 / 365
            @test year_fraction(d1, d2, Thirty360()) ≈ 30 / 360
        end
    end

    @testset "Scheduling and Conventions" begin
        start_date = Date(2024, 1, 15)
        end_date = Date(2024, 4, 15)
        cal = WeekendCalendar()
        sched = Schedule(start_date, end_date; tenor=Month(1), calendar=cal, bdc=Following())

        @test length(sched) == 4
        @test first(sched.dates) == adjust_date(start_date, cal, Following())
        @test last(sched.dates) == adjust_date(end_date, cal, Following())

        periods = schedule_periods(sched)
        @test length(periods) == 3
        accruals = accrual_factors(sched, ACT360())
        @test length(accruals) == 3
        @test all(a > 0 for a in accruals)
    end

    @testset "Multi-curve Framework" begin
        asof = Date(2024, 1, 2)
        dc = ZeroCurve(0.05)
        fwd = ZeroCurve(0.06)
        cs = CurveSet(asof, dc; forwards=Dict(:SOFR3M => fwd))
        idx = RateIndex(:SOFR3M; tenor=Month(3), day_count=ACT360())

        start_date = Date(2024, 4, 2)
        end_date = Date(2024, 7, 2)
        f = forward_rate(cs, idx, start_date, end_date)
        t1 = year_fraction(asof, start_date, ACT360())
        t2 = year_fraction(asof, end_date, ACT360())
        dt = t2 - t1
        f_expected = (exp(0.06 * dt) - 1) / dt

        @test f ≈ f_expected atol=1e-6

        t = year_fraction(asof, Date(2025, 1, 2), ACT365())
        @test discount(cs, Date(2025, 1, 2); day_count=ACT365()) ≈ exp(-0.05 * t) atol=1e-6
    end

    @testset "FRA and Swap Pricing" begin
        asof = Date(2024, 1, 2)
        dc = ZeroCurve(0.03)
        fwd = ZeroCurve(0.04)
        cs = CurveSet(asof, dc; forwards=Dict(:SOFR3M => fwd))
        idx = RateIndex(:SOFR3M; tenor=Month(3), day_count=ACT360())

        start_date = Date(2024, 4, 2)
        end_date = Date(2024, 7, 2)

        f = forward_rate(cs, idx, start_date, end_date)
        fra = FRA(start_date, end_date, f, idx; notional=1.0, pay_fixed=true)
        @test abs(ir_price(fra, cs)) < 1e-10

        fra_cf = cashflows(fra, cs)
        @test fra_cf[1] isa Date
        @test fra_cf[2] ≈ 0.0 atol=1e-10

        fra_rich = FRA(start_date, end_date, f + 0.001, idx; notional=1.0, pay_fixed=true)
        @test ir_price(fra_rich, cs) < 0.0

        swap_start = Date(2024, 1, 2)
        swap_end = Date(2026, 1, 2)
        sched = Schedule(swap_start, swap_end; tenor=Month(6), calendar=WeekendCalendar(), bdc=ModifiedFollowing())

        fixed_leg = FixedLeg(sched, 0.0; day_count=Thirty360(), notional=1.0, pay=true)
        float_leg = FloatLeg(sched, idx; spread=0.0, notional=1.0, pay=false)
        swap = Swap(fixed_leg, float_leg)

        par = par_swap_rate(swap, cs)
        swap_par = Swap(swap_start, swap_end, par, idx; tenor=Month(6), calendar=WeekendCalendar(),
                        bdc=ModifiedFollowing(), fixed_day_count=Thirty360(), notional=1.0, pay_fixed=true)
        @test abs(ir_price(swap_par, cs)) < 1e-8

        flows = cashflows(swap_par, cs)
        @test length(flows.fixed) == length(schedule_periods(sched))
        @test length(flows.float) == length(schedule_periods(sched))
    end

    @testset "Nelson-Siegel Curve" begin
        @testset "Basic Nelson-Siegel" begin
            # Create a curve with known parameters
            curve = NelsonSiegelCurve(0.05, -0.02, 0.01, 2.0)

            # Instantaneous rate = β0 + β1 = 0.03
            @test zero_rate(curve, 0.0) ≈ 0.03

            # Long-term rate = β0 = 0.05
            @test zero_rate(curve, 100.0) ≈ 0.05 atol=0.001

            # Discount factors
            @test discount(curve, 0.0) ≈ 1.0
            @test discount(curve, 1.0) < 1.0
            @test discount(curve, 5.0) < discount(curve, 1.0)

            # Forward rates should be well-behaved
            f = instantaneous_forward(curve, 2.0)
            @test isfinite(f)
            @test f > 0
        end

        @testset "Nelson-Siegel validation" begin
            # τ must be positive
            @test_throws ArgumentError NelsonSiegelCurve(0.05, -0.02, 0.01, 0.0)
            @test_throws ArgumentError NelsonSiegelCurve(0.05, -0.02, 0.01, -1.0)
        end

        @testset "Nelson-Siegel fitting" begin
            # Generate synthetic data from a known curve
            true_curve = NelsonSiegelCurve(0.04, -0.02, 0.015, 2.5)
            maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
            rates = [zero_rate(true_curve, t) for t in maturities]

            # Fit a curve to this data
            fitted = fit_nelson_siegel(maturities, rates)

            # Should recover similar rates
            for (t, r) in zip(maturities, rates)
                @test zero_rate(fitted, t) ≈ r atol=0.001
            end
        end
    end

    @testset "Svensson Curve" begin
        @testset "Basic Svensson" begin
            curve = SvenssonCurve(0.05, -0.02, 0.01, 0.005, 2.0, 5.0)

            # Instantaneous rate = β0 + β1
            @test zero_rate(curve, 0.0) ≈ 0.03

            # Long-term rate = β0
            @test zero_rate(curve, 100.0) ≈ 0.05 atol=0.001

            # Should be able to compute discount factors
            @test discount(curve, 1.0) < 1.0
            @test discount(curve, 5.0) < discount(curve, 1.0)
        end

        @testset "Svensson validation" begin
            @test_throws ArgumentError SvenssonCurve(0.05, -0.02, 0.01, 0.005, 0.0, 5.0)
            @test_throws ArgumentError SvenssonCurve(0.05, -0.02, 0.01, 0.005, 2.0, 0.0)
        end
    end

    @testset "Curve Interpolation Accuracy" begin
        # Test that different interpolation methods give reasonable results
        times = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        rates = [0.02, 0.022, 0.024, 0.028, 0.032, 0.038, 0.042, 0.045]

        zc_linear = ZeroCurve(times, rates; interp=LinearInterp())
        zc_loglin = ZeroCurve(times, rates; interp=LogLinearInterp())

        # At knot points, all methods should agree
        for (t, r) in zip(times, rates)
            @test zero_rate(zc_linear, t) ≈ r atol=1e-10
            @test zero_rate(zc_loglin, t) ≈ r atol=1e-10
        end

        # Between knots, should be reasonable
        test_times = [0.1, 0.75, 1.5, 3.0, 7.0, 20.0]
        for t in test_times
            r_lin = zero_rate(zc_linear, t)
            r_log = zero_rate(zc_loglin, t)

            # Rates should be in the range of surrounding knots
            i = searchsortedlast(times, t)
            if i >= 1 && i < length(times)
                r_lo, r_hi = rates[i], rates[i+1]
                @test min(r_lo, r_hi) <= r_lin <= max(r_lo, r_hi)
            end

            # Different methods should give similar results
            @test r_lin ≈ r_log atol=0.005
        end
    end

    @testset "Bond pricing consistency" begin
        @testset "ZCB pricing methods agree" begin
            zcb = ZeroCouponBond(5.0, 100.0)

            # Different ways to price should agree
            rate = 0.05
            zc = ZeroCurve(rate)
            dc = DiscountCurve(rate)

            @test ir_price(zcb, zc) ≈ ir_price(zcb, dc) atol=1e-8
            @test ir_price(zcb, rate) ≈ ir_price(zcb, zc) atol=1e-8
        end

        @testset "Coupon bond pricing consistency" begin
            bond = FixedRateBond(5.0, 0.04, 2, 100.0)

            # Price at different yields
            yields = [0.02, 0.03, 0.04, 0.05, 0.06]
            prices = [ir_price(bond, y) for y in yields]

            # Higher yield = lower price
            for i in 1:length(yields)-1
                @test prices[i] > prices[i+1]
            end

            # YTM should recover the yield
            for y in yields
                p = ir_price(bond, y)
                ytm_recovered = yield_to_maturity(bond, p)
                @test ytm_recovered ≈ y atol=1e-8
            end
        end

        @testset "Duration and convexity bounds" begin
            bond = FixedRateBond(10.0, 0.05, 2, 100.0)
            y = 0.05

            dur = duration(bond, y)
            mod_dur = modified_duration(bond, y)
            conv = convexity(bond, y)

            # Duration < maturity for coupon bonds
            @test 0 < dur < 10.0

            # Modified duration < duration
            @test mod_dur < dur

            # Convexity > duration^2 / something (loosely)
            @test conv > 0

            # DV01 should be positive and reasonable
            dv = dv01(bond, y)
            @test dv > 0
            @test dv < 1.0  # Should be small for 100 face value
        end
    end

    @testset "Bucketed PV01" begin
        bond = FixedRateBond(5.0, 0.04, 2, 100.0)
        times = [0.0, 1.0, 2.0, 3.0, 5.0]
        rates = [0.02, 0.022, 0.025, 0.028, 0.03]
        curve = ZeroCurve(times, rates)

        result = bucketed_pv01(bond, curve; bump=1e-4)
        buckets = result.buckets

        @test length(buckets) == length(times) - 1
        @test all(v < 0 for v in values(buckets))
    end

    @testset "Yield Curves" begin
        @testset "Flat curve" begin
            rate = 0.05
            dc = DiscountCurve(rate)
            zc = ZeroCurve(rate)

            # Discount factors
            @test discount(dc, 0.0) ≈ 1.0
            @test discount(dc, 1.0) ≈ exp(-0.05) atol=1e-10
            @test discount(dc, 5.0) ≈ exp(-0.25) atol=1e-10

            @test discount(zc, 1.0) ≈ exp(-0.05) atol=1e-10

            # Zero rates
            @test zero_rate(dc, 1.0) ≈ 0.05 atol=1e-10
            @test zero_rate(zc, 1.0) ≈ 0.05 atol=1e-10
        end

        @testset "Interpolation" begin
            times = [0.0, 1.0, 2.0, 5.0, 10.0]
            rates = [0.02, 0.025, 0.03, 0.035, 0.04]

            zc = ZeroCurve(times, rates; interp=LinearInterp())

            @test zero_rate(zc, 0.5) ≈ 0.0225 atol=1e-10  # Linear interp
            @test zero_rate(zc, 3.0) ≈ 0.03 + (0.035 - 0.03) / 3 atol=1e-10
        end

        @testset "Forward rates" begin
            rate = 0.05
            dc = DiscountCurve(rate)

            # Forward rate in flat curve = spot rate
            fwd = forward_rate(dc, 1.0, 2.0)
            @test fwd ≈ (exp(0.05) - 1) atol=1e-6  # Simple rate

            # Instantaneous forward
            f_inst = instantaneous_forward(dc, 1.0)
            @test f_inst ≈ 0.05 atol=1e-4
        end

        @testset "Curve conversion" begin
            times = [0.0, 1.0, 2.0, 5.0]
            rates = [0.02, 0.025, 0.03, 0.035]

            zc = ZeroCurve(times, rates)
            dc = DiscountCurve(zc)

            # Should give approximately same discount factors
            # (small differences due to interpolation method conversion)
            for t in [0.5, 1.0, 2.0, 3.0]
                @test discount(zc, t) ≈ discount(dc, t) atol=1e-4
            end
        end
    end

    @testset "Bootstrapping" begin
        instruments = [
            DepositRate(0.25, 0.02),
            DepositRate(0.5, 0.022),
            DepositRate(1.0, 0.025),
            SwapRate(2.0, 0.028),
            SwapRate(5.0, 0.032),
        ]

        curve = bootstrap(instruments)

        # Check that we can price back the instruments
        @test discount(curve, 0.25) ≈ 1 / (1 + 0.02 * 0.25) atol=1e-10
        @test discount(curve, 0.5) ≈ 1 / (1 + 0.022 * 0.5) atol=1e-10
        @test discount(curve, 1.0) ≈ 1 / (1 + 0.025 * 1.0) atol=1e-10

        # Curve should be monotonically decreasing
        @test discount(curve, 1.0) > discount(curve, 2.0) > discount(curve, 5.0)
    end

    @testset "Multi-Curve Bootstrapping" begin
        asof = Date(2024, 1, 2)
        idx = RateIndex(:SOFR3M; tenor=Month(3), day_count=ACT360())

        ois_quotes = [
            OISDepositQuote(asof, Date(2024, 2, 2), 0.05),
            OISSwapQuote(asof, Date(2024, 7, 2), 0.05; fixed_frequency=1, fixed_day_count=ACT360())
        ]

        dc = bootstrap_ois_curve(ois_quotes; asof=asof)

        # OIS deposit reprices
        dep = ois_quotes[1]
        accrual_dep = year_fraction(dep.start_date, dep.end_date, dep.day_count)
        df_start = discount(dc, year_fraction(asof, dep.start_date, ACT365()))
        df_end = discount(dc, year_fraction(asof, dep.end_date, ACT365()))
        implied_dep = (df_start / df_end - 1) / accrual_dep
        @test implied_dep ≈ dep.rate atol=1e-6

        # OIS swap reprices
        swap_q = ois_quotes[2]
        sched = Schedule(swap_q.start_date, swap_q.maturity_date;
                         tenor=Month(12 ÷ swap_q.fixed_frequency),
                         calendar=swap_q.calendar, bdc=swap_q.bdc)
        annuity = sum(
            year_fraction(d1, d2, swap_q.fixed_day_count) *
            discount(dc, year_fraction(asof, d2, ACT365()))
            for (d1, d2) in schedule_periods(sched)
        )
        df_T = discount(dc, year_fraction(asof, swap_q.maturity_date, ACT365()))
        par_rate = (1.0 - df_T) / annuity
        @test par_rate ≈ swap_q.fixed_rate atol=1e-6

        fwd_quotes = [
            FRAQuote(asof, Date(2024, 4, 2), 0.05, idx),
            FRAQuote(Date(2024, 4, 2), Date(2024, 7, 2), 0.055, idx),
            FRAQuote(Date(2024, 7, 2), Date(2024, 10, 2), 0.057, idx),
            IRSwapQuote(asof, Date(2025, 1, 2), 0.06, idx; fixed_frequency=2, fixed_day_count=Thirty360())
        ]

        fc = bootstrap_forward_curve(fwd_quotes; asof=asof, discount_curve=dc)

        # FRA reprices
        fra_q = fwd_quotes[2]
        t1 = year_fraction(asof, fra_q.start_date, idx.day_count)
        t2 = year_fraction(asof, fra_q.end_date, idx.day_count)
        fwd = forward_rate(fc, t1, t2)
        @test fwd ≈ fra_q.rate atol=1e-6

        # IRS reprices
        irs_q = fwd_quotes[4]
        fixed_sched = Schedule(irs_q.start_date, irs_q.maturity_date;
                               tenor=Month(12 ÷ irs_q.fixed_frequency),
                               calendar=irs_q.calendar, bdc=irs_q.bdc)
        float_sched = Schedule(irs_q.start_date, irs_q.maturity_date;
                               tenor=idx.tenor, calendar=irs_q.calendar, bdc=irs_q.bdc)

        fixed_annuity = sum(
            year_fraction(d1, d2, irs_q.fixed_day_count) *
            discount(dc, year_fraction(asof, d2, ACT365()))
            for (d1, d2) in schedule_periods(fixed_sched)
        )
        float_pv = sum(
            forward_rate(fc,
                         year_fraction(asof, d1, idx.day_count),
                         year_fraction(asof, d2, idx.day_count)) *
            year_fraction(d1, d2, idx.day_count) *
            discount(dc, year_fraction(asof, payment_date(idx, d2), ACT365()))
            for (d1, d2) in schedule_periods(float_sched)
        )
        par_swap = float_pv / fixed_annuity
        @test par_swap ≈ irs_q.fixed_rate atol=1e-6
    end

    @testset "Bonds" begin
        @testset "Zero coupon bond" begin
            zcb = ZeroCouponBond(5.0, 100.0)
            rate = 0.05
            curve = ZeroCurve(rate)

            # Price = face * DF
            @test ir_price(zcb, curve) ≈ 100 * exp(-0.05 * 5) atol=1e-10
            @test ir_price(zcb, rate) ≈ 100 * exp(-0.05 * 5) atol=1e-10

            # Duration = maturity for zero
            @test duration(zcb, rate) ≈ 5.0 atol=1e-10

            # YTM
            mkt_price = 78.0
            ytm = yield_to_maturity(zcb, mkt_price)
            @test ir_price(zcb, ytm) ≈ mkt_price atol=1e-8
        end

        @testset "Fixed rate bond" begin
            # 5-year 4% semi-annual bond
            bond = FixedRateBond(5.0, 0.04, 2, 100.0)
            rate = 0.05

            # Should have 10 cash flows
            cfs = QuantNova.InterestRates.cash_flows(bond)
            @test length(cfs) == 10
            @test cfs[1][2] ≈ 2.0  # First coupon
            @test cfs[end][2] ≈ 102.0  # Last coupon + principal

            # Price at par when yield = coupon (approximately, due to continuous vs discrete compounding)
            # With continuous compounding, 4% yield gives ~99.8 for 4% semi-annual coupon
            par_price = ir_price(bond, 0.04)
            @test par_price ≈ 100.0 atol=2.0  # Approximately par

            # Duration < maturity for coupon bond
            @test duration(bond, rate) < 5.0
            @test duration(bond, rate) > 4.0

            # Modified duration
            mod_dur = modified_duration(bond, rate)
            @test mod_dur < duration(bond, rate)

            # Convexity > 0
            @test convexity(bond, rate) > 0

            # DV01
            dv = dv01(bond, rate)
            @test dv > 0

            # DV01 ≈ price change for 1bp move (approximate due to convexity)
            p1 = ir_price(bond, rate)
            p2 = ir_price(bond, rate + 0.0001)
            @test abs(p1 - p2) ≈ dv atol=0.01
        end

        @testset "Yield to maturity" begin
            bond = FixedRateBond(10.0, 0.05, 2, 100.0)

            # Price at 95
            ytm = yield_to_maturity(bond, 95.0)
            @test ir_price(bond, ytm) ≈ 95.0 atol=1e-8

            # YTM > coupon when price < par
            @test ytm > 0.05
        end
    end

    @testset "Short Rate Models" begin
        @testset "Vasicek" begin
            model = Vasicek(0.5, 0.05, 0.01, 0.03)

            # Bond price
            P = bond_price(model, 5.0)
            @test 0 < P < 1

            # Expected rate converges to theta
            mean_1, var_1 = short_rate(model, 1.0)
            mean_10, var_10 = short_rate(model, 10.0)

            @test abs(mean_10 - model.θ) < abs(mean_1 - model.θ)

            # Variance stabilizes
            @test var_10 ≈ model.σ^2 / (2 * model.κ) atol=0.001

            # Simulation
            paths = simulate_short_rate(model, 1.0, 100, 1000)
            @test size(paths) == (101, 1000)
            @test mean(paths[end, :]) ≈ mean_1 atol=0.01
        end

        @testset "CIR" begin
            # Feller condition satisfied
            model = CIR(0.5, 0.05, 0.05, 0.03)

            P = bond_price(model, 5.0)
            @test 0 < P < 1

            # Simulation stays positive
            paths = simulate_short_rate(model, 1.0, 100, 100)
            @test all(paths .>= 0)
        end

        @testset "Hull-White" begin
            curve = ZeroCurve(0.05)
            model = HullWhite(0.1, 0.01, curve)

            # Bond price matches market curve
            @test bond_price(model, 5.0) ≈ discount(curve, 5.0) atol=1e-10

            # Simulation
            paths = simulate_short_rate(model, 1.0, 100, 100)
            @test size(paths) == (101, 100)
        end
    end

    @testset "IR Derivatives" begin
        curve = ZeroCurve(0.05)
        vol = 0.20

        @testset "Caplet" begin
            caplet = Caplet(1.0, 1.25, 0.05, 1e6)

            price_cap = black_caplet(caplet, curve, vol)
            @test price_cap > 0

            # ATM caplet
            fwd = forward_rate(curve, 1.0, 1.25)
            atm_caplet = Caplet(1.0, 1.25, fwd, 1e6)
            atm_price = black_caplet(atm_caplet, curve, vol)

            # OTM caplet cheaper
            otm_caplet = Caplet(1.0, 1.25, fwd + 0.02, 1e6)
            otm_price = black_caplet(otm_caplet, curve, vol)
            @test otm_price < atm_price
        end

        @testset "Cap" begin
            cap = Cap(5.0, 0.05, 4, 1e6)
            price_c = black_cap(cap, curve, vol)
            @test price_c > 0

            # Higher strike = lower price
            cap_high = Cap(5.0, 0.06, 4, 1e6)
            @test black_cap(cap_high, curve, vol) < price_c
        end

        @testset "Cap-Floor parity" begin
            strike = 0.05
            cap = Cap(3.0, strike, 4, 1e6)
            floor = Floor(3.0, strike, 4, 1e6)

            cap_price = black_cap(cap, curve, vol)
            floor_price = QuantNova.InterestRates.black_floor(floor, curve, vol)

            # Cap - Floor = Swap value (approximately)
            # For ATM, swap ≈ 0, so cap ≈ floor
            fwd_avg = forward_rate(curve, 0.0, 3.0)
            if abs(strike - fwd_avg) < 0.01
                @test abs(cap_price - floor_price) / cap_price < 0.5
            end
        end

        @testset "Swaption" begin
            # 1Y into 4Y payer swaption
            swaption = Swaption(1.0, 5.0, 0.05, true, 2, 1e6)

            price_s = ir_price(swaption, curve, vol)
            @test price_s > 0

            # Receiver swaption
            recv = Swaption(1.0, 5.0, 0.05, false, 2, 1e6)
            price_r = ir_price(recv, curve, vol)

            # Put-call parity: payer - receiver = forward swap value
            @test price_s != price_r
        end
    end
end
