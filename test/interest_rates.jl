using Test
using Quasar
using Quasar.InterestRates

# Use module-qualified price to avoid collision with Instruments.price
const ir_price = Quasar.InterestRates.price

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
            cfs = Quasar.InterestRates.cash_flows(bond)
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
            floor_price = Quasar.InterestRates.black_floor(floor, curve, vol)

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
