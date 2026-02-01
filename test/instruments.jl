using Test
using SuperNova

@testset "Instruments" begin
    @testset "Stock" begin
        stock = Stock("AAPL")

        @test stock.symbol == "AAPL"
        @test stock isa AbstractEquity
        @test stock isa AbstractInstrument

        # Traits
        @test ispriceable(stock)
        @test isdifferentiable(stock)

        # Pricing
        state = MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )
        @test price(stock, state) == 150.0

        # Validation - empty symbol should throw
        @test_throws ArgumentError Stock("")
    end

    @testset "EuropeanOption validation" begin
        # Valid option
        @test EuropeanOption("AAPL", 100.0, 1.0, :call) isa EuropeanOption

        # Invalid strike (negative)
        @test_throws ArgumentError EuropeanOption("AAPL", -100.0, 1.0, :call)

        # Invalid strike (zero)
        @test_throws ArgumentError EuropeanOption("AAPL", 0.0, 1.0, :call)

        # Invalid expiry (negative)
        @test_throws ArgumentError EuropeanOption("AAPL", 100.0, -1.0, :call)

        # Invalid expiry (zero)
        @test_throws ArgumentError EuropeanOption("AAPL", 100.0, 0.0, :call)

        # Invalid option type
        @test_throws ArgumentError EuropeanOption("AAPL", 100.0, 1.0, :foo)

        # Infinite strike
        @test_throws ArgumentError EuropeanOption("AAPL", Inf, 1.0, :call)

        # NaN expiry
        @test_throws ArgumentError EuropeanOption("AAPL", 100.0, NaN, :call)
    end

    @testset "EuropeanOption" begin
        # Call option
        call = EuropeanOption("AAPL", 150.0, 1.0, :call)

        @test call.underlying == "AAPL"
        @test call.strike == 150.0
        @test call.expiry == 1.0
        @test call.optiontype == :call
        @test call isa AbstractOption

        # Traits
        @test ispriceable(call)
        @test isdifferentiable(call)
        @test hasgreeks(call)

        # Black-Scholes pricing (ATM call, 1 year, 20% vol, 5% rate)
        state = MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        # ATM call should be worth roughly S * 0.4 * σ * √T for small σ
        # More precisely, use known BS value
        p = price(call, state)
        @test p > 0.0
        @test p < 150.0  # Can't be worth more than underlying

        # Put option
        put = EuropeanOption("AAPL", 150.0, 1.0, :put)
        p_put = price(put, state)

        # Put-call parity: C - P = S - K*exp(-rT)
        S = 150.0
        K = 150.0
        r = 0.05
        T = 1.0
        @test p - p_put ≈ S - K * exp(-r * T) atol=1e-10
    end

    @testset "Greeks (Analytical Black-Scholes)" begin
        call = EuropeanOption("AAPL", 150.0, 1.0, :call)

        state = MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        # Get all Greeks (uses analytical formulas for EuropeanOption)
        greeks = compute_greeks(call, state)

        # Validate against known Black-Scholes values
        # S=150, K=150, T=1, r=0.05, σ=0.2 (ATM call)
        # d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T) = (0 + 0.07) / 0.2 = 0.35
        # d2 = d1 - σ√T = 0.35 - 0.2 = 0.15

        # Delta = N(d1)
        @test greeks.delta ≈ 0.6368306511756191 atol=1e-10

        # Gamma = n(d1) / (S * σ * √T)
        @test greeks.gamma ≈ 0.012508011563897931 atol=1e-10

        # Vega = S * n(d1) * √T * 0.01
        @test greeks.vega ≈ 0.5628605203754069 atol=1e-10

        # Theta = -S*n(d1)*σ/(2√T) - r*K*exp(-rT)*N(d2)
        @test greeks.theta ≈ -9.621041319657294 atol=1e-8

        # Rho = K*T*exp(-rT)*N(d2) * 0.01
        @test greeks.rho ≈ 0.798487223180645 atol=1e-10

        # Sanity checks
        @test 0.0 < greeks.delta < 1.0  # Call delta bounded
        @test greeks.gamma > 0.0         # Gamma always positive
        @test greeks.vega > 0.0          # Long options have positive vega
        @test greeks.theta < 0.0         # Time decay for long options
        @test greeks.rho > 0.0           # Call rho positive

        # Put Greeks sanity check
        put = EuropeanOption("AAPL", 150.0, 1.0, :put)
        put_greeks = compute_greeks(put, state)

        @test -1.0 < put_greeks.delta < 0.0  # Put delta negative
        @test put_greeks.gamma > 0.0          # Same gamma as call
        @test put_greeks.gamma ≈ greeks.gamma atol=1e-12  # Gamma same for call/put
        @test put_greeks.vega ≈ greeks.vega atol=1e-12    # Vega same for call/put
        @test put_greeks.rho < 0.0            # Put rho negative
    end

    @testset "Black-Scholes edge cases" begin
        # Edge case: T → 0 (at expiry)
        @testset "T → 0 (expiry)" begin
            # ITM call at expiry = intrinsic value
            @test black_scholes(150.0, 100.0, 0.0, 0.05, 0.2, :call) ≈ 50.0
            @test black_scholes(150.0, 100.0, 1e-10, 0.05, 0.2, :call) ≈ 50.0 atol=0.01

            # OTM call at expiry = 0
            @test black_scholes(100.0, 150.0, 0.0, 0.05, 0.2, :call) ≈ 0.0
            @test black_scholes(100.0, 150.0, 1e-10, 0.05, 0.2, :call) ≈ 0.0 atol=0.01

            # ITM put at expiry = intrinsic value
            @test black_scholes(100.0, 150.0, 0.0, 0.05, 0.2, :put) ≈ 50.0
            @test black_scholes(100.0, 150.0, 1e-10, 0.05, 0.2, :put) ≈ 50.0 atol=0.01

            # OTM put at expiry = 0
            @test black_scholes(150.0, 100.0, 0.0, 0.05, 0.2, :put) ≈ 0.0
            @test black_scholes(150.0, 100.0, 1e-10, 0.05, 0.2, :put) ≈ 0.0 atol=0.01

            # ATM at expiry = 0
            @test black_scholes(100.0, 100.0, 0.0, 0.05, 0.2, :call) ≈ 0.0
            @test black_scholes(100.0, 100.0, 0.0, 0.05, 0.2, :put) ≈ 0.0
        end

        # Edge case: σ → 0 (zero volatility)
        @testset "σ → 0 (zero vol)" begin
            S, K, T, r = 100.0, 100.0, 1.0, 0.05

            # Zero vol: deterministic forward price
            forward = S * exp(r * T)  # 105.127...
            df = exp(-r * T)

            # ATM call with zero vol: forward > K, so intrinsic = forward - K
            @test black_scholes(S, K, T, r, 0.0, :call) ≈ df * max(forward - K, 0) atol=1e-10

            # Deep ITM call
            @test black_scholes(150.0, 100.0, 1.0, 0.05, 0.0, :call) > 0

            # Deep OTM call
            @test black_scholes(50.0, 100.0, 1.0, 0.05, 0.0, :call) ≈ 0.0

            # Deep ITM put
            @test black_scholes(50.0, 100.0, 1.0, 0.05, 0.0, :put) > 0

            # Deep OTM put
            @test black_scholes(150.0, 100.0, 1.0, 0.05, 0.0, :put) ≈ 0.0
        end

        # Edge case: very small T (near expiry)
        @testset "Small T" begin
            # 1 day to expiry
            T = 1/365
            S, K, r, σ = 100.0, 100.0, 0.05, 0.2

            p = black_scholes(S, K, T, r, σ, :call)
            @test p > 0  # Should be positive
            @test p < S  # Should be less than stock price
            @test isfinite(p)  # Should not be NaN or Inf
        end

        # Edge case: very high strike (K → ∞)
        @testset "High strike" begin
            S, T, r, σ = 100.0, 1.0, 0.05, 0.2

            # Very high strike call should be nearly worthless
            @test black_scholes(S, 1e6, T, r, σ, :call) < 0.01

            # Very high strike put should be nearly K * exp(-rT)
            K = 1e6
            put_price = black_scholes(S, K, T, r, σ, :put)
            @test put_price > 0.99 * K * exp(-r * T)  # Close to discounted strike
        end

        # Edge case: S → 0 (worthless underlying)
        @testset "S → 0" begin
            K, T, r, σ = 100.0, 1.0, 0.05, 0.2

            # Call with zero underlying is worthless
            @test black_scholes(1e-10, K, T, r, σ, :call) ≈ 0.0 atol=1e-8

            # Put with zero underlying = discounted strike
            put_price = black_scholes(1e-10, K, T, r, σ, :put)
            @test put_price ≈ K * exp(-r * T) atol=0.01
        end
    end

    @testset "Greeks - second order (vanna, volga, charm)" begin
        call = EuropeanOption("AAPL", 150.0, 1.0, :call)

        state = MarketState(
            prices=Dict("AAPL" => 150.0),
            rates=Dict("USD" => 0.05),
            volatilities=Dict("AAPL" => 0.2),
            timestamp=0.0
        )

        greeks = compute_greeks(call, state)

        # Vanna: d²V/dSdσ - should be non-zero for ATM options
        @test isfinite(greeks.vanna)

        # Volga: d²V/dσ² - should be non-zero
        @test isfinite(greeks.volga)

        # Charm: d²V/dSdT - should be non-zero
        @test isfinite(greeks.charm)

        # For ATM options, vanna is typically small but non-zero
        @test greeks.vanna != 0.0

        # Put should have same vanna magnitude (opposite sign for some definitions)
        put = EuropeanOption("AAPL", 150.0, 1.0, :put)
        put_greeks = compute_greeks(put, state)
        @test isfinite(put_greeks.vanna)
        @test isfinite(put_greeks.volga)
        @test isfinite(put_greeks.charm)

        # Volga should be same for call and put (same gamma profile)
        @test put_greeks.volga ≈ greeks.volga atol=1e-10
    end
end
