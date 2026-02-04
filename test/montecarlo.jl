using Random
using Statistics: mean, std

const RUN_GPU_BACKENDS = get(ENV, "QUANTNOVA_TEST_GPU_BACKENDS", "0") == "1"
if RUN_GPU_BACKENDS
    import Enzyme
end

@testset "Monte Carlo" begin

    @testset "GBM Path Simulation" begin
        dynamics = GBMDynamics(0.05, 0.2)
        S0, T, nsteps = 100.0, 1.0, 252

        path = QuantNova.MonteCarlo.simulate_gbm(S0, T, nsteps, dynamics)
        @test length(path) == nsteps + 1
        @test path[1] == S0
        @test all(path .> 0)  # Prices stay positive

        # Antithetic paths
        path1, path2 = QuantNova.MonteCarlo.simulate_gbm_antithetic(S0, T, nsteps, dynamics)
        @test length(path1) == nsteps + 1
        @test length(path2) == nsteps + 1
        @test path1[1] == path2[1] == S0
    end

    @testset "Heston Path Simulation" begin
        dynamics = HestonDynamics(0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        S0, T, nsteps = 100.0, 1.0, 252

        path = QuantNova.MonteCarlo.simulate_heston(S0, T, nsteps, dynamics)
        @test length(path) == nsteps + 1
        @test path[1] == S0
        @test all(path .> 0)
    end

    @testset "European Option Pricing" begin
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        dynamics = GBMDynamics(r, sigma)

        # Price European call
        call_result = mc_price(S0, T, EuropeanCall(K), dynamics; npaths=50000)
        @test call_result.price > 0
        @test call_result.stderr > 0
        @test call_result.stderr < 0.5  # Reasonable precision

        # Compare to Black-Scholes (should be close)
        bs_price = black_scholes(S0, K, T, r, sigma, :call)
        @test abs(call_result.price - bs_price) < 3 * call_result.stderr  # Within 3 SE

        # Put-call parity check
        put_result = mc_price(S0, T, EuropeanPut(K), dynamics; npaths=50000)
        parity_diff = call_result.price - put_result.price - (S0 - K * exp(-r * T))
        @test abs(parity_diff) < 0.5
    end

    @testset "Asian Option Pricing" begin
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        dynamics = GBMDynamics(r, sigma)

        asian_call = mc_price(S0, T, AsianCall(K), dynamics; npaths=20000)
        euro_call = mc_price(S0, T, EuropeanCall(K), dynamics; npaths=20000)

        # Asian options have lower value than European (less volatility in average)
        @test asian_call.price > 0
        @test asian_call.price < euro_call.price + 3 * euro_call.stderr
    end

    @testset "Barrier Option Pricing" begin
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        dynamics = GBMDynamics(r, sigma)

        # Up-and-out call with barrier above spot
        barrier_call = mc_price(S0, T, UpAndOutCall(K, 130.0), dynamics; npaths=20000)
        euro_call = mc_price(S0, T, EuropeanCall(K), dynamics; npaths=20000)

        # Barrier option worth less than vanilla (can knock out)
        @test barrier_call.price >= 0
        @test barrier_call.price < euro_call.price + euro_call.stderr
    end

    @testset "Variance Reduction" begin
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        dynamics = GBMDynamics(r, sigma)

        # Compare standard vs antithetic
        rng1 = Random.MersenneTwister(123)
        rng2 = Random.MersenneTwister(123)

        result_std = mc_price(S0, T, EuropeanCall(K), dynamics;
                              npaths=10000, antithetic=false, rng=rng1)
        result_anti = mc_price(S0, T, EuropeanCall(K), dynamics;
                               npaths=10000, antithetic=true, rng=rng2)

        # Antithetic should have lower standard error
        @test result_anti.stderr < result_std.stderr
    end

    @testset "Heston Monte Carlo vs Analytic" begin
        S0, K, T, r = 100.0, 100.0, 1.0, 0.05
        params = HestonParams(0.04, 0.04, 2.0, 0.3, -0.7)
        dynamics = HestonDynamics(r, params.v0, params.kappa, params.theta, params.sigma, params.rho)

        mc_result = mc_price(S0, T, EuropeanCall(K), dynamics; npaths=50000, nsteps=252)
        analytic = heston_price(S0, K, T, r, params, :call)

        # Should be reasonably close (within a few standard errors)
        @test abs(mc_result.price - analytic) < 5 * mc_result.stderr
    end

    @testset "Monte Carlo Greeks" begin
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        dynamics = GBMDynamics(r, sigma)

        # Delta should be positive for call, between 0 and 1
        delta = mc_delta(S0, T, EuropeanCall(K), dynamics; npaths=5000)
        @test 0 < delta < 1

        # Full Greeks
        greeks = mc_greeks(S0, T, EuropeanCall(K), dynamics; npaths=5000)
        @test 0 < greeks.delta < 1
        @test greeks.vega > 0  # Vega always positive for long options
    end

    @testset "Monte Carlo Greeks Backend Parity" begin
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        dynamics = GBMDynamics(r, sigma)
        payoff = EuropeanCall(K)

        # ForwardDiff as reference
        delta_fd = mc_delta(S0, T, payoff, dynamics; npaths=1000, nsteps=50, backend=ForwardDiffBackend())
        greeks_fd = mc_greeks(S0, T, payoff, dynamics; npaths=1000, nsteps=50, backend=ForwardDiffBackend())

        # PureJulia (finite differences) should match
        delta_pj = mc_delta(S0, T, payoff, dynamics; npaths=1000, nsteps=50, backend=PureJuliaBackend())
        @test delta_pj ≈ delta_fd atol=0.05  # Finite diff has lower precision

        greeks_pj = mc_greeks(S0, T, payoff, dynamics; npaths=1000, nsteps=50, backend=PureJuliaBackend())
        @test greeks_pj.delta ≈ greeks_fd.delta atol=0.05
        @test greeks_pj.vega ≈ greeks_fd.vega atol=1.0  # Vega can vary more

        # Test with_backend context works with MC
        result = with_backend(PureJuliaBackend()) do
            mc_delta(S0, T, payoff, dynamics; npaths=1000, nsteps=50)
        end
        @test result ≈ delta_pj atol=1e-10
    end

    @testset "Quasi-Monte Carlo" begin
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        dynamics = GBMDynamics(r, sigma)

        # QMC pricing is deterministic
        result1 = mc_price_qmc(S0, T, EuropeanCall(K), dynamics; npaths=5000, nsteps=50)
        result2 = mc_price_qmc(S0, T, EuropeanCall(K), dynamics; npaths=5000, nsteps=50)
        @test result1.price == result2.price  # Exactly equal (deterministic)

        # QMC price should be close to Black-Scholes
        bs_price = black_scholes(S0, K, T, r, sigma, :call)
        @test abs(result1.price - bs_price) < 3 * result1.stderr

        # Sobol normals are deterministic
        Z1 = QuantNova.MonteCarlo.sobol_normals(10, 100)
        Z2 = QuantNova.MonteCarlo.sobol_normals(10, 100)
        @test Z1 == Z2

        # Sobol normals have approximately correct statistics
        Z = QuantNova.MonteCarlo.sobol_normals(1, 10000)
        @test abs(mean(Z)) < 0.05  # Near zero mean
        @test abs(std(Z) - 1.0) < 0.1  # Near unit variance
    end

    if RUN_GPU_BACKENDS
        @testset "Enzyme MC Greeks with QMC" begin
            S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
            dynamics = GBMDynamics(r, sigma)
            payoff = EuropeanCall(K)

            # Enzyme now works with MC Greeks via QMC
            delta_enz = mc_delta(S0, T, payoff, dynamics; npaths=1000, nsteps=50, backend=EnzymeBackend())
            @test 0 < delta_enz < 1  # Valid delta range for call

            greeks_enz = mc_greeks(S0, T, payoff, dynamics; npaths=1000, nsteps=50, backend=EnzymeBackend())
            @test 0 < greeks_enz.delta < 1
            @test greeks_enz.vega > 0  # Vega always positive

            # Compare to ForwardDiff (uses different RNG so won't match exactly,
            # but should be in same ballpark)
            delta_fd = mc_delta(S0, T, payoff, dynamics; npaths=1000, nsteps=50, backend=ForwardDiffBackend())
            @test abs(delta_enz - delta_fd) < 0.1  # Within 0.1 of each other
        end
    end

    @testset "Longstaff-Schwartz American Options" begin
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        dynamics = GBMDynamics(r, sigma)

        # American put price
        am_put = lsm_price(S0, T, AmericanPut(K), dynamics; npaths=20000, nsteps=50)
        @test am_put.price > 0
        @test am_put.stderr > 0

        # American put >= European put (early exercise premium)
        eu_put = mc_price(S0, T, EuropeanPut(K), dynamics; npaths=20000)
        @test am_put.price >= eu_put.price - 3 * eu_put.stderr

        # Deep ITM American put should have significant early exercise value
        deep_itm_am = lsm_price(S0, T, AmericanPut(130.0), dynamics; npaths=20000, nsteps=50)
        deep_itm_eu = mc_price(S0, T, EuropeanPut(130.0), dynamics; npaths=20000)
        # Early exercise premium should be positive for deep ITM
        @test deep_itm_am.price > deep_itm_eu.price - 2 * deep_itm_eu.stderr

        # American call on non-dividend stock ≈ European call
        am_call = lsm_price(S0, T, AmericanCall(K), dynamics; npaths=20000, nsteps=50)
        eu_call = mc_price(S0, T, EuropeanCall(K), dynamics; npaths=20000)
        # Should be very close (no early exercise premium for calls without dividends)
        @test abs(am_call.price - eu_call.price) < 3 * max(am_call.stderr, eu_call.stderr)

        # Intrinsic value sanity check
        @test QuantNova.MonteCarlo.intrinsic(AmericanPut(100.0), 80.0) == 20.0
        @test QuantNova.MonteCarlo.intrinsic(AmericanPut(100.0), 120.0) == 0.0
        @test QuantNova.MonteCarlo.intrinsic(AmericanCall(100.0), 120.0) == 20.0
        @test QuantNova.MonteCarlo.intrinsic(AmericanCall(100.0), 80.0) == 0.0
    end

end
