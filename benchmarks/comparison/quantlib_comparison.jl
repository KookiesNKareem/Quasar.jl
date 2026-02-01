# SuperNova vs QuantLib Comparison Benchmark
#
# Compares SuperNova.jl against QuantLib (the industry-standard C++ library)
# via Python bindings for:
# 1. European option pricing (Black-Scholes)
# 2. Greeks computation (Delta, Gamma, Vega, Theta, Rho)
# 3. American option pricing (binomial tree)

using SuperNova
using PyCall
using Statistics
using Printf

# Import QuantLib via PyCall
const ql = PyNULL()

function init_quantlib()
    copy!(ql, pyimport("QuantLib"))
end

# ============================================================================
# QuantLib Pricing Functions
# ============================================================================

function quantlib_european_price(S, K, T, r, σ, optiontype::Symbol)
    # Set up dates
    today = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    maturity = today + Int(round(T * 365))

    # Option type
    opt_type = optiontype == :call ? ql.Option.Call : ql.Option.Put

    # Build option
    payoff = ql.PlainVanillaPayoff(opt_type, K)
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)

    # Market data
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed())
    )
    div_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0, ql.Actual365Fixed())
    )
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), σ, ql.Actual365Fixed())
    )

    # BSM process and engine
    process = ql.BlackScholesMertonProcess(spot_handle, div_handle, rate_handle, vol_handle)
    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)

    return option.NPV()
end

function quantlib_european_greeks(S, K, T, r, σ, optiontype::Symbol)
    today = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    maturity = today + Int(round(T * 365))

    opt_type = optiontype == :call ? ql.Option.Call : ql.Option.Put

    payoff = ql.PlainVanillaPayoff(opt_type, K)
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed())
    )
    div_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0, ql.Actual365Fixed())
    )
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), σ, ql.Actual365Fixed())
    )

    process = ql.BlackScholesMertonProcess(spot_handle, div_handle, rate_handle, vol_handle)
    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)

    return (
        delta = option.delta(),
        gamma = option.gamma(),
        vega = option.vega() / 100,  # QuantLib returns per 100% vol change
        theta = option.theta(),  # QuantLib returns per-day theta; SuperNova returns per-year
        rho = option.rho() / 100  # QuantLib returns per 100% rate change
    )
end

function quantlib_american_price(S, K, T, r, σ, optiontype::Symbol; nsteps=100)
    today = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    maturity = today + Int(round(T * 365))

    opt_type = optiontype == :call ? ql.Option.Call : ql.Option.Put

    payoff = ql.PlainVanillaPayoff(opt_type, K)
    exercise = ql.AmericanExercise(today, maturity)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed())
    )
    div_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0, ql.Actual365Fixed())
    )
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), σ, ql.Actual365Fixed())
    )

    process = ql.BlackScholesMertonProcess(spot_handle, div_handle, rate_handle, vol_handle)
    engine = ql.BinomialVanillaEngine(process, "crr", nsteps)
    option.setPricingEngine(engine)

    return option.NPV()
end

# ============================================================================
# Timing Utilities
# ============================================================================

function benchmark(f, n_runs=100, n_warmup=5)
    # Warmup
    for _ in 1:n_warmup
        f()
    end

    # Timed runs
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed f()
        push!(times, t * 1e6)  # Convert to microseconds
    end

    return (
        median = median(times),
        mean = mean(times),
        std = std(times),
        min = minimum(times),
        max = maximum(times)
    )
end

# ============================================================================
# Comparison Benchmarks
# ============================================================================

function run_european_pricing_benchmark(; n_runs=100)
    println("\n" * "="^70)
    println("EUROPEAN OPTION PRICING BENCHMARK")
    println("="^70)

    # Test parameters
    S, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.2

    # Verify correctness first
    nova_price = black_scholes(S, K, T, r, σ, :call)
    ql_price = quantlib_european_price(S, K, T, r, σ, :call)

    println("\nPrice verification (S=$S, K=$K, T=$T, r=$r, σ=$σ):")
    println("  SuperNova.jl:   $(round(nova_price, digits=6))")
    println("  QuantLib:  $(round(ql_price, digits=6))")
    println("  Diff:      $(round(abs(nova_price - ql_price), digits=10))")

    # Benchmark
    println("\nTiming ($n_runs runs):")

    nova_times = benchmark(() -> black_scholes(S, K, T, r, σ, :call), n_runs)
    ql_times = benchmark(() -> quantlib_european_price(S, K, T, r, σ, :call), n_runs)

    speedup = ql_times.median / nova_times.median

    println("  SuperNova.jl:   $(round(nova_times.median, digits=2)) μs (median)")
    println("  QuantLib:  $(round(ql_times.median, digits=2)) μs (median)")
    println("  Speedup:   $(round(speedup, digits=1))x")

    return (nova=nova_times, quantlib=ql_times, speedup=speedup)
end

function run_greeks_benchmark(; n_runs=100)
    println("\n" * "="^70)
    println("GREEKS COMPUTATION BENCHMARK")
    println("="^70)

    S, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.2

    # Create SuperNova option and state
    state = MarketState(
        prices = Dict("TEST" => S),
        rates = Dict("USD" => r),
        volatilities = Dict("TEST" => σ),
        timestamp = 0.0
    )
    option = EuropeanOption("TEST", K, T, :call)

    # Verify correctness
    nova_greeks = compute_greeks(option, state)
    ql_greeks = quantlib_european_greeks(S, K, T, r, σ, :call)

    # Note: Theta units differ - SuperNova is per-year, QuantLib is per-day
    # Convert SuperNova theta to per-day for comparison
    nova_theta_daily = nova_greeks.theta / 365

    println("\nGreeks verification:")
    println("  Metric    SuperNova.jl     QuantLib    Diff")
    println("  ─────────────────────────────────────────")
    @printf("  Delta     %+.6f   %+.6f   %.2e\n", nova_greeks.delta, ql_greeks.delta, abs(nova_greeks.delta - ql_greeks.delta))
    @printf("  Gamma     %+.6f   %+.6f   %.2e\n", nova_greeks.gamma, ql_greeks.gamma, abs(nova_greeks.gamma - ql_greeks.gamma))
    @printf("  Vega      %+.6f   %+.6f   %.2e\n", nova_greeks.vega, ql_greeks.vega, abs(nova_greeks.vega - ql_greeks.vega))
    @printf("  Theta*    %+.6f   %+.6f   %.2e\n", nova_theta_daily, ql_greeks.theta, abs(nova_theta_daily - ql_greeks.theta))
    @printf("  Rho       %+.6f   %+.6f   %.2e\n", nova_greeks.rho, ql_greeks.rho, abs(nova_greeks.rho - ql_greeks.rho))
    println("  * Theta converted to per-day for comparison")

    # Benchmark
    println("\nTiming ($n_runs runs):")

    nova_times = benchmark(() -> compute_greeks(option, state), n_runs)
    ql_times = benchmark(() -> quantlib_european_greeks(S, K, T, r, σ, :call), n_runs)

    speedup = ql_times.median / nova_times.median

    println("  SuperNova.jl (AD):      $(round(nova_times.median, digits=2)) μs (median)")
    println("  QuantLib (analytic): $(round(ql_times.median, digits=2)) μs (median)")
    println("  Speedup:           $(round(speedup, digits=1))x")

    return (nova=nova_times, quantlib=ql_times, speedup=speedup)
end

function run_american_benchmark(; n_runs=50, nsteps=100)
    println("\n" * "="^70)
    println("AMERICAN OPTION PRICING BENCHMARK (Binomial Tree, $nsteps steps)")
    println("="^70)

    S, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.2

    # Verify correctness
    nova_price = american_binomial(S, K, T, r, σ, :put, nsteps)
    ql_price = quantlib_american_price(S, K, T, r, σ, :put; nsteps=nsteps)

    println("\nPrice verification (American put):")
    println("  SuperNova.jl:   $(round(nova_price, digits=6))")
    println("  QuantLib:  $(round(ql_price, digits=6))")
    println("  Diff:      $(round(abs(nova_price - ql_price), digits=6))")

    # Benchmark
    println("\nTiming ($n_runs runs):")

    nova_times = benchmark(() -> american_binomial(S, K, T, r, σ, :put, nsteps), n_runs)
    ql_times = benchmark(() -> quantlib_american_price(S, K, T, r, σ, :put; nsteps=nsteps), n_runs)

    speedup = ql_times.median / nova_times.median

    println("  SuperNova.jl:   $(round(nova_times.median, digits=2)) μs (median)")
    println("  QuantLib:  $(round(ql_times.median, digits=2)) μs (median)")
    println("  Speedup:   $(round(speedup, digits=1))x")

    return (nova=nova_times, quantlib=ql_times, speedup=speedup)
end

function run_batch_pricing_benchmark(; n_options=1000, n_runs=10)
    println("\n" * "="^70)
    println("BATCH PRICING BENCHMARK ($n_options options)")
    println("="^70)

    # Generate random option parameters
    S = 100.0
    r = 0.05
    Ks = 80.0 .+ 40.0 .* rand(n_options)
    Ts = 0.1 .+ 1.9 .* rand(n_options)
    σs = 0.1 .+ 0.4 .* rand(n_options)

    # SuperNova batch pricing
    function nova_batch()
        prices = Vector{Float64}(undef, n_options)
        for i in 1:n_options
            prices[i] = black_scholes(S, Ks[i], Ts[i], r, σs[i], :call)
        end
        return prices
    end

    # QuantLib batch pricing
    function ql_batch()
        prices = Vector{Float64}(undef, n_options)
        for i in 1:n_options
            prices[i] = quantlib_european_price(S, Ks[i], Ts[i], r, σs[i], :call)
        end
        return prices
    end

    println("\nTiming ($n_runs runs):")

    nova_times = benchmark(nova_batch, n_runs, 2)
    ql_times = benchmark(ql_batch, n_runs, 2)

    speedup = ql_times.median / nova_times.median

    # Convert to per-option time
    nova_per_opt = nova_times.median / n_options
    ql_per_opt = ql_times.median / n_options

    println("  SuperNova.jl:   $(round(nova_times.median/1000, digits=2)) ms total, $(round(nova_per_opt, digits=3)) μs/option")
    println("  QuantLib:  $(round(ql_times.median/1000, digits=2)) ms total, $(round(ql_per_opt, digits=3)) μs/option")
    println("  Speedup:   $(round(speedup, digits=1))x")

    return (nova=nova_times, quantlib=ql_times, speedup=speedup)
end

# ============================================================================
# Main Entry Point
# ============================================================================

function run_all_benchmarks(; verbose=true)
    println("\n" * "="^70)
    println("NOVA.JL vs QUANTLIB BENCHMARK SUITE")
    println("="^70)
    println("\nQuantLib version: $(ql.__version__)")
    println("SuperNova.jl: Differentiable Quantitative Finance Library")

    results = Dict{String, Any}()

    results["european"] = run_european_pricing_benchmark()
    results["greeks"] = run_greeks_benchmark()
    results["american"] = run_american_benchmark()
    results["batch"] = run_batch_pricing_benchmark()

    # Summary
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    println("\n  Benchmark              SuperNova.jl (μs)   QuantLib (μs)   Speedup")
    println("  ─────────────────────────────────────────────────────────────")
    @printf("  European pricing       %8.1f       %8.1f         %5.1fx\n",
            results["european"].nova.median, results["european"].quantlib.median, results["european"].speedup)
    @printf("  Greeks (all 5)         %8.1f       %8.1f         %5.1fx\n",
            results["greeks"].nova.median, results["greeks"].quantlib.median, results["greeks"].speedup)
    @printf("  American (100 steps)   %8.1f       %8.1f         %5.1fx\n",
            results["american"].nova.median, results["american"].quantlib.median, results["american"].speedup)
    @printf("  Batch (1000 opts)      %8.1f       %8.1f         %5.1fx\n",
            results["batch"].nova.median, results["batch"].quantlib.median, results["batch"].speedup)

    println("\n" * "="^70)

    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    init_quantlib()
    run_all_benchmarks()
end
