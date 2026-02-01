# Calibration Performance Benchmark
# Compares CPU (ForwardDiff) vs GPU (Reactant) calibration timing

using SuperNova
using Printf
using Statistics: mean, std, median

# ============================================================================
# Synthetic Data Generation
# ============================================================================

"""
    generate_synthetic_smile(; n_strikes=20, F=100.0, T=1.0, r=0.05, base_vol=0.2)

Generate synthetic smile data for benchmarking.
Uses SABR-like smile shape with realistic parameters.
"""
function generate_synthetic_smile(; n_strikes::Int=20, F::Float64=100.0, T::Float64=1.0,
                                   r::Float64=0.05, base_vol::Float64=0.2)
    # Generate strikes around ATM
    moneyness = range(0.7, 1.3, length=n_strikes)
    strikes = F .* moneyness

    # Generate realistic smile using simplified SABR-like shape
    # Higher vol for OTM puts (negative skew typical for equities)
    quotes = OptionQuote[]
    for K in strikes
        log_m = log(K / F)
        # Skew: OTM puts have higher vol
        skew = -0.15 * log_m
        # Smile: wings have higher vol
        smile = 0.05 * log_m^2
        implied_vol = base_vol + skew + smile

        # Compute price using Black-76 for consistency
        opttype = K < F ? :put : :call
        price = black76(F, K, T, r, implied_vol, opttype)

        push!(quotes, OptionQuote(K, T, price, opttype, implied_vol))
    end

    return SmileData(T, F, r, quotes)
end

"""
    generate_synthetic_surface(; n_expiries=5, n_strikes=20, S=100.0, r=0.05)

Generate synthetic volatility surface for benchmarking.
"""
function generate_synthetic_surface(; n_expiries::Int=5, n_strikes::Int=20,
                                     S::Float64=100.0, r::Float64=0.05)
    expiries = [0.25, 0.5, 1.0, 1.5, 2.0][1:n_expiries]

    smiles = SmileData[]
    for T in expiries
        # Forward price
        F = S * exp(r * T)
        # Term structure: slightly higher vol for longer expiries
        base_vol = 0.18 + 0.02 * sqrt(T)
        smile = generate_synthetic_smile(n_strikes=n_strikes, F=F, T=T, r=r, base_vol=base_vol)
        push!(smiles, smile)
    end

    return VolSurface(S, r, smiles)
end

# ============================================================================
# Timing Utilities
# ============================================================================

"""
    time_calibration(calibrate_fn, data, backend; n_runs=5, warmup=1)

Time a calibration function with warmup runs.
Returns (median_time_ms, std_time_ms, result).
"""
function time_calibration(calibrate_fn, data, backend; n_runs::Int=5, warmup::Int=1)
    # Warmup runs
    for _ in 1:warmup
        calibrate_fn(data; backend=backend, max_iter=100)
    end

    # Timed runs
    times = Float64[]
    result = nothing
    for _ in 1:n_runs
        t0 = time_ns()
        result = calibrate_fn(data; backend=backend, max_iter=500)
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)  # Convert to ms
    end

    return (median(times), std(times), result)
end

# ============================================================================
# Main Benchmark
# ============================================================================

"""
    run_calibration_timing_benchmark(; verbose=true, n_strikes=20, n_expiries=5)

Run calibration timing benchmarks comparing backends.
"""
function run_calibration_timing_benchmark(; verbose::Bool=true,
                                           n_strikes::Int=20,
                                           n_expiries::Int=5)
    results = Dict{String, Any}()

    # Generate test data
    smile = generate_synthetic_smile(n_strikes=n_strikes)
    surface = generate_synthetic_surface(n_expiries=n_expiries, n_strikes=n_strikes)

    if verbose
        println()
        println("Calibration Performance Benchmark")
        println("=" ^ 65)
        println("Configuration:")
        println("  SABR: $n_strikes strikes, 1 expiry")
        println("  Heston: $n_expiries expiries x $n_strikes strikes = $(n_expiries * n_strikes) quotes")
        println("=" ^ 65)
    end

    # Check available backends
    backends = Dict{String, ADBackend}()
    backends["ForwardDiff"] = ForwardDiffBackend()

    # Check if Reactant is available (via the extension being loaded)
    reactant_available = false
    try
        # Try to create a ReactantBackend - this will work if the extension is loaded
        test_backend = ReactantBackend()
        # Test if it actually works with a simple function
        test_grad = gradient(x -> sum(x.^2), [1.0, 2.0]; backend=test_backend)
        if test_grad ≈ [2.0, 4.0]
            backends["Reactant (GPU)"] = test_backend
            reactant_available = true
        end
    catch e
        if verbose
            println("Note: Reactant test failed: $(typeof(e))")
        end
    end

    # SABR Calibration Benchmarks
    if verbose
        println()
        println("SABR Calibration ($n_strikes strikes)")
        println("-" ^ 65)
        @printf("%-20s %-15s %-15s %-12s\n", "Backend", "Time (ms)", "Std (ms)", "Converged")
        println("-" ^ 65)
    end

    sabr_results = Dict{String, Any}()
    sabr_baseline = nothing

    for (name, backend) in backends
        try
            median_ms, std_ms, result = time_calibration(calibrate_sabr, smile, backend)
            sabr_results[name] = (time=median_ms, std=std_ms, converged=result.converged)

            if name == "ForwardDiff"
                sabr_baseline = median_ms
            end

            if verbose
                @printf("%-20s %-15.2f %-15.2f %-12s\n",
                        name, median_ms, std_ms, result.converged ? "Yes" : "No")
            end
        catch e
            if verbose
                @printf("%-20s %-15s %-15s %-12s\n", name, "ERROR", "-", "-")
                println("  Error: ", sprint(showerror, e))
            end
        end
    end

    results["SABR"] = sabr_results

    # Heston Calibration Benchmarks
    if verbose
        println()
        println("Heston Calibration ($n_expiries expiries x $n_strikes strikes)")
        println("-" ^ 65)
        @printf("%-20s %-15s %-15s %-12s\n", "Backend", "Time (ms)", "Std (ms)", "Converged")
        println("-" ^ 65)
    end

    heston_results = Dict{String, Any}()
    heston_baseline = nothing

    for (name, backend) in backends
        # Skip Reactant for Heston - complex number AD not supported
        if contains(name, "Reactant")
            if verbose
                @printf("%-20s %-15s %-15s %-12s\n", name, "SKIP", "-", "-")
                println("  Note: Heston uses complex numbers, not supported by Reactant AD")
            end
            continue
        end

        try
            median_ms, std_ms, result = time_calibration(calibrate_heston, surface, backend)
            heston_results[name] = (time=median_ms, std=std_ms, converged=result.converged)

            if name == "ForwardDiff"
                heston_baseline = median_ms
            end

            if verbose
                @printf("%-20s %-15.2f %-15.2f %-12s\n",
                        name, median_ms, std_ms, result.converged ? "Yes" : "No")
            end
        catch e
            if verbose
                @printf("%-20s %-15s %-15s %-12s\n", name, "ERROR", "-", "-")
                println("  Error: ", sprint(showerror, e))
            end
        end
    end

    results["Heston"] = heston_results

    # Print speedup summary if multiple backends available
    if verbose && reactant_available
        println()
        println("Speedup Summary")
        println("-" ^ 65)

        if haskey(sabr_results, "Reactant (GPU)") && sabr_baseline !== nothing
            speedup = sabr_baseline / sabr_results["Reactant (GPU)"].time
            @printf("SABR:   ForwardDiff -> Reactant (GPU): %.2fx\n", speedup)
        end

        if haskey(heston_results, "Reactant (GPU)") && heston_baseline !== nothing
            speedup = heston_baseline / heston_results["Reactant (GPU)"].time
            @printf("Heston: ForwardDiff -> Reactant (GPU): %.2fx\n", speedup)
        end
    elseif verbose && !reactant_available
        println()
        println("Note: Reactant not loaded. To enable GPU benchmarks:")
        println("  using Reactant")
        println("  using SuperNova")
    end

    if verbose
        println()
    end

    return results
end

# ============================================================================
# Quick Benchmark (for CI)
# ============================================================================

"""
    run_quick_benchmark()

Run a quick benchmark with reduced iterations for CI/testing.
"""
function run_quick_benchmark()
    println("Quick Calibration Benchmark (reduced size)")
    run_calibration_timing_benchmark(n_strikes=10, n_expiries=3)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_calibration_timing_benchmark()
end
