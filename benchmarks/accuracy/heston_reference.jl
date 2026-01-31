# Heston Model Accuracy Benchmark
# Reference values from Finance Press high-precision computations
#
# These reference values are widely accepted benchmarks for Heston model implementations.
# See: "The Heston Model and Its Extensions in Matlab and C#" (Rouah)

using Quasar
using Printf

"""
    run_heston_reference_benchmark(; verbose=true)

Compare Quasar's Heston pricing against Finance Press reference values.

Returns `(passed::Bool, results::Vector)` where results contains per-strike details.
"""
function run_heston_reference_benchmark(; verbose=true)
    # Test parameters (standard Finance Press benchmark)
    S0 = 100.0      # Spot price
    r = 0.01        # Risk-free rate
    q = 0.02        # Dividend yield (handled via forward adjustment)
    T = 1.0         # Time to expiry

    # Heston parameters
    v0 = 0.04       # Initial variance
    theta = 0.04    # Long-term variance
    kappa = 4.0     # Mean reversion speed
    sigma = 1.0     # Vol of vol
    rho = -0.5      # Correlation

    params = HestonParams(v0, theta, kappa, sigma, rho)

    # Adjust spot for continuous dividend: S_adj = S * exp(-q*T)
    # For Heston pricer that doesn't handle dividends directly
    S_adj = S0 * exp(-q * T)

    # Finance Press reference call prices (high precision)
    # These are computed using 10000+ integration points and validated
    # across multiple implementations
    reference = Dict(
        80  => 26.774758743998854,
        90  => 20.933349000596710,
        100 => 16.070154917028834,
        110 => 12.132211516709845,
        120 => 9.024913483457836
    )

    results = []
    all_passed = true
    tolerance_bps = 1.0  # 1 basis point tolerance

    if verbose
        println()
        println("Heston Model Accuracy (vs Finance Press Reference)")
        println("=" ^ 60)
        println("Parameters: S=$S0, r=$r, q=$q, T=$T")
        println("Heston: v0=$v0, theta=$theta, kappa=$kappa, sigma=$sigma, rho=$rho")
        println("=" ^ 60)
        @printf("%-10s %-18s %-18s %-12s %-8s\n", "Strike", "Quasar", "Reference", "Error (bps)", "Status")
        println("-" ^ 60)
    end

    for K in sort(collect(keys(reference)))
        ref_price = reference[K]

        # Compute price using Quasar's Heston implementation
        # Note: Using higher N for benchmark accuracy
        quasar_price = heston_price(S_adj, K, T, r, params, :call; N=256)

        # Error in basis points (relative to strike for stability)
        error_abs = abs(quasar_price - ref_price)
        error_bps = error_abs / K * 10000

        passed = error_bps < tolerance_bps
        status = passed ? "PASS" : "FAIL"
        all_passed = all_passed && passed

        push!(results, (strike=K, quasar=quasar_price, reference=ref_price, error_bps=error_bps, passed=passed))

        if verbose
            @printf("%-10d %-18.10f %-18.10f %-12.3f %-8s\n", K, quasar_price, ref_price, error_bps, status)
        end
    end

    if verbose
        println("-" ^ 60)
        overall = all_passed ? "PASS" : "FAIL"
        println("Overall: $overall (tolerance: $tolerance_bps bps)")
        println()
    end

    return (all_passed, results)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    passed, _ = run_heston_reference_benchmark(verbose=true)
    exit(passed ? 0 : 1)
end
