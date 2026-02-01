# Heston Model Accuracy Benchmark
# Tests internal consistency via put-call parity and convergence behavior
#
# Reference: Heston (1993), Gatheral "The Volatility Surface"

using SuperNova
using Printf

"""
    run_heston_reference_benchmark(; verbose=true)

Test Heston model implementation accuracy via:
1. Put-call parity: C - P = S - K*exp(-rT)
2. Convergence: prices stabilize with increasing integration points
3. Boundary conditions: deep ITM/OTM behavior

Returns `(passed::Bool, results::Dict)` with detailed test results.
"""
function run_heston_reference_benchmark(; verbose=true)
    # Standard test parameters (Gatheral-style)
    S0 = 100.0      # Spot price
    r = 0.05        # Risk-free rate
    T = 1.0         # Time to expiry

    # Heston parameters (moderate vol-of-vol, negative correlation)
    v0 = 0.04       # Initial variance (σ₀ = 20%)
    theta = 0.04    # Long-term variance
    kappa = 1.5     # Mean reversion speed
    sigma = 0.3     # Vol of vol
    rho = -0.9      # Correlation (equity-like skew)

    params = HestonParams(v0, theta, kappa, sigma, rho)

    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]

    results = Dict{String, Any}()
    all_passed = true

    if verbose
        println()
        println("Heston Model Accuracy Benchmark")
        println("=" ^ 70)
        println("Parameters: S=$S0, r=$r, T=$T")
        println("Heston: v0=$v0, theta=$theta, kappa=$kappa, sigma=$sigma, rho=$rho")
        println("=" ^ 70)
    end

    # =========================================================================
    # Test 1: Put-Call Parity
    # =========================================================================
    if verbose
        println()
        println("Test 1: Put-Call Parity (C - P = S - K*exp(-rT))")
        println("-" ^ 70)
        @printf("%-10s %-12s %-12s %-12s %-12s %-8s\n",
                "Strike", "Call", "Put", "C-P", "S-Ke^(-rT)", "Status")
        println("-" ^ 70)
    end

    parity_results = []
    parity_tolerance = 1e-8
    parity_passed = true

    for K in strikes
        call = heston_price(S0, K, T, r, params, :call; N=256)
        put = heston_price(S0, K, T, r, params, :put; N=256)

        lhs = call - put
        rhs = S0 - K * exp(-r * T)
        error = abs(lhs - rhs)

        passed = error < parity_tolerance
        parity_passed = parity_passed && passed
        status = passed ? "PASS" : "FAIL"

        push!(parity_results, (K=K, call=call, put=put, lhs=lhs, rhs=rhs, error=error, passed=passed))

        if verbose
            @printf("%-10.0f %-12.6f %-12.6f %-12.6f %-12.6f %-8s\n",
                    K, call, put, lhs, rhs, status)
        end
    end

    results["parity"] = parity_results
    all_passed = all_passed && parity_passed

    if verbose
        println("-" ^ 70)
        println("Put-Call Parity: ", parity_passed ? "PASS" : "FAIL",
                " (tolerance: ", parity_tolerance, ")")
    end

    # =========================================================================
    # Test 2: Convergence with Integration Points
    # =========================================================================
    if verbose
        println()
        println("Test 2: Convergence (ATM Call with increasing N)")
        println("-" ^ 70)
        @printf("%-10s %-18s %-18s %-12s\n", "N", "Price", "Change", "Status")
        println("-" ^ 70)
    end

    K_atm = 100.0
    N_values = [32, 64, 128, 256, 512]
    convergence_results = []
    convergence_tolerance = 0.001  # Price should stabilize to 0.1% relative change
    convergence_passed = true
    prev_price = nothing

    for N in N_values
        price = heston_price(S0, K_atm, T, r, params, :call; N=N)

        if prev_price === nothing
            change = NaN
            rel_change = NaN
            passed = true
            status = "-"
        else
            change = price - prev_price
            rel_change = abs(change / prev_price)
            passed = N < 256 || rel_change < convergence_tolerance
            convergence_passed = convergence_passed && passed
            status = passed ? "PASS" : "FAIL"
        end

        push!(convergence_results, (N=N, price=price, change=change, passed=passed))

        if verbose
            if isnan(change)
                @printf("%-10d %-18.10f %-18s %-12s\n", N, price, "-", status)
            else
                @printf("%-10d %-18.10f %-18.10f %-12s\n", N, price, change, status)
            end
        end

        prev_price = price
    end

    results["convergence"] = convergence_results
    all_passed = all_passed && convergence_passed

    if verbose
        println("-" ^ 70)
        println("Convergence: ", convergence_passed ? "PASS" : "FAIL",
                " (tolerance: ", convergence_tolerance * 100, "% change at N>=256)")
    end

    # =========================================================================
    # Test 3: Boundary Behavior
    # =========================================================================
    if verbose
        println()
        println("Test 3: Boundary Conditions")
        println("-" ^ 70)
    end

    boundary_passed = true
    boundary_results = []

    # Deep ITM call should be close to S - K*exp(-rT)
    K_deep_itm = 50.0
    call_deep_itm = heston_price(S0, K_deep_itm, T, r, params, :call; N=256)
    intrinsic_itm = S0 - K_deep_itm * exp(-r * T)
    itm_error = abs(call_deep_itm - intrinsic_itm) / intrinsic_itm
    itm_passed = itm_error < 0.01  # Within 1% of intrinsic
    boundary_passed = boundary_passed && itm_passed

    push!(boundary_results, (test="Deep ITM Call (K=50)", value=call_deep_itm,
                             expected=intrinsic_itm, rel_error=itm_error, passed=itm_passed))

    if verbose
        @printf("Deep ITM Call (K=50): %.6f (expected ~%.6f, error: %.4f%%) %s\n",
                call_deep_itm, intrinsic_itm, itm_error * 100, itm_passed ? "PASS" : "FAIL")
    end

    # Deep OTM call should be close to 0
    K_deep_otm = 200.0
    call_deep_otm = heston_price(S0, K_deep_otm, T, r, params, :call; N=256)
    otm_passed = call_deep_otm < 0.01  # Essentially zero
    boundary_passed = boundary_passed && otm_passed

    push!(boundary_results, (test="Deep OTM Call (K=200)", value=call_deep_otm,
                             expected=0.0, rel_error=call_deep_otm, passed=otm_passed))

    if verbose
        @printf("Deep OTM Call (K=200): %.6f (expected ~0, threshold: 0.01) %s\n",
                call_deep_otm, otm_passed ? "PASS" : "FAIL")
    end

    # Non-negative prices
    all_positive = all(r -> r.call >= 0 && r.put >= 0, parity_results)
    boundary_passed = boundary_passed && all_positive

    if verbose
        @printf("All prices non-negative: %s\n", all_positive ? "PASS" : "FAIL")
    end

    results["boundary"] = boundary_results
    all_passed = all_passed && boundary_passed

    if verbose
        println("-" ^ 70)
        println("Boundary Conditions: ", boundary_passed ? "PASS" : "FAIL")
    end

    # =========================================================================
    # Test 4: Dividend Yield Support
    # =========================================================================
    if verbose
        println()
        println("Test 4: Dividend Yield Support")
        println("-" ^ 70)
    end

    dividend_passed = true
    dividend_results = []
    q = 0.03  # 3% dividend yield

    # Test 4a: Put-call parity with dividends
    # C - P = S*exp(-qT) - K*exp(-rT)
    if verbose
        println("4a. Put-Call Parity with Dividends (q=$q)")
        @printf("    C - P = S*exp(-qT) - K*exp(-rT)\n")
        println("-" ^ 70)
        @printf("%-10s %-12s %-12s %-12s %-12s %-8s\n",
                "Strike", "Call", "Put", "C-P", "Expected", "Status")
        println("-" ^ 70)
    end

    for K in strikes
        call_div = heston_price(S0, K, T, r, q, params, :call; N=256)
        put_div = heston_price(S0, K, T, r, q, params, :put; N=256)

        lhs = call_div - put_div
        rhs = S0 * exp(-q * T) - K * exp(-r * T)
        error = abs(lhs - rhs)

        passed = error < parity_tolerance
        dividend_passed = dividend_passed && passed
        status = passed ? "PASS" : "FAIL"

        push!(dividend_results, (K=K, call=call_div, put=put_div, lhs=lhs, rhs=rhs, error=error, passed=passed))

        if verbose
            @printf("%-10.0f %-12.6f %-12.6f %-12.6f %-12.6f %-8s\n",
                    K, call_div, put_div, lhs, rhs, status)
        end
    end

    # Test 4b: Dividends should reduce call prices
    if verbose
        println()
        println("4b. Dividend Effect on Call Prices")
        println("-" ^ 70)
        @printf("%-10s %-15s %-15s %-15s %-8s\n",
                "Strike", "No Dividend", "With q=$q", "Difference", "Status")
        println("-" ^ 70)
    end

    for K in strikes
        call_no_div = heston_price(S0, K, T, r, params, :call; N=256)
        call_with_div = heston_price(S0, K, T, r, q, params, :call; N=256)
        diff = call_no_div - call_with_div

        # Dividends should reduce call prices
        passed = diff > 0
        dividend_passed = dividend_passed && passed
        status = passed ? "PASS" : "FAIL"

        if verbose
            @printf("%-10.0f %-15.6f %-15.6f %-15.6f %-8s\n",
                    K, call_no_div, call_with_div, diff, status)
        end
    end

    results["dividend"] = dividend_results
    all_passed = all_passed && dividend_passed

    if verbose
        println("-" ^ 70)
        println("Dividend Support: ", dividend_passed ? "PASS" : "FAIL")
    end

    # =========================================================================
    # Summary
    # =========================================================================
    if verbose
        println()
        println("=" ^ 70)
        println("SUMMARY")
        println("=" ^ 70)
        println("Put-Call Parity:     ", parity_passed ? "PASS" : "FAIL")
        println("Convergence:         ", convergence_passed ? "PASS" : "FAIL")
        println("Boundary Conditions: ", boundary_passed ? "PASS" : "FAIL")
        println("Dividend Support:    ", dividend_passed ? "PASS" : "FAIL")
        println("-" ^ 70)
        println("Overall: ", all_passed ? "PASS" : "FAIL")
        println()
    end

    return (all_passed, results)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    passed, _ = run_heston_reference_benchmark(verbose=true)
    exit(passed ? 0 : 1)
end
