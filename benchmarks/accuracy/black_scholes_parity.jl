# Black-Scholes / Black-76 Accuracy Benchmark
# Tests put-call parity and analytical accuracy

using Quasar
using Printf

"""
    run_black_scholes_parity_benchmark(; verbose=true)

Test Black-76 implementation against put-call parity and analytical values.

Put-call parity for Black-76: C - P = exp(-rT) * (F - K)

Returns `(passed::Bool, results::Vector)`.
"""
function run_black_scholes_parity_benchmark(; verbose=true)
    results = []
    all_passed = true
    tolerance = 1e-10  # Numerical tolerance for parity check

    # Test parameters
    F_values = [90.0, 100.0, 110.0]  # Forward prices
    K_values = [80.0, 90.0, 100.0, 110.0, 120.0]  # Strikes
    T_values = [0.25, 0.5, 1.0, 2.0]  # Expiries
    r = 0.05  # Risk-free rate
    sigma_values = [0.1, 0.2, 0.3, 0.5]  # Volatilities

    if verbose
        println()
        println("Black-76 Put-Call Parity Test")
        println("=" ^ 70)
        println("Parity: C - P = exp(-rT) * (F - K)")
        println("=" ^ 70)
        @printf("%-8s %-8s %-8s %-8s %-18s %-18s %-8s\n",
                "F", "K", "T", "sigma", "C - P", "exp(-rT)*(F-K)", "Status")
        println("-" ^ 70)
    end

    test_count = 0
    pass_count = 0

    for F in F_values
        for K in K_values
            for T in T_values
                for sigma in sigma_values
                    test_count += 1

                    # Compute call and put
                    call = black76(F, K, T, r, sigma, :call)
                    put = black76(F, K, T, r, sigma, :put)

                    # Parity check
                    lhs = call - put
                    rhs = exp(-r * T) * (F - K)
                    error = abs(lhs - rhs)

                    passed = error < tolerance
                    if passed
                        pass_count += 1
                    end
                    all_passed = all_passed && passed

                    push!(results, (F=F, K=K, T=T, sigma=sigma, lhs=lhs, rhs=rhs, error=error, passed=passed))

                    # Only print failures or a sample in verbose mode
                    if verbose && (!passed || test_count <= 5)
                        status = passed ? "PASS" : "FAIL"
                        @printf("%-8.1f %-8.1f %-8.2f %-8.2f %-18.10f %-18.10f %-8s\n",
                                F, K, T, sigma, lhs, rhs, status)
                    end
                end
            end
        end
    end

    if verbose
        println("-" ^ 70)
        println("Tested $test_count combinations, $pass_count passed")
        overall = all_passed ? "PASS" : "FAIL"
        println("Overall: $overall")
        println()
    end

    return (all_passed, results)
end

"""
    run_atm_vol_test(; verbose=true)

Test that ATM options give expected prices based on vol.

For ATM options (F=K), Black-76 call = exp(-rT) * F * (2*N(d1) - 1)
where d1 = sigma * sqrt(T) / 2
"""
function run_atm_vol_test(; verbose=true)
    using Distributions: Normal, cdf

    results = []
    all_passed = true
    tolerance = 1e-12

    F = 100.0
    K = 100.0  # ATM
    r = 0.05

    if verbose
        println()
        println("ATM Option Pricing Test")
        println("=" ^ 60)
        println("F = K = $F (ATM)")
        println("=" ^ 60)
        @printf("%-8s %-8s %-18s %-18s %-8s\n", "T", "sigma", "Black76", "Analytical", "Status")
        println("-" ^ 60)
    end

    N = Normal()

    for T in [0.25, 0.5, 1.0, 2.0]
        for sigma in [0.1, 0.2, 0.3, 0.5]
            # Black-76 result
            black76_call = black76(F, K, T, r, sigma, :call)

            # Analytical ATM formula
            d1 = 0.5 * sigma * sqrt(T)
            analytical = exp(-r * T) * F * (2 * cdf(N, d1) - 1)

            error = abs(black76_call - analytical)
            passed = error < tolerance
            all_passed = all_passed && passed

            push!(results, (T=T, sigma=sigma, black76=black76_call, analytical=analytical, error=error, passed=passed))

            if verbose
                status = passed ? "PASS" : "FAIL"
                @printf("%-8.2f %-8.2f %-18.10f %-18.10f %-8s\n", T, sigma, black76_call, analytical, status)
            end
        end
    end

    if verbose
        println("-" ^ 60)
        overall = all_passed ? "PASS" : "FAIL"
        println("Overall: $overall")
        println()
    end

    return (all_passed, results)
end

"""
    run_black_scholes_benchmark(; verbose=true)

Run all Black-Scholes/Black-76 accuracy tests.
"""
function run_black_scholes_benchmark(; verbose=true)
    parity_passed, parity_results = run_black_scholes_parity_benchmark(verbose=verbose)
    atm_passed, atm_results = run_atm_vol_test(verbose=verbose)

    all_passed = parity_passed && atm_passed

    if verbose
        println()
        println("Summary")
        println("=" ^ 40)
        println("Put-Call Parity: ", parity_passed ? "PASS" : "FAIL")
        println("ATM Pricing:     ", atm_passed ? "PASS" : "FAIL")
        println("Overall:         ", all_passed ? "PASS" : "FAIL")
        println()
    end

    return (all_passed, (parity=parity_results, atm=atm_results))
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    passed, _ = run_black_scholes_benchmark(verbose=true)
    exit(passed ? 0 : 1)
end
