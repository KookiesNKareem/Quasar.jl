#!/usr/bin/env julia
# SuperNova Benchmark Suite
# Entry point for running accuracy and performance benchmarks

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Try to load Reactant for GPU benchmarks (must be before SuperNova)
try
    @eval using Reactant
    @info "Reactant loaded for GPU benchmarks"
catch
    @info "Reactant not available, GPU benchmarks will be skipped"
end

using SuperNova
using Printf

# Include benchmark modules
include("accuracy/heston_reference.jl")
include("accuracy/black_scholes_parity.jl")
include("performance/calibration_timing.jl")

# ============================================================================
# Main Entry Point
# ============================================================================

function print_header()
    println()
    println("=" ^ 70)
    println("                    NOVA BENCHMARK SUITE")
    println("=" ^ 70)
    println()
end

function print_usage()
    println("""
    Usage: julia run_benchmarks.jl [OPTIONS]

    Options:
      --accuracy     Run accuracy benchmarks only
      --performance  Run performance benchmarks only
      --all          Run all benchmarks (default)
      --quick        Run quick benchmarks (reduced size, for CI)
      --help         Show this help message

    Examples:
      julia --project=.. run_benchmarks.jl --accuracy
      julia --project=.. run_benchmarks.jl --performance
      julia --project=.. run_benchmarks.jl --all
    """)
end

function run_accuracy_benchmarks(; verbose=true)
    println()
    println("=" ^ 70)
    println("                    ACCURACY BENCHMARKS")
    println("=" ^ 70)

    results = Dict{String, Bool}()

    # Heston Reference Test
    println("\n[1/2] Heston Model vs Finance Press Reference")
    heston_passed, _ = run_heston_reference_benchmark(verbose=verbose)
    results["Heston Reference"] = heston_passed

    # Black-Scholes Parity Test
    println("\n[2/2] Black-Scholes Put-Call Parity")
    bs_passed, _ = run_black_scholes_benchmark(verbose=verbose)
    results["Black-Scholes Parity"] = bs_passed

    # Summary
    println()
    println("=" ^ 70)
    println("ACCURACY BENCHMARK SUMMARY")
    println("=" ^ 70)
    all_passed = true
    for (name, passed) in results
        status = passed ? "PASS" : "FAIL"
        status_color = passed ? "" : ""  # Terminal colors could be added
        @printf("  %-30s %s\n", name, status)
        all_passed = all_passed && passed
    end
    println("-" ^ 70)
    overall = all_passed ? "PASS" : "FAIL"
    println("Overall: $overall")
    println()

    return all_passed
end

function run_performance_benchmarks(; verbose=true, quick=false)
    println()
    println("=" ^ 70)
    println("                   PERFORMANCE BENCHMARKS")
    println("=" ^ 70)

    if quick
        results = run_calibration_timing_benchmark(verbose=verbose, n_strikes=10, n_expiries=3)
    else
        results = run_calibration_timing_benchmark(verbose=verbose, n_strikes=20, n_expiries=5)
    end

    return results
end

function main(args=ARGS)
    # Parse arguments
    run_accuracy = false
    run_performance = false
    quick_mode = false
    show_help = false

    if isempty(args)
        run_accuracy = true
        run_performance = true
    else
        for arg in args
            if arg == "--accuracy"
                run_accuracy = true
            elseif arg == "--performance"
                run_performance = true
            elseif arg == "--all"
                run_accuracy = true
                run_performance = true
            elseif arg == "--quick"
                quick_mode = true
                run_accuracy = true
                run_performance = true
            elseif arg == "--help" || arg == "-h"
                show_help = true
            else
                println("Unknown argument: $arg")
                print_usage()
                return 1
            end
        end
    end

    if show_help
        print_usage()
        return 0
    end

    print_header()

    exit_code = 0

    if run_accuracy
        accuracy_passed = run_accuracy_benchmarks()
        if !accuracy_passed
            exit_code = 1
        end
    end

    if run_performance
        run_performance_benchmarks(quick=quick_mode)
    end

    println("=" ^ 70)
    println("Benchmark suite complete.")
    println("=" ^ 70)
    println()

    return exit_code
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
