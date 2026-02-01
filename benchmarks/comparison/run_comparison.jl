# SuperNova vs QuantLib C++ Performance Comparison
#
# This script runs both the C++ QuantLib benchmark and Julia SuperNova benchmark
# and compares the results. Use this to detect performance regressions.
#
# Prerequisites:
# - QuantLib C++ must be built at ~/dev/QuantLib/build
# - The C++ benchmark must be compiled (run `make` in this directory)
#
# Usage:
#   julia --project run_comparison.jl [--compile]

using SuperNova
using Statistics

const QUANTLIB_DIR = expanduser("~/dev/QuantLib")
const BENCHMARK_DIR = @__DIR__

# Fallback QuantLib C++ timings (μs) if parsing fails
# Measured on Apple M1, QuantLib 1.41
const QUANTLIB_BASELINE = (
    european = 5.71,
    greeks = 5.71,
    american = 66.96
)

function compile_cpp_benchmark()
    cpp_file = joinpath(BENCHMARK_DIR, "quantlib_benchmark.cpp")
    exe_file = joinpath(BENCHMARK_DIR, "quantlib_benchmark")

    if !isfile(cpp_file)
        error("C++ benchmark file not found: $cpp_file")
    end

    println("Compiling QuantLib C++ benchmark...")
    cmd = ```
    clang++ -std=c++17 -O3
        -I$(QUANTLIB_DIR)
        -I/opt/homebrew/opt/boost/include
        -L$(QUANTLIB_DIR)/build/ql
        -lQuantLib
        -o $(exe_file)
        $(cpp_file)
    ```

    run(cmd)
    println("Compiled successfully.")
    return exe_file
end

function run_cpp_benchmark()
    exe_file = joinpath(BENCHMARK_DIR, "quantlib_benchmark")

    if !isfile(exe_file)
        compile_cpp_benchmark()
    end

    println("\nRunning QuantLib C++ benchmark...")
    env = copy(ENV)
    env["DYLD_LIBRARY_PATH"] = "$(QUANTLIB_DIR)/build/ql"

    output = read(setenv(`$exe_file`, env), String)
    println(output)

    # Parse timings from output
    european = match(r"European pricing\s+(\d+\.?\d*)", output)
    greeks = match(r"Greeks \(all 5\)\s+(\d+\.?\d*)", output)
    american = match(r"American \(100 steps\)\s+(\d+\.?\d*)", output)

    return (
        european = european !== nothing ? parse(Float64, european.captures[1]) : QUANTLIB_BASELINE.european,
        greeks = greeks !== nothing ? parse(Float64, greeks.captures[1]) : QUANTLIB_BASELINE.greeks,
        american = american !== nothing ? parse(Float64, american.captures[1]) : QUANTLIB_BASELINE.american
    )
end

function benchmark_nova(n_runs=1000, n_warmup=100)
    S, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.2

    state = MarketState(
        prices = Dict("T" => S),
        rates = Dict("USD" => r),
        volatilities = Dict("T" => σ),
        timestamp = 0.0
    )
    opt = EuropeanOption("T", K, T, :call)

    function bench(f, n=n_runs, w=n_warmup)
        for _ in 1:w; f(); end
        times = [(@elapsed f()) * 1e6 for _ in 1:n]
        return median(times)
    end

    println("\nRunning SuperNova.jl benchmark...")

    european = bench(() -> black_scholes(S, K, T, r, σ, :call))
    greeks = bench(() -> compute_greeks(opt, state))
    american = bench(() -> american_binomial(S, K, T, r, σ, :put, 100), 500, 50)

    return (european=european, greeks=greeks, american=american)
end

function run_comparison(; compile=false)
    println("="^70)
    println("SUPERNOVA.JL vs QUANTLIB C++ PERFORMANCE COMPARISON")
    println("="^70)

    if compile
        compile_cpp_benchmark()
    end

    ql = run_cpp_benchmark()
    nova = benchmark_nova()

    println("\n" * "="^70)
    println("RESULTS")
    println("="^70)
    println()
    println("  Benchmark              SuperNova.jl (μs)   QuantLib (μs)   Speedup")
    println("  " * "─"^60)

    function row(name, n, q)
        speedup = q / n
        status = speedup >= 1.0 ? "✓" : "✗"
        println("  $(rpad(name, 22)) $(lpad(round(n, digits=2), 8))       $(lpad(round(q, digits=2), 8))         $(lpad(round(speedup, digits=1), 5))x  $status")
        return speedup
    end

    s1 = row("European pricing", nova.european, ql.european)
    s2 = row("Greeks (all 5)", nova.greeks, ql.greeks)
    s3 = row("American (100 steps)", nova.american, ql.american)

    println()
    println("="^70)

    # Check for regressions
    all_passed = s1 >= 1.0 && s2 >= 1.0 && s3 >= 0.5  # Allow American to be up to 2x slower

    if all_passed
        println("✓ All benchmarks passed!")
    else
        println("✗ Performance regression detected!")
    end

    return (nova=nova, quantlib=ql, passed=all_passed)
end

# Main entry point
if abspath(PROGRAM_FILE) == @__FILE__
    compile = "--compile" in ARGS
    result = run_comparison(; compile=compile)
    exit(result.passed ? 0 : 1)
end
