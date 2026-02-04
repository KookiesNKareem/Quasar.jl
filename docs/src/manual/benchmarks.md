# Benchmark Methodology

This page documents how QuantNova performance numbers are produced and how to reproduce them.

## Principles

- **Warmup first:** all benchmarks run warmup iterations to remove JIT compilation effects.
- **Median reporting:** we report the **median** runtime across repeated runs.
- **Fixed inputs:** parameters and random seeds are fixed in each benchmark script.
- **Single-process by default:** scripts do not enable explicit multithreading or GPU backends.

## Benchmark Suites

### 1) QuantNova Comprehensive Suite (Julia)

**Script:** `benchmarks/comparison/comprehensive_benchmark.jl`

**Timing method:** `@elapsed` in Julia, times converted to microseconds, median reported.

**Key parameters (fixed in the script):**
- **European Black‑Scholes:** `S=100`, `K=100`, `T=1`, `r=0.05`, `σ=0.2`  
  Runs: 10,000 (warmup: 1,000)
- **American binomial:** 100 steps  
  Runs: 500 (warmup: 50)
- **SABR implied vol:** `F=100`, `K=100`, `T=1`, `α=0.2`, `β=0.5`, `ρ=-0.3`, `ν=0.4`  
  Runs: 10,000 (warmup: 1,000)
- **Batch pricing:** 1,000 options, `K∈[80,120]`, `T∈[0.1,2.0]`, `σ∈[0.1,0.5]`  
  `Random.seed!(42)`  
  Runs: 100 (warmup: 10)
- **Greeks (AD):** `compute_greeks` on a European option  
  Runs: 1,000 (warmup: 100)
- **Monte Carlo:** 10,000 paths, 50 steps for European and Asian  
  Runs: 20 (warmup: 2)
- **American LSM:** 10,000 paths, 50 steps  
  Runs: 10 (warmup: 2)
- **Backtesting / Factor / Statistics:** synthetic data, `Random.seed!(42)`  
  5 years of daily data (252 * 5)

**Run:**
```bash
julia --project=. benchmarks/comparison/comprehensive_benchmark.jl
```

### 2) QuantLib C++ Comparison (Direct C++)

**Scripts:**  
- `benchmarks/comparison/quantlib_benchmark.cpp`  
- `benchmarks/comparison/quantlib_benchmark_extended.cpp`

**Timing method:** `std::chrono` in C++, median reported.

**Parameters:** Same as the Julia benchmarks (e.g., `S=100`, `K=100`, `T=1`, `r=0.05`, `σ=0.2`).

**Compile & run (from script comments):**
```bash
clang++ -std=c++17 -O3 -I<HOME>/dev/QuantLib -L<HOME>/dev/QuantLib/build/ql \
  -lQuantLib -o quantlib_benchmark benchmarks/comparison/quantlib_benchmark.cpp

DYLD_LIBRARY_PATH=<HOME>/dev/QuantLib/build/ql ./quantlib_benchmark
```

### 3) QuantLib Comparison via PyCall (Julia)

**Script:** `benchmarks/comparison/quantlib_comparison.jl`

This uses **QuantLib’s Python bindings** through `PyCall`. It verifies correctness first and then benchmarks.

**Notes:**
- **Theta units differ:** QuantLib returns per‑day theta; QuantNova reports per‑year.  
  The script converts QuantNova theta to per‑day for a fair comparison.

**Run:**
```bash
julia --project=. benchmarks/comparison/quantlib_comparison.jl
```

### 4) Python Baselines (pandas / statsmodels / vectorbt)

**Script:** `benchmarks/comparison/python_benchmark.py`

**Timing method:** `time.perf_counter`, median reported.

**Dependencies:** `numpy`, `pandas`, `scipy`, `statsmodels`, optional `vectorbt`.

**Run:**
```bash
python benchmarks/comparison/python_benchmark.py
```

## Reproducibility Tips

- Run on an idle machine when possible.
- Report **CPU model, OS, and Julia version** with your results.
- If you change `JULIA_NUM_THREADS` or enable GPU backends, note it explicitly.

## Where Results Appear

Performance summaries in the README and docs are derived from these scripts and should be updated whenever benchmarks are re‑run.
