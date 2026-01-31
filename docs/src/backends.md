# AD Backends

Quasar provides a flexible automatic differentiation (AD) backend system that allows you to choose the best differentiation engine for your use case. All backends provide the same API, making it easy to switch between them.

## Available Backends

| Backend | Engine | Best For |
|---------|--------|----------|
| `ForwardDiffBackend()` | ForwardDiff.jl | Default choice, low-dimensional problems, nested derivatives |
| `PureJuliaBackend()` | Finite differences | Debugging, testing, reference implementation |
| `EnzymeBackend()` | Enzyme.jl (LLVM) | Large-scale problems, reverse-mode, CPU/GPU |
| `ReactantBackend()` | Reactant.jl (XLA) | GPU acceleration, high-dimensional optimization |

## Quick Start

```julia
using Quasar

# Default backend (ForwardDiff)
f(x) = sum(x.^2)
x = [1.0, 2.0, 3.0]

gradient(f, x)  # Uses ForwardDiff by default
```

## Choosing a Backend

### Per-Call Selection

```julia
# Explicit backend selection
gradient(f, x; backend=EnzymeBackend())
gradient(f, x; backend=ReactantBackend())
gradient(f, x; backend=PureJuliaBackend())
```

### Global Backend

```julia
# Set global default
set_backend!(EnzymeBackend())
gradient(f, x)  # Now uses Enzyme

# Check current backend
current_backend()  # Returns EnzymeBackend()
```

### Scoped Backend (Recommended)

```julia
# Temporarily use a different backend
result = with_backend(ReactantBackend()) do
    gradient(f, x)
    hessian(f, x)
    # All AD calls use Reactant here
end
# Original backend restored automatically
```

## Backend Details

### ForwardDiffBackend (Default)

The default backend using forward-mode AD with dual numbers.

```julia
using Quasar

gradient(f, x; backend=ForwardDiffBackend())
```

**Pros:**
- Most reliable, works with any Julia code
- Efficient for low-dimensional inputs (n < 100)
- Supports nested derivatives (Hessians via forward-over-forward)
- No compilation overhead

**Cons:**
- O(n) cost scales with input dimension
- Not optimal for high-dimensional problems

**Best for:** Options pricing, Greeks computation, small optimizations

### PureJuliaBackend

Reference implementation using finite differences.

```julia
gradient(f, x; backend=PureJuliaBackend())
```

**Pros:**
- Works with absolutely any function
- No AD-specific restrictions
- Great for debugging other backends

**Cons:**
- O(n) function evaluations
- Lower numerical precision (~1e-7)
- Slow for production use

**Best for:** Testing, debugging, validating other backends

### EnzymeBackend

LLVM-based AD that works at the compiler level.

```julia
using Enzyme  # Must load Enzyme first
using Quasar

gradient(f, x; backend=EnzymeBackend())

# Forward or reverse mode
gradient(f, x; backend=EnzymeBackend(:reverse))  # default
gradient(f, x; backend=EnzymeBackend(:forward))
```

**Pros:**
- State-of-the-art performance
- Works on CPU and GPU
- Supports mutation and complex control flow
- O(1) cost for reverse mode (independent of input dimension)

**Cons:**
- Longer compilation times
- Some operations not supported (RNG internals)
- BLAS operations may show warnings

**Best for:** Large-scale optimization, portfolio problems, production systems

**Limitation - Random Number Generators:**
```julia
# This will fail with Enzyme:
function mc_price(S0)
    paths = [randn() for _ in 1:1000]  # RNG not differentiable
    # ...
end
gradient(x -> mc_price(x[1]), [100.0]; backend=EnzymeBackend())  # Error!

# Solution: Use QMC (automatic for mc_delta/mc_greeks)
mc_delta(S0, T, payoff, dynamics; backend=EnzymeBackend())  # Works! Uses Sobol sequences
```

### ReactantBackend

XLA-based compilation for GPU acceleration.

```julia
using Reactant  # Must load Reactant first
using Quasar

gradient(f, x; backend=ReactantBackend())
```

**Pros:**
- GPU acceleration via XLA
- Optimized for large batch operations
- Automatic kernel fusion

**Cons:**
- Compilation overhead on first call
- Some reductions not supported (prod)
- Requires Reactant.jl installation

**Best for:** High-dimensional portfolio optimization, batch pricing

**Limitation - Reductions:**
```julia
# This will fail:
f(x) = prod(x)  # Multiply reduction not supported in reverse mode
gradient(f, x; backend=ReactantBackend())  # Error!

# These work fine:
f(x) = sum(x.^2)  # Sum reduction works
gradient(f, x; backend=ReactantBackend())  # OK
```

## API Reference

### Core Functions

All backends support these functions with identical signatures:

```julia
# Gradient of scalar function
gradient(f, x; backend=current_backend()) -> Vector

# Value and gradient in one pass
value_and_gradient(f, x; backend=current_backend()) -> (value, gradient)

# Hessian matrix
hessian(f, x; backend=current_backend()) -> Matrix

# Jacobian matrix
jacobian(f, x; backend=current_backend()) -> Matrix
```

### Backend Management

```julia
# Get/set global backend
current_backend() -> ADBackend
set_backend!(backend::ADBackend) -> ADBackend

# Scoped backend
with_backend(f, backend::ADBackend) -> result

# GPU initialization
enable_gpu!()           # Auto-detect
enable_gpu!(:enzyme)    # Force Enzyme
enable_gpu!(:reactant)  # Force Reactant
```

## Monte Carlo Greeks

Monte Carlo Greeks require special handling because standard RNGs are not differentiable. Quasar automatically uses Quasi-Monte Carlo (Sobol sequences) when using Enzyme:

```julia
using Enzyme
using Quasar

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
dynamics = GBMDynamics(r, sigma)
payoff = EuropeanCall(K)

# ForwardDiff: uses pseudo-random with fixed seed
delta_fd = mc_delta(S0, T, payoff, dynamics; backend=ForwardDiffBackend())

# Enzyme: automatically uses QMC (Sobol sequences)
delta_enz = mc_delta(S0, T, payoff, dynamics; backend=EnzymeBackend())

# Both give similar results
```

### Direct QMC Access

```julia
# QMC pricing (deterministic, always reproducible)
result = mc_price_qmc(S0, T, payoff, dynamics; npaths=10000)

# Generate Sobol-based normal samples
Z = sobol_normals(nsteps, npaths)  # Matrix of N(0,1) samples
```

## Performance Comparison

Typical performance characteristics (varies by problem):

| Operation | ForwardDiff | Enzyme | Reactant |
|-----------|-------------|--------|----------|
| Gradient (n=10) | 1x | 0.8x | 2x (compilation) |
| Gradient (n=1000) | 100x | 1x | 0.5x |
| Hessian (n=10) | 1x | 1.2x | 1.5x |
| Compilation | None | ~1s | ~2s |

**Rule of thumb:**
- n < 50: ForwardDiff
- n > 50, CPU: Enzyme
- n > 100, GPU available: Reactant

## Troubleshooting

### "EnzymeBackend requires Enzyme.jl to be loaded"

```julia
# Load Enzyme before using EnzymeBackend
using Enzyme
using Quasar

gradient(f, x; backend=EnzymeBackend())  # Now works
```

### "ReactantBackend requires Reactant.jl to be loaded"

```julia
using Reactant
using Quasar

gradient(f, x; backend=ReactantBackend())  # Now works
```

### Enzyme RNG Error

```
EnzymeNoDerivativeError: No augmented forward pass found for dsfmt_fill_array
```

Your function uses `randn()` or similar. Use QMC functions or ForwardDiff:

```julia
# Option 1: Use ForwardDiff for MC
mc_delta(...; backend=ForwardDiffBackend())

# Option 2: Use QMC-based functions (automatic for Enzyme)
mc_delta(...; backend=EnzymeBackend())  # Uses Sobol internally
```

### Reactant prod() Error

```
Unsupported operation in reduction rev autodiff: stablehlo.multiply
```

Reactant doesn't support `prod()` in reverse mode. Rewrite using logs:

```julia
# Instead of:
f(x) = prod(x)

# Use:
f(x) = exp(sum(log.(x)))
```

## Backend Compatibility Matrix

| Feature | ForwardDiff | PureJulia | Enzyme | Reactant |
|---------|-------------|-----------|--------|----------|
| gradient | ✅ | ✅ | ✅ | ✅ |
| value_and_gradient | ✅ | ✅ | ✅ | ✅ |
| hessian | ✅ | ✅ | ✅ | ✅* |
| jacobian | ✅ | ✅ | ✅ | ✅* |
| mc_delta | ✅ | ✅ | ✅ (QMC) | ❌ |
| mc_greeks | ✅ | ✅ | ✅ (QMC) | ❌ |
| GPU | ❌ | ❌ | ✅ | ✅ |
| prod() | ✅ | ✅ | ✅ | ❌ |
| RNG | ✅ | ✅ | ❌ | ❌ |

*Reactant hessian/jacobian use Enzyme internally
