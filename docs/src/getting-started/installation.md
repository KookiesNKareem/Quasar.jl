# Installation

## Requirements

- Julia 1.10 or later
- For GPU support: CUDA-capable GPU (optional)

## Basic Installation

```julia
using Pkg
Pkg.add("QuantNova")
```

## With GPU Backends

For Enzyme (LLVM-based AD, CPU/GPU):

```julia
using Pkg
Pkg.add("Enzyme")

using Enzyme
using QuantNova
set_backend!(EnzymeBackend())
```

For Reactant (XLA compilation, GPU):

```julia
using Pkg
Pkg.add("Reactant")

using Reactant
using QuantNova
set_backend!(ReactantBackend())
```

## Verify Installation

```julia
using QuantNova

# Test basic pricing
price = black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, :call)
println("Black-Scholes price: $price")

# Test AD
f(x) = sum(x.^2)
g = gradient(f, [1.0, 2.0, 3.0])
println("Gradient: $g")
```

Expected output:
```
Black-Scholes price: 10.450583572185565
Gradient: [2.0, 4.0, 6.0]
```

## Next Steps

- [Quick Start](quickstart.md) - Learn the basics
- [AD Backends](../manual/backends.md) - Choose the right backend
- [Monte Carlo](../manual/montecarlo.md) - Price exotic options
