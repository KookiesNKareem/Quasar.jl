using Test
using Quasar

@testset "Quasar.jl" begin
    include("Core/test_core.jl")
    include("AD/test_ad.jl")
    include("Instruments/test_instruments.jl")
end
