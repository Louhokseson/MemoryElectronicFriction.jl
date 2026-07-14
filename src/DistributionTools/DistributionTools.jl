module DistributionTools

using ..MemoryElectronicFriction: MemoryElectronicFriction, MemoryElectronicFriction.Distribution
using Requires: Requires
using Parameters: Parameters

using Unitful: @u_str, ustrip
using UnitfulAtomic: austrip, auconvert


include("lorentzian.jl")
export Lorentzian

include("fermidirac.jl")
export FermiDirac

include("gaussian.jl")
export Gaussian


end