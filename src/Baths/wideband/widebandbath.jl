module WideBand

using ..Baths:Baths, MemoryElectronicFriction.Bath


abstract type WideBandBath <: Bath end

include("wide_band_bath_discretisation.jl")

"""
export functions: using .MemoryElectronicFriction: TrapezoidalRule, ShenviGaussLegendre
"""

export TrapezoidalRule
export ShenviGaussLegendre

end