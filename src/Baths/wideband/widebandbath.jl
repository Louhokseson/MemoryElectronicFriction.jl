module WideBand

using ..Baths:Baths, HokseonReproduce.Bath


abstract type WideBandBath <: Bath end

include("wide_band_bath_discretisation.jl")

"""
export functions: using .HokseonReproduce: TrapezoidalRule, ShenviGaussLegendre
"""

export TrapezoidalRule
export ShenviGaussLegendre

end