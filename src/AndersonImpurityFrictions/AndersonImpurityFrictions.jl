"""
    AndersonImpurityFrictions

Andersom Impurity type of models based on the second quantization formalism.
"""

module AndersonImpurityFrictions

# Linking the current module to the parent module 
# (parent module has exported its older silbling module-Distributions)
using ..MemoryElectronicFriction: MemoryElectronicFriction, DistributionTools , AndersonImpurityModels
using ..AndersonImpurityModels: AndersonImpurityModel
using LinearAlgebra: dot
# we use the parent module's abstract type Model and the broadcastable function


# Importing the external packages
using Unitful: @u_str
using UnitfulAtomic: austrip
using Parameters: Parameters
using LinearAlgebra: LinearAlgebra, Hermitian
using StaticArrays: SMatrix, SVector
using DrWatson

abstract type AndersonImpurityFriction <: MemoryElectronicFriction.Friction end

abstract type AndersonImpurityMemoryFriction <: AndersonImpurityFriction end

abstract type AndersonImpurityMarkovianFriction <: AndersonImpurityFriction end

memorypath = "memory/"

include(memorypath * "ImaginaryGreens.jl")
export ImaginaryGreens


include(memorypath * "WeightedEHPDOS.jl")
export WeightedEHPDOS

include(memorypath * "FrequencyLambda.jl")
export FrequencyLambda


nonmemorypath = "Markovian/"
include(nonmemorypath * "MarkovianLambda.jl")
export MarkovianLambda

end