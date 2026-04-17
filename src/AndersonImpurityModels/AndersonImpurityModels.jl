"""
    AndersonImpurityModels

Andersom Impurity type of models based on the second quantization formalism.
"""

module AndersonImpurityModels

# Linking the current module to the parent module 
# (parent module has exported its older silbling module-Distributions)
using ..HokseonReproduce: HokseonReproduce, DistributionTools 
using NQCModels
using NQCModels.BathDiscretisations: WideBandBathDiscretisation
using NQCModels.QuantumModels
using Unitful,UnitfulAtomic
using SpecialFunctions
using StaticArrays
# we use the parent module's abstract type Model and the broadcastable function


# Importing the external packages
using Unitful: @u_str
using UnitfulAtomic: austrip
using Parameters: Parameters
using LinearAlgebra: LinearAlgebra, Hermitian
using StaticArrays: SMatrix, SVector
using DrWatson

abstract type AndersonImpurityModel <: HokseonReproduce.Model end

# --- grouped by DOF ---
abstract type AndersonImpurityModel1DOF <: AndersonImpurityModel end
abstract type AndersonImpurityModelNDOF <: AndersonImpurityModel end

# --- grouped by approximation, under the DOF intermediate types ---
abstract type WideBandLimitModel1DOF    <: AndersonImpurityModel1DOF end
abstract type FrequencyDependentModel1DOF <: AndersonImpurityModel1DOF end

abstract type WideBandLimitModelNDOF    <: AndersonImpurityModelNDOF end
abstract type FrequencyDependentModelNDOF <: AndersonImpurityModelNDOF end

# --- convenience aliases grouping by approximation across DOF ---
# (use Union rather than abstract type since single-parent rule prevents dual placement)
const WideBandLimitModel      = Union{WideBandLimitModel1DOF, WideBandLimitModelNDOF}
const FrequencyDependentModel = Union{FrequencyDependentModel1DOF, FrequencyDependentModelNDOF}

export AndersonImpurityModel, AndersonImpurityModel1DOF, AndersonImpurityModelNDOF,
       WideBandLimitModel1DOF, WideBandLimitModelNDOF,
       FrequencyDependentModel1DOF, FrequencyDependentModelNDOF,
       WideBandLimitModel, FrequencyDependentModel

adsorbatepath = "adsorbates/"

include(adsorbatepath * "BrandbygeAdsorbate.jl")
include(adsorbatepath * "ErpenbeckThossAdsorbate.jl")
include(adsorbatepath * "HGeAdsorbate.jl")
include(adsorbatepath * "NOAuAdsorbate.jl")
include(adsorbatepath * "POGOModel.jl")
export BrandbygeAdsorbate, ErpenbeckThossAdsorbate, HGeAdsorbate, NOAuAdsorbate, POGOModel



end