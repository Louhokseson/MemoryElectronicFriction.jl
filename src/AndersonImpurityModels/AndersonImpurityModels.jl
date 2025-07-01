"""
    AndersonImpurityModels

Andersom Impurity type of models based on the second quantization formalism.
"""

module AndersonImpurityModels

# Linking the current module to the parent module 
# (parent module has exported its older silbling module-Distributions)
using ..HokseonReproduce: HokseonReproduce, DistributionTools 
# we use the parent module's abstract type Model and the broadcastable function


# Importing the external packages
using Unitful: @u_str
using UnitfulAtomic: austrip
using Parameters: Parameters
using LinearAlgebra: LinearAlgebra, Hermitian
using StaticArrays: SMatrix, SVector
using DrWatson

abstract type AndersonImpurityModel <: HokseonReproduce.Model end

adsorbatepath = "adsorbates/"

include(adsorbatepath * "BrandbygeAdsorbate.jl")
export BrandbygeAdsorbate, AndersonImpurityModel


end