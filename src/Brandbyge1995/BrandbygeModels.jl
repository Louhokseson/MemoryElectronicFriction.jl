"""
    BrandbygeModels

Models and functions defined with this module are based on the work of Brandbyge et al. (1995) [1]
"""

module BrandbygeModels

# Linking the current module to the parent module 
# (parent module has exported its older silbling module-Distributions)
using ..HokseonReproduce: HokseonReproduce, Distributions 
# we use the parent module's abstract type Model and the broadcastable function


# Importing the external packages
using Unitful: @u_str
using UnitfulAtomic: austrip
using Parameters: Parameters
using LinearAlgebra: LinearAlgebra, Hermitian
using StaticArrays: SMatrix, SVector
using DrWatson

abstract type BrandbygeModel <: HokseonReproduce.Model end

absorbatepath = "absorbate_dist/"

include(absorbatepath * "BrandbygeAbsorbate.jl")
export BrandbygeAbsorbate


end