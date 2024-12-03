"""
    BrandbygeModels

Models and functions defined with this module are based on the work of Brandbyge et al. (1995) [1]
"""

module BrandbygeModels

# Importing the parent module
using ..HokseonReproduce: HokseonReproduce 
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

include(absorbatepath * "BrandbygeAborbate.jl")
export BrandbygeAborbate


end