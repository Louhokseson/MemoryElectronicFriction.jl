using DrWatson
@quickactivate "MemoryElectronicFriction"

# making sure that MemoryElectronicFriction module is loaded once
if !isdefined(Main, :MemoryElectronicFriction)
    include(srcdir("MemoryElectronicFriction.jl"))
    using .MemoryElectronicFriction
end

using HokseonAssistant, HokseonPlots
using Unitful, UnitfulAtomic
using NQCModels
using NQCModels.DiabaticModels
using LinearAlgebra
using CairoMakie
using ColorSchemes
using Colors
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;
HokseonAssistant.julia_session()

include("parameters.jl")




function R_matrix(energy::Real, bath, adsorbate_m::AndersonImpurityModel, position::Real)
    """
    R_matrix : Calculate the R matrix for the given energy and parameters.
    
    energy : Energy value
    bathstates : Vector of bath states
    adsorbate_m : AndersonImpurityModel object
    position : Position of the adsorbate from substrate
    coupling_k : Coupling constant for the k-th substrate state
    
    Returns a matrix of size (matrix_size, matrix_size)
    """


    coupling_k = MemoryElectronicFriction.Aak(bath, adsorbate_m, position)

    bathstates = collect(bath.bathstates)
    
    # Initialize the R matrix impurity + bath states
    R_matrix = zeros(Float64, length(bathstates)+1, length(bathstates)+1)


    # Insert the ImaginaryGreens functions Rak Rkk′Rak

    R_matrix[1,1] =  AndersonImpurityFrictions.ImaginaryGreens.Raa(energy,adsorbate_m,position)

    for k in 1:length(bathstates)
        R_matrix[1,k+1] = AndersonImpurityFrictions.ImaginaryGreens.Rak(energy, bathstates, k, adsorbate_m, position, coupling_k[k])
    end
    R_matrix[2:end,1] .= R_matrix[1,2:end]  # Make it symmetric


    for k in 1:length(bathstates)
        for k′ in 1:length(bathstates)
            R_matrix[k+1,k′+1] = AndersonImpurityFrictions.ImaginaryGreens.Rkk′(energy, bathstates, k, k′, adsorbate_m, position, coupling_k[k])
        end
    end

    #R_matrix[2:end,2:end] .= Rkk′  # Set all substrate elements to Rkk′

    return R_matrix
end

function A′_matrix(bath, adsorbate_m::AndersonImpurityModel, position::Real)
    """
    A′_matrix : Calculate the A′ matrix for the given energy and parameters.
    
    energy : Energy value
    bathstates : Vector of bath states
    adsorbate_m : AndersonImpurityModel object
    position : Position of the adsorbate from substrate
    coupling_k : Coupling constant for the k-th substrate state
    
    Returns a matrix of size (matrix_size, matrix_size)
    """

    A′_matrix = zeros(Float64, length(bath.bathstates)+1, length(bath.bathstates)+1)
    coupling_k_vector = MemoryElectronicFriction.A′ak(bath, adsorbate_m, position)


    A′_matrix[1,2:end] = coupling_k_vector
    A′_matrix[2:end,1] = A′_matrix[1,2:end]  # Make it symmetric

    return A′_matrix

end


function Gamma(energy_1::Real, energy_2::Real, bath, adsorbate_m::AndersonImpurityModel, position::Real)
    """
    Gamma : Calculate the Gamma matrix from the R and V matrices. (A33) in https://doi.org/10.1103/PhysRevB.52.6042
    
    Returns a matrix of size (matrix_size, matrix_size)
    """
    
    # Calculate the Gamma value

    A′ = A′_matrix(bath, adsorbate_m, position)
    Gamma_1 = R_matrix(energy_1, bath, adsorbate_m, position) * A′
    Gamma_2 = R_matrix(energy_2, bath, adsorbate_m, position) * A′

    Gamma_val = dot(Gamma_1, Gamma_2')

    return Gamma_val
end

energy_1 = austrip.(0.5*u"eV")
energy_2 = austrip.(2.0*u"eV")


position = austrip(0.5*u"Å")

R = R_matrix(energy_1, bath, AndersonImpurityModels.BrandbygeAdsorbate(), position)


A′ = A′_matrix(bath,AndersonImpurityModels.BrandbygeAdsorbate(), position)

Gamma_val = Gamma(energy_1, energy_2, bath, AndersonImpurityModels.BrandbygeAdsorbate(), position)

@info R == AndersonImpurityFrictions.WeightedEHPDOS.R_matrix(energy_1, bath, AndersonImpurityModels.BrandbygeAdsorbate(), position)

@info A′ == AndersonImpurityFrictions.WeightedEHPDOS.A′_matrix(bath, AndersonImpurityModels.BrandbygeAdsorbate(), position)

@info Gamma_val == AndersonImpurityFrictions.WeightedEHPDOS.Gamma(energy_1, energy_2, bath, AndersonImpurityModels.BrandbygeAdsorbate(), position)
#Gamma_val