"""
    WeightedEHPDOS

    Module for calculating the weighted electron-hole pair density of states and related matrices.
    
    Definition: (A33) in https://doi.org/10.1103/PhysRevB.52.6042
"""

module WeightedEHPDOS
import ..AndersonImpurityModels, ..MemoryElectronicFriction, ..ImaginaryGreens, ..dot
using ..AndersonImpurityModels: AndersonImpurityModel


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


    coupling_vec = MemoryElectronicFriction.Vak(bath, adsorbate_m, position)

    bathstates = collect(bath.bathstates)
    
    # Initialize the R matrix impurity + bath states
    R_matrix = zeros(Float64, length(bathstates)+1, length(bathstates)+1)


    # Insert the ImaginaryGreens functions Rak Rkk′Rak

    R_matrix[1,1] =  ImaginaryGreens.Raa(energy,adsorbate_m,position)

    for k in 1:length(bathstates)
        R_matrix[1,k+1] = ImaginaryGreens.Rak(energy, bathstates, k, adsorbate_m, position, coupling_vec[k])
    end
    R_matrix[2:end,1] .= R_matrix[1,2:end]  # Make it symmetric


    for k in 1:length(bathstates)
        for k′ in 1:length(bathstates)
            R_matrix[k+1,k′+1] = ImaginaryGreens.Rkk′(energy, bathstates, k, k′, adsorbate_m, position, coupling_vec[k], coupling_vec[k′])
        end
    end

    #R_matrix[2:end,2:end] .= Rkk′  # Set all substrate elements to Rkk′

    return R_matrix
end


function V′_matrix(bath, adsorbate_m::AndersonImpurityModel, position::Real)
    """
    V′_matrix : Calculate the V′ matrix for the given energy and parameters.
    
    energy : Energy value
    bathstates : Vector of bath states
    adsorbate_m : AndersonImpurityModel object
    position : Position of the adsorbate from substrate
    coupling_k : Coupling constant for the k-th substrate state
    
    Returns a matrix of size (matrix_size, matrix_size)
    """

    V′_matrix = zeros(Float64, length(bath.bathstates)+1, length(bath.bathstates)+1)
    coupling_k_vector = MemoryElectronicFriction.V′ak(bath, adsorbate_m, position)


    V′_matrix[1,2:end] = coupling_k_vector
    V′_matrix[2:end,1] = V′_matrix[1,2:end]  # Make it symmetric

    return V′_matrix

end


function Gamma_from_matrix(energy_1::Real, energy_2::Real, bath, adsorbate_m::AndersonImpurityModel, position::Real)
    """
    Gamma : Calculate the Gamma matrix from the R and V matrices. (A33) in https://doi.org/10.1103/PhysRevB.52.6042
    
    Returns a matrix of size (matrix_size, matrix_size)
    """
    
    # Calculate the Gamma value

    V′ = V′_matrix(bath, adsorbate_m, position)
    Gamma_1 = R_matrix(energy_1, bath, adsorbate_m, position) * V′
    Gamma_2 = R_matrix(energy_2, bath, adsorbate_m, position) * V′

    ## faster than tr(Gamma_1 * Gamma_2) O(n³)
    Gamma_val = dot(Gamma_1, Gamma_2')  # O(n²)

    return Gamma_val
end


function Gamma(energy_1::Real, energy_2::Real, bath, adsorbate_m::AndersonImpurityModel, position::Real)

    V′_vec = MemoryElectronicFriction.V′ak(bath, adsorbate_m, position)

    V_vec = MemoryElectronicFriction.Vak(bath, adsorbate_m, position)

    bathstates = collect(bath.bathstates)

    N = length(bathstates) # bath number of states

    R_ak_ω₁_vector = [ImaginaryGreens.Rak(energy_1, bathstates, k, adsorbate_m, position, V_vec[k]) for k in 1:N]

    R_ak_ω₂_vector = [ImaginaryGreens.Rak(energy_2, bathstates, k, adsorbate_m, position, V_vec[k]) for k in 1:N]

    ## k₁ = k₃ = a & k₂ = k₄ = a terms same value

    case₁ = dot(R_ak_ω₁_vector,V′_vec) * dot(R_ak_ω₂_vector, V′_vec) * 2 # two equvivalent terms 

    ## k₁ = k₄ = a

    case₂ = 0.0
    Rₐₐ_ω₁ = ImaginaryGreens.Raa(energy_1, adsorbate_m, position)
    for k3 in 1:N
        for k4 in 1:N
            # R_k3k4 = R_k4k3
            R_k₃k₄_ω₁ = ImaginaryGreens.Rkk′(energy_2, bathstates, k3, k4, adsorbate_m, position, V_vec[k3], V_vec[k4]) 
            case₂ += V′_vec[k3] * V′_vec[k4] * R_k₃k₄_ω₁
        end
    end
    case₂ *= Rₐₐ_ω₁


    ## k₃ = k₄ = a
    case₃ = 0.0
    Rₐₐ_ω₂ = ImaginaryGreens.Raa(energy_2, adsorbate_m, position)
    for k1 in 1:N
        for k2 in 1:N
            R_k₁k₂_ω₁ = ImaginaryGreens.Rkk′(energy_1, bathstates, k1, k2, adsorbate_m, position, V_vec[k1], V_vec[k2])
            case₃ += V′_vec[k1] * V′_vec[k2] * R_k₁k₂_ω₁
        end
    end
    case₃ *= Rₐₐ_ω₂

    #@info "case₁ = $case₁, case₂ = $case₂, case₃ = $case₃"

    return case₁ + case₂ + case₃
end


end