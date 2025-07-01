module ImaginaryGreens
using ..HokseonReproduce: HokseonReproduce, DistributionTools
using ..HokseonReproduce.AndersonImpurityModels: AndersonImpurityModel

function Raa(energy::Real, adsorbate_m::AndersonImpurityModel, position)
    # Documentation and logic here
    lorentzian = HokseonReproduce.DOS(position, adsorbate_m)
    return HokseonReproduce.PDF(energy, lorentzian) .* 2pi
end

function Rak(energy, bathstates::Vector{Float64}, k, adsorbate_m::AndersonImpurityModel, position, coupling_k::Float64)
    """
    Rak : Imaginary part of the retarded Green's function Gʳᵉᵗak
    a   : Index of the adsorbate
    k   : Index of the substrate state

    adsorbate_m : BrandbygeAdsorbate model
    position    : Position of the adsorbate from substrate
    coupling_k  : Tₖ in Eq. (A48b) in paper https://doi.org/10.1103/PhysRevB.52.6042

    Δ   : Lorentzian width
    ϵ   : ϵₐ + ℋ (see Eq. (A47) in paper https://doi.org/10.1103/PhysRevB.52.6042)


    return : see Eq. (A48b) in paper https://doi.org/10.1103/PhysRevB.52.6042
    """
    Jₐ = HokseonReproduce.DOS(position, adsorbate_m)

    Δ = Jₐ.Γ # Lorentzian width
    ϵ = Jₐ.ω0 # Lorentzian centre

    # dirac delta function approximation
    delta_dist_approx = DistributionTools.Gaussian(bathstates[k], 1e-4)

    Raa_value = Raa(energy, adsorbate_m, position)

    A48b_bracket_left = Raa_value * (1/(energy - bathstates[k])) 
    A48b_bracket_right = Raa_value * (energy - ϵ) / (Δ) * pi * HokseonReproduce.PDF(energy, delta_dist_approx)

    return coupling_k * (A48b_bracket_left + A48b_bracket_right)
end

end # module
