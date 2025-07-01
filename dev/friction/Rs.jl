using DrWatson
@quickactivate "HokseonReproduce"

# making sure that HokseonReproduce module is loaded once
if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce
end

using HokseonAssistant, HokseonPlots
using Unitful, UnitfulAtomic
using NQCModels
using CairoMakie
using ColorSchemes
using Colors
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;
HokseonAssistant.julia_session()

include("parameters.jl")


function Raa(energy::Real, adsorbate_m::AndersonImpurityModel, position)
    """
    Raa : Imaginary part of retarded Green's function Gaa
    a   : Index of the adsorbate
    position    : Position of the adsorbate from substrate
    adsorbate_m : BrandbygeAdsorbate model

    return : see Eq. (A47) in paper https://doi.org/10.1103/PhysRevB.52.6042 
             and HokseonReproduce/src/DistributionTools/lorentzian.jl
    """

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
    delta_dist_approx = DistributionTools.Gaussian(bathstates[k], 0.0001)

    Raa_value = Raa(energy, adsorbate_m, position)

    A48b_bracket_left = Raa_value * (1/(energy - bathstates[k])) 
    A48b_bracket_right = Raa_value * (energy - ϵ) / (Δ) * pi * HokseonReproduce.PDF(energy, delta_dist_approx)

    return coupling_k * (A48b_bracket_left + A48b_bracket_right)
end


energy = 1
k = 3
adsorbate_m = AndersonImpurityModels.BrandbygeAdsorbate()
bathstates = bath.bathstates

coupling_k = 3.0



AndersonImpurityFrictions.ImaginaryGreens.Raa(energy,adsorbate_m,position) == Raa(energy,adsorbate_m,position)
AndersonImpurityFrictions.ImaginaryGreens.Rak(energy, bathstates, k, adsorbate_m, position, coupling_k) == Rak(energy, bathstates, k, adsorbate_m, position, coupling_k)

