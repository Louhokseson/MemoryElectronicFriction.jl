
Base.@kwdef struct BrandbygeAdsorbate <: AndersonImpurityModel
    Δ₀::AbstractFloat = austrip(0.2*u"eV")
    β::AbstractFloat = 1.0
    ε∞::AbstractFloat = austrip(5.0*u"eV")
    C::AbstractFloat = austrip(3.0*u"eV")
    α::AbstractFloat = 0.5
    shift::AbstractFloat = austrip(4.0*u"eV")
end


function HokseonReproduce.DOS(r::Real, m::BrandbygeAdsorbate)
    """
    DOS of the impurity follows the equation (22 a,b)

    we assume the absorbate is a Cauchy/Lorentz distribution

    r: distance from the impurity to the reservoir
    
    m: BrandbygeAdorbateModel
    """
    Δ = m.Δ₀ * exp(-m.β * r)
    ϵₐ = m.ε∞ - m.C * exp(-m.α * r) - m.shift

    lorentzian = DistributionTools.Lorentzian(ϵₐ, Δ)
    
    return lorentzian
end

function HokseonReproduce.dΔ_dr(r::Real, m::BrandbygeAdsorbate)
    """
    Broadening function for the Brandbyge adsorbate model.
    
    ω: frequency
    m: BrandbygeAdsorbate model

    return: dΔ/dr 
    """
    return -m.β * m.Δ₀ * exp(-m.β * r)
end

function HokseonReproduce.Aak(bath::WideBandBathDiscretisation, adsorbate_m::BrandbygeAdsorbate, position::Real)
    """
    Aak : Calculate the Aak vector for the constant bath and BrandbygeAdsorbate model.
    
    bath : Bath object containing bath states and coupling
    adsorbate_m : BrandbygeAdsorbate model
    position : Position of the adsorbate from substrate
    
    Returns a vector of size (length(bath.bathstates),)
    """
    
    bathstates = collect(bath.bathstates)
    Nstates = length(bathstates)
    width = bath.bathstates[end] - bath.bathstates[1]
    height = Nstates / width 

    lorentzian = HokseonReproduce.DOS(position, adsorbate_m)

    Δ = lorentzian.Γ # Lorentzian width

    Aak_vec = zeros(Float64, length(bathstates))

    Aak_vec .= bath.bathcoupling .* Δ ./ height ./ 2pi

    return Aak_vec
end

function HokseonReproduce.A′ak(bath::WideBandBathDiscretisation, adsorbate_m::BrandbygeAdsorbate, position::Real)
    """
    A′ak : Calculate the A′ak vector for the constant bath and BrandbygeAdsorbate model.
    
    bath : Bath object containing bath states and coupling
    adsorbate_m : BrandbygeAdsorbate model
    position : Position of the adsorbate from substrate
    
    Returns a vector of size (length(bath.bathstates),)
    """
    
    bathstates = collect(bath.bathstates)
    Nstates = length(bathstates)
    width = bath.bathstates[end] - bath.bathstates[1]
    height = Nstates / width

    Δ′ = HokseonReproduce.dΔ_dr(position, adsorbate_m)

    A′ak_vec = zeros(Float64, length(bathstates))

    A′ak_vec .= Δ′ .* bath.bathcoupling ./ height ./ 2pi

    return A′ak_vec
    
end