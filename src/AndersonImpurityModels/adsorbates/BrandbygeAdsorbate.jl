
Base.@kwdef struct BrandbygeAdsorbate <: AndersonImpurityModel
    Δ₀::AbstractFloat = austrip(0.2*u"eV")
    β::AbstractFloat = 1.0
    ε∞::AbstractFloat = austrip(5.0*u"eV")
    C::AbstractFloat = austrip(3.0*u"eV")
    α::AbstractFloat = 0.5
    shift::AbstractFloat = austrip(0.0*u"eV")
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

function HokseonReproduce.Vak(bath::WideBandBathDiscretisation, adsorbate_m::BrandbygeAdsorbate, position::Real)
    """
    Vak : Calculate the Vak vector for the constant bath and BrandbygeAdsorbate model.
    
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

    Δ = lorentzian.Γ # Lorentzian width aka hybridisation (eV)

    Vak_vec = zeros(Float64, length(bathstates))

    A = sqrt(Δ / (height * pi * -1)) # coupling strength (eV)

    ā = 1 # coupling resale (eV^{-1/2})

    Vak_vec .= A .* bath.bathcoupling .* ā  # Impurity-bath coupling vector (eV)

    return Vak_vec
end

function HokseonReproduce.V′ak(bath::WideBandBathDiscretisation, adsorbate_m::BrandbygeAdsorbate, position::Real)
    """
    V′ak : Calculate the V′ak vector for the constant bath and BrandbygeAdsorbate model.
    
    bath : Bath object containing bath states and coupling
    adsorbate_m : BrandbygeAdsorbate model
    position : Position of the adsorbate from substrate
    
    Returns a vector of size (length(bath.bathstates),)
    """
    
    bathstates = collect(bath.bathstates)
    Nstates = length(bathstates)
    width = bath.bathstates[end] - bath.bathstates[1]
    height = Nstates / width

    Δ = HokseonReproduce.DOS(position, adsorbate_m).Γ

    Δ′ = HokseonReproduce.dΔ_dr(position, adsorbate_m)

    V′ak_vec = zeros(Float64, length(bathstates))

    V′ =  - (Δ′/ (2 * sqrt(Δ))) / sqrt(height * pi)

    ā = 1 # coupling resale (eV^{-1/2})

    V′ak_vec .= V′ .* bath.bathcoupling .* ā

    return V′ak_vec
    
end