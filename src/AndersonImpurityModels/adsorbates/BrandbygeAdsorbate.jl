
Base.@kwdef struct BrandbygeAdsorbate <: WideBandLimitModel
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

function HokseonReproduce.dϵₐ_dr(r::Real, m::BrandbygeAdsorbate)
    """
    Centre of the Lorentzian for the Brandbyge adsorbate model.
    
    r: distance from the impurity to the reservoir

    m: BrandbygeAdorbateModel

    return: dϵₐ/dr 
    """
    return m.α * m.C * exp(-m.α * r)
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

function HokseonReproduce.Ak(bath::WideBandBathDiscretisation, adsorbate_m::BrandbygeAdsorbate, position::Real)
    """
    Ak : Calculate the Ak coupling strength between impurity and bath in 
         second quantised Newns-Anderson Hamiltonian.
    
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



    Ak_vec = ones(Nstates) .* sqrt(Δ / (height * pi)) # coupling strength (eV)

    return Ak_vec
end

function HokseonReproduce.Vak(bath::WideBandBathDiscretisation, adsorbate_m::BrandbygeAdsorbate, position::Real)
    """
    Vak : Calculate the Vak vector for the constant bath and BrandbygeAdsorbate model 
          incoprorating the discretisation weights.
    
    bath : Bath object containing bath states and coupling
    adsorbate_m : BrandbygeAdsorbate model
    position : Position of the adsorbate from substrate
    
    Returns a vector of size (length(bath.bathstates),)
    """

    Ak_vec = HokseonReproduce.Ak(bath, adsorbate_m, position)

    ā = 1 # coupling resale (eV^{-1/2})

    Vak_vec = Ak_vec .* bath.bathcoupling .* ā  # bath in NQCModels has weight bathcoupling.^2

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

    V′ =  (Δ′/ (2 * sqrt(Δ))) / sqrt(height * pi)

    ā = 1 # coupling resale (eV^{-1/2})

    V′ak_vec .= V′ .* bath.bathcoupling .* ā # bath in NQCModels has weight bathcoupling.^2

    return V′ak_vec
    
end