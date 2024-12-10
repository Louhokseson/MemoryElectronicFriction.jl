
Base.@kwdef struct BrandbygeAbsorbate <: BrandbygeModel
    Δ₀::AbstractFloat = 0.2
    β::AbstractFloat = 1.0
    ε∞::AbstractFloat = 5.0
    C::AbstractFloat = 3.0
    α::AbstractFloat = 0.5
end


function HokseonReproduce.DOS(r::Real, m::BrandbygeAbsorbate)
    """
    DOS of the impurity follows the equation (22 a,b)

    we assume the absorbate is a Cauchy/Lorentz distribution

    r: distance from the impurity to the reservoir
    
    m: BrandbygeAborbateModel
    """
    Δ = m.Δ₀ * exp(-m.β * r)
    ϵₐ = m.ε∞ - m.C * exp(-m.α * r)

    lorentzian = DistributionTools.Lorentzian(ϵₐ, Δ)
    
    return lorentzian
end