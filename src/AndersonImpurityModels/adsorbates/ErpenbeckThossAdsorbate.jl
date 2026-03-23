"""

ErpenbeckThossAdsorbate

This model is for impurity coupled with a constant wideband limit bath, and the impurity DOS follows reference J. Chem. Phys. 158, 064101 (2023) https://doi.org/10.1063/5.0137137

    Vₖ = V̄ₖ * ((1-q)/2*(1 - tanh((r-x̃)/ã)) + q) # unit eV^(1/2) ensures the hybridization function is independent of bath density.

    Δ = 2π * Vₖ^2  # Lorentzian width, unit eV

    ϵₐ = U₁ - U₀ is the centre of the Lorentzian, unit eV

    U₀ = Dₑ * (exp(-a * (r-x₀))-1)^2 + c

    U₁ = D₁*exp(-2a′*(r-x₀′)) - D₂*exp(-a′*(r-x₀′)) + V∞

"""

Base.@kwdef struct ErpenbeckThossAdsorbate <: AndersonImpurityModel
    Γ::AbstractFloat
    m::AbstractFloat = austrip(1.0u"u")
    Dₑ::AbstractFloat = austrip(3.52u"eV")
    x₀::AbstractFloat = austrip(1.78u"Å")
    a::AbstractFloat = austrip(1.7361u"Å^-1")
    D₁::AbstractFloat = austrip(4.52u"eV")
    D₂::AbstractFloat = austrip(0.79u"eV")
    x₀′::AbstractFloat = x₀
    a′::AbstractFloat = austrip(1.379u"Å^-1")
    V∞::AbstractFloat = austrip(-1.5u"eV")
    c::AbstractFloat = austrip(-45.7u"meV")

    q::AbstractFloat = 0.05
    ã::AbstractFloat = austrip(0.5u"Å")
    x̃::AbstractFloat = austrip(3.5u"Å")
    V̄ₖ::AbstractFloat = sqrt(austrip(Γ)/2π)

end


"""
DOS of the impurity follows reference J. Chem. Phys. 158, 064101 (2023) https://doi.org/10.1063/5.0137137

The adsorbate is coupled with a constant wideband limit bath.

r: distance from the impurity to the reservoir

adsorbate_m: ErpenbeckThossAdsorbate
"""
function HokseonReproduce.DOS(r::Real, adsorbate_m::ErpenbeckThossAdsorbate)
    Dₑ, x₀, a, D₁, D₂, x₀′, a′, V∞, c, q, ã, x̃, V̄ₖ = getfield.(Ref(adsorbate_m), (:Dₑ, :x₀, :a, :D₁, :D₂, :x₀′, :a′, :V∞, :c, :q, :ã, :x̃, :V̄ₖ))

    U₀ = Dₑ * (exp(-a * (r-x₀))-1)^2 + c

    U₁ = D₁*exp(-2a′*(r-x₀′)) - D₂*exp(-a′*(r-x₀′)) + V∞

    Vₖ = V̄ₖ * ((1-q)/2*(1 - tanh((r-x̃)/ã)) + q) # unit eV^(1/2)

    ## centre of the Lorentzian
    ϵₐ = U₁ - U₀ 

    ## Lorentzian width: delta function has eV^-1 unit
    Δ = π * Vₖ^2

    lorentzian = DistributionTools.Lorentzian(ϵₐ, Δ)

    return lorentzian

end

"""
dΔ/dr for the ErpenbeckThoss adsorbate model.

adsorbate_m: ErpenbeckThossAdsorbate model

return: dΔ/dr 
"""
function HokseonReproduce.dΔ_dr(r::Real, adsorbate_m::ErpenbeckThossAdsorbate)
    q, ã, x̃, V̄ₖ = getfield.(Ref(adsorbate_m), (:q, :ã, :x̃, :V̄ₖ))

    V = V̄ₖ * ((1-q)/2 * (1 - tanh((r-x̃)/ã)) + q)

    dVdr = V̄ₖ * (1-q)/2 * (-sech((r-x̃)/ã)^2) / ã

    return 2π * V * dVdr
end

"""
dϵₐ/dr for the ErpenbeckThoss adsorbate model.

adsorbate_m: ErpenbeckThossAdsorbate model

ϵₐ = U₁ - U₀ is the centre of the Lorentzian.

return: dϵₐ/dr 
"""
function HokseonReproduce.dϵₐ_dr(r::Real, adsorbate_m::ErpenbeckThossAdsorbate)
    Dₑ, x₀, a, D₁, D₂, x₀′, a′ = getfield.(Ref(adsorbate_m), (:Dₑ, :x₀, :a, :D₁, :D₂, :x₀′, :a′))

    dU₀dr = 2Dₑ * (exp(-a*(r-x₀)) - 1) * (-a) * exp(-a*(r-x₀))

    dU₁dr = -2a′*D₁*exp(-2a′*(r-x₀′)) + a′*D₂*exp(-a′*(r-x₀′))

    return dU₁dr - dU₀dr
end