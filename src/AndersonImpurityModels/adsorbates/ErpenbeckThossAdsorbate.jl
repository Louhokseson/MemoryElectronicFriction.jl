"""

ErpenbeckThossAdsorbate

This model is for impurity coupled with a constant wideband limit bath, and the impurity DOS follows reference J. Chem. Phys. 158, 064101 (2023) https://doi.org/10.1063/5.0137137

    Vₖ = V̄ₖ * ((1-q)/2*(1 - tanh((r-x̃)/ã)) + q) # unit eV^(1/2) ensures the hybridization function is independent of bath density.

    Δ = 2π * Vₖ^2  # Lorentzian width, unit eV

    ϵₐ = U₁ - U₀ is the centre of the Lorentzian, unit eV

    U₀ = Dₑ * (exp(-a * (r-x₀))-1)^2 + c

    U₁ = D₁*exp(-2a′*(r-x₀′)) - D₂*exp(-a′*(r-x₀′)) + V∞

"""

Base.@kwdef struct ErpenbeckThossAdsorbate <: WideBandLimitModel
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
    c::Union{AbstractFloat, Nothing} = nothing


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
    h = HokseonReproduce.adsorbate_h(r,adsorbate_m)

    Δ = HokseonReproduce.Δ(r,adsorbate_m)

    lorentzian = DistributionTools.Lorentzian(h, Δ)

    return lorentzian

end


function getω₀(adsorbate_m::ErpenbeckThossAdsorbate)
    (;Dₑ, a, m) = adsorbate_m
    return sqrt(2Dₑ * a^2 / m)
end

"Eq. 36"
function getλ(adsorbate_m::ErpenbeckThossAdsorbate)
    (;Dₑ, a, m) = adsorbate_m
    return sqrt(2m*Dₑ) / a
end

function eigenenergy(model::ErpenbeckThossAdsorbate, n)
    λ = getλ(model)
    n > trunc(Int, λ - 1/2) && throw(DomainError(n, "State $n is unbound!")) # Eq. 40
    ω = getω₀(model)
    harmonic = (n + 1/2)
    Eₙ = harmonic - 1/2λ * harmonic^2
    return Eₙ * ω
end

function HokseonReproduce.adsorbate_h(r::Real, adsorbate_m::ErpenbeckThossAdsorbate)
    Dₑ, x₀, a, D₁, D₂, x₀′, a′, V∞, c = getfield.(Ref(adsorbate_m), (:Dₑ, :x₀, :a, :D₁, :D₂, :x₀′, :a′, :V∞, :c))
    if isnothing(c)
        c = -eigenenergy(adsorbate_m, 0)
    end
    U₀ = Dₑ * (exp(-a * (r-x₀))-1)^2 + c
    U₁ = D₁*exp(-2a′*(r-x₀′)) - D₂*exp(-a′*(r-x₀′)) + V∞
    return U₁ - U₀
end

"""
Δ for the ErpenbeckThoss adsorbate model.

Wideband limit model: DOSof the bath ρ₀ = constant

The Hybridisation function in ErpenbeckThoss model is defined as
    Δ = π ∑ₖ |sqrt(wₖ)⋅Vₖ|²δ(ϵ - ϵₖ) 
where 
    Vₖ = V̄ₖ * ((1-q)/2*(1 - tanh((r-x̃)/ã)) + q)  [unit eV^(1/2)]
and 
    √wₖ = sqrt(1/ρ₀) (ρ₀ is the nstates / bandwidth DOS value)   [unit eV^(1/2)]

Since Vₖ is independent of ϵₖ, we now denote sqrt(wₖ)⋅Vₖ := V with unit eV.

Therefore, we arrive to 
    Δ = π|V|²∑ₖδ(ϵ - ϵₖ) = π|V|²ρ₀ = πVₖ^2.   [unit eV]

"""
function HokseonReproduce.Δ(r::Real, adsorbate_m::ErpenbeckThossAdsorbate)
    q, ã, x̃, V̄ₖ = getfield.(Ref(adsorbate_m), (:q, :ã, :x̃, :V̄ₖ))

    Vₖ = V̄ₖ * ((1-q)/2*(1 - tanh((r-x̃)/ã)) + q)
    
    return π * Vₖ^2
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