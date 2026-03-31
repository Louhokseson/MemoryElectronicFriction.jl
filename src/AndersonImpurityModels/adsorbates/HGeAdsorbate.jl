"""

HGeAdsorbate is a model for hydrogen scattering on Germanium(111) surface.

This model is for impurity coupled with a constant bath with 0.49 eV bandgap, 
and the impurity DOS follows reference J. Chem. Phys. 164, 024707 (2026) https://doi.org/10.1063/5.0297254

"""

@kwdef struct HGeAdsorbate <: FrequencyDependentModel
    # Morse Potential
    m::AbstractFloat  = austrip(1.0u"u")
    Dₑ::AbstractFloat = austrip(0.0502596u"eV")
    x₀::AbstractFloat = austrip(2.07276u"Å")
    a::AbstractFloat  = austrip(2.39727u"Å^-1")
    c::AbstractFloat  = austrip(2.67532u"eV")

    # Logistic for h(x)
    D₁::AbstractFloat  = austrip(4.351u"eV")
    a′::AbstractFloat  = austrip(3.9796u"Å^-1")
    x₀′::AbstractFloat = austrip(2.2798u"Å")
    c′::AbstractFloat  = austrip(-0.33513u"eV")
    b::AbstractFloat   = 1.02971

    # Coupling Function
    Ã::AbstractFloat  = austrip(2.28499u"eV")
    L::AbstractFloat  = austrip(1.10122u"Å")
    x̃₀::AbstractFloat = austrip(2.0589u"Å")
    q::AbstractFloat  = 2.26384e-16
    
    scaledown::AbstractFloat = 1.0
    
    ## Bandgap of the substrate, unit eV
    E₉::AbstractFloat = austrip(0.49u"eV")

    ## Smearing parameter for the gapped DOS, unit eV
    η::AbstractFloat = austrip(0.02u"eV")

    ## constant density of states of the bath, unit eV^-1
    ρ₀::AbstractFloat = austrip(1.0u"eV^-1")
end

fermi(e, e0, η) = 1 / (1 + exp((e - e0)/η))
gappedDOS_smeared(ω, η, E₉) = fermi(ω, -E₉/2, η) + 1 - fermi(ω, E₉/2, η)

function HokseonReproduce.coupling_V(r::Real, adsorbate_m::HGeAdsorbate)
    Ã, L, x̃₀, q, scaledown = getfield.(Ref(adsorbate_m), (:Ã, :L, :x̃₀, :q, :scaledown))
    A = Ã .* ((1 .- q) ./2 .* (1 .- tanh((r .- x̃₀) ./ L)) .+ q)
    return A * scaledown
end

function HokseonReproduce.dV_dr(r::Real, adsorbate_m::HGeAdsorbate)
    Ã, L, x̃₀, q, scaledown = getfield.(Ref(adsorbate_m), (:Ã, :L, :x̃₀, :q, :scaledown))
    dA_dr = - (1 .- q) .* Ã ./ (2L) .* sech((r .- x̃₀) ./ L).^2
    return dA_dr * scaledown
end


function 𝓗_energyshift(r::Real, ω::Real, adsorbate_m::HGeAdsorbate)
    V_value = HokseonReproduce.coupling_V(r, adsorbate_m)
    E₉ = adsorbate_m.E₉
    η = adsorbate_m.η
    ρ₀ = adsorbate_m.ρ₀

    zp = 0.5 + im*(ω + E₉/2)/(2π*η)
    zm = 0.5 + im*(ω - E₉/2)/(2π*η)

    bracket = real(digamma(zm) - digamma(zp))

    return V_value^2 * ρ₀ * bracket
end


function HokseonReproduce.adsorbate_h(r::Real,adsorbate_m::HGeAdsorbate)
    D₁, a′, x₀′, c′, b = getfield.(Ref(adsorbate_m), (:D₁, :a′, :x₀′, :c′, :b))
    h = D₁ ./ (1 .+ exp.(-a′ .* (b .* r .- x₀′))) .+ c′
    return h
end

function HokseonReproduce.ϵₐ(r::Real, ω::Real, adsorbate_m::HGeAdsorbate)

    h = HokseonReproduce.adsorbate_h(r,adsorbate_m)

    𝓗 = 𝓗_energyshift(r, ω, adsorbate_m)

    return h + 𝓗
end

function HokseonReproduce.Δ(r::Real, ω::Real, adsorbate_m::HGeAdsorbate)
    V_value = HokseonReproduce.coupling_V(r, adsorbate_m)
    E₉ = adsorbate_m.E₉
    η = adsorbate_m.η
    ρ₀ = adsorbate_m.ρ₀
    return π * V_value^2 * gappedDOS_smeared(ω, η, E₉) * ρ₀
end


#function HokseonReproduce.DOS(r::Real, ω::Real, adsorbate_m::HGeAdsorbate)
#    ϵₐ = HokseonReproduce.ϵₐ(r, ω, adsorbate_m)

#    Δ = Δ_hybridisation(r, ω, adsorbate_m)

#    lorentzian = DistributionTools.Lorentzian(ϵₐ, Δ)

#    return lorentzian
#end

function HokseonReproduce.dh_dr(r::Real, adsorbate_m::HGeAdsorbate)
    D₁, a′, x₀′, c′, b = getfield.(Ref(adsorbate_m), (:D₁, :a′, :x₀′, :c′, :b))

    dhdr = D₁ .* a′ .* b .* exp.(-a′ .* (b .* r .- x₀′)) ./ (1 .+ exp.(-a′ .* (b .* r .- x₀′))).^2

    return dhdr
end


#function HokseonReproduce.dΔ_dr(r::Real, ω::Real, adsorbate_m::HGeAdsorbate)
#    A_value = A(r, adsorbate_m)
#    dA_value_dr = dA_dr(r, adsorbate_m)

#    E₉ = adsorbate_m.E₉
#    η = adsorbate_m.η
#    ρ₀ = adsorbate_m.ρ₀

#    gappedDOS = gappedDOS_smeared(ω, η, E₉)

#    return 2π * A_value * dA_value_dr * gappedDOS * ρ₀
#end

function HokseonReproduce.dϵₐ_dr(r::Real, ω::Real, adsorbate_m::HGeAdsorbate)
    dhdr = HokseonReproduce.dh_dr(r, adsorbate_m)

    E₉ = adsorbate_m.E₉
    η = adsorbate_m.η
    ρ₀ = adsorbate_m.ρ₀

    d𝓗dr = 2 * A(r, adsorbate_m) * dA_dr(r, adsorbate_m) * ρ₀ * real(digamma(0.5 + im*(ω - E₉/2)/(2π*η)) - digamma(0.5 + im*(ω + E₉/2)/(2π*η)))

    return dhdr + d𝓗dr
end