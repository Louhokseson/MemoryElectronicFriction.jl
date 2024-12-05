
Base.@kwdef struct BrandbygeAborbate <: BrandbygeModel
    Δ₀::AbstractFloat = 0.2
    β::AbstractFloat = 1.0
    ε∞::AbstractFloat = 5.0
    C::AbstractFloat = 3.0
    α::AbstractFloat = 0.5
end


function HokseonReproduce.DOS(r::Real ,Δϵ::Real, m::BrandbygeAborbate)
    """
    DOS of the impurity follows the equation (22 a,b)

    we assume the absorbate is a Cauchy/Lorentz distribution

    r: distance from the impurity to the reservoir

    Δϵ: energy range for plotting the absorbate DOS
    
    m: BrandbygeAborbateModel
    """
    Δ = m.Δ₀ * exp(-m.β * r)
    ϵₐ = m.ε∞ - m.C * exp(-m.α * r)

    lorentzian = Distributions.Lorentzian(ϵₐ, Δ)

    lorentzian_pdf(ω) = HokseonReproduce.PDF.(ω, lorentzian)

    dos_r = collect(range(ϵₐ - Δϵ, ϵₐ + Δϵ, length = 100))

    dos = lorentzian_pdf.(dos_r)
    return dos_r, dos
end