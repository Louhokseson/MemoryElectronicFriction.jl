using Distributions: pdf, Normal

"""
    Gaussian 
    Gaussian distribution with mean μ and standard deviation σ

    You can use Gaussian to construct a Dirac delta function by setting σ → 0.
"""

Base.@kwdef struct Gaussian <: Distribution
    μ::AbstractFloat = austrip(u"eV", 1)
    σ::AbstractFloat = austrip(u"eV", 1)
end

function HokseonReproduce.PDF(ϵ::Real, d::Gaussian)
    """
    Gaussian probability density function

    Dirac delta PDF is defined as
        δ(x-μ) ≈ Normal(μ, σ→0)

    If μ and σ are in eV, then the PDF is in eV⁻¹.
    """
    gaussian = Normal(d.μ, d.σ)
    return pdf(gaussian, ϵ)
end