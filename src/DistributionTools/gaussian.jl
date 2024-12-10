using Distributions: pdf, Normal

"""
    Gaussian 
    Gaussian distribution with mean μ and standard deviation σ

    You can use Gaussian to construct a Dirac delta function by setting σ → 0.
"""

Base.@kwdef struct Gaussian <: Distribution
    μ::AbstractFloat
    σ::AbstractFloat
end

function HokseonReproduce.PDF(ϵ::Real, d::Gaussian)
    """
    Gaussian probability density function

    Dirac delta PDF is defined as
        δ(x-μ) ≈ Normal(μ, σ→0)
    """
    gaussian = Normal(d.μ, d.σ)
    return pdf(gaussian, ϵ)
end