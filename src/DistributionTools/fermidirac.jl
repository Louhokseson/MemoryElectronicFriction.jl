Base.@kwdef struct FermiDirac <: Distribution
    ϵf::AbstractFloat = austrip(0.0 * u"eV")  # Fermi energy
    T::AbstractFloat = austrip(300.0 * u"K")  # Temperature in Kelvin
end

function HokseonReproduce.PDF(ϵ::Real, d::FermiDirac)
    """
    Fermi-Dirac probability density function

    PDF is unitless, so called probability.
    """
    return 1 / (exp((ϵ - d.ϵf) / d.T) + 1)
end

function ∂fermi(ϵ::Real, d::FermiDirac)
    """
    Derivative of Fermi-Dirac distribution with respect to energy ϵ
    输入:
        ϵ : Energy value
        d : FermiDirac distribution object from HokseonReproduce.DistributionTools
    输出:
        ∂f : Derivative of Fermi-Dirac distribution at energy ϵ
    """

    βₜ = 1 / d.T

    ∂f = -βₜ * exp(βₜ*(ϵ-d.ϵf)) / (1 + exp(βₜ*(ϵ-d.ϵf)))^2

    return isnan(∂f) ? zero(ϵ) : ∂f
end