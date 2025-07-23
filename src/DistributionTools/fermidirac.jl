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