Base.@kwdef struct FermiDirac <: Distribution
    ϵf::AbstractFloat = 0.0
    T::AbstractFloat = 1.0
end

function HokseonReproduce.PDF(ϵ::Real, d::FermiDirac)
    """
    Fermi-Dirac probability density function
    """
    return 1 / (exp((ϵ - d.ϵf) / d.T) + 1)
end