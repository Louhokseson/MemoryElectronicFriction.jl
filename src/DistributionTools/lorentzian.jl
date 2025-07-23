
"""
Lorentzian distribution as known as Cauchy distribution.

ω0: center frequency
Γ: half-width at half-maximum

"""
Base.@kwdef struct Lorentzian <: Distribution
    ω0::AbstractFloat = austrip(0.0*u"eV")
    Γ::AbstractFloat = austrip(1.0*u"eV")
end

function HokseonReproduce.PDF(ω::Real, d::Lorentzian)
    """
    Lorentzian probability density function

    if ω0 and Γ are in atomic units, then the PDF need to convert back to eV⁻¹
    """
    return 1/π * d.Γ / ((ω - d.ω0)^2 + d.Γ^2)
end
