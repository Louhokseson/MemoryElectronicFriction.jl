
"""
Lorentzian distribution

ω0: center frequency
Γ: half-width at half-maximum

"""
Base.@kwdef struct Lorentzian <: Distribution
    ω0::AbstractFloat = 0.0
    Γ::AbstractFloat = 1.0
end

function HokseonReproduce.PDF(ω::Real, d::Lorentzian)
    """
    Lorentzian probability density function
    """
    return 1/π * d.Γ / ((ω - d.ω0)^2 + d.Γ^2)
end
