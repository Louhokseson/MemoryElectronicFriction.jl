module MarkovianLambda
import ..MemoryElectronicFriction, ..DistributionTools, ..AndersonImpurityModel
using ..DistributionTools: FermiDirac
using ..AndersonImpurityModels: BrandbygeAdsorbate, WideBandLimitModel, FrequencyDependentModel
using QuadGK

function ∂fermi(ϵ, fermidirac::FermiDirac)
    """
    Derivative of Fermi-Dirac distribution with respect to energy ϵ
    输入:
        ϵ : Energy value
        fermidirac : FermiDirac distribution object from MemoryElectronicFriction.DistributionTools
    输出:
        ∂f : Derivative of Fermi-Dirac distribution at energy ϵ
    """

    μ = fermidirac.ϵf
    T = fermidirac.T
    βₜ = 1 / T

    ∂f = -βₜ * exp(βₜ*(ϵ-μ)) / (1 + exp(βₜ*(ϵ-μ)))^2

    return isnan(∂f) ? zero(ϵ) : ∂f
end

function widebandfriction(adsorbate_m::WideBandLimitModel, r::Real, temperature::Real, fermi_level::Real=0.0)

    """
    Wideband Markovian friction calculation based Gardner et al. 2023 https://doi.org/10.1063/5.0137137

    输入:

    adsorbate_model : AndersonImpurityModel containing parameters
    r               : Position of the adsorbate from substrate
    fermidirac     : FermiDirac distribution containing Fermi level and temperature

    输出:

    Returns a scalar value representing the wideband Markovian friction at the given position.
    """


    fermidirac = DistributionTools.FermiDirac(fermi_level, temperature)

    h = MemoryElectronicFriction.adsorbate_h(r, adsorbate_m)
    Δ = MemoryElectronicFriction.Δ(r, adsorbate_m) ## Eq. (11) in https://doi.org/10.1103/PhysRevB.52.6042
    Γ = Δ * 2 ## Eq. (3) in https://doi.org/10.1063/5.0137137 

    dhdx = MemoryElectronicFriction.dϵₐ_dr(r, adsorbate_m)
    dΓdx = MemoryElectronicFriction.dΔ_dr(r, adsorbate_m) * 2

    A(ϵ) = (1 / π) * ((Γ/2) ./ ((ϵ .- h).^2 .+ (Γ/2).^2)) ## Eq.(34) in https://doi.org/10.1063/5.0137137
    
    kernel(ϵ) = -π * (dhdx + (ϵ .- h) .* dΓdx ./ Γ) .^ 2  .* A(ϵ).^2 * ∂fermi(ϵ, fermidirac) ## Eq. (33) in https://doi.org/10.1063/5.0137137

    integral,_ = quadgk(kernel, -Inf, Inf; rtol=1e-6)

    return integral
end

end