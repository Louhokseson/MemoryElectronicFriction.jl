module MarkovianLambda
import ..HokseonReproduce, ..DistributionTools
using ..DistributionTools: FermiDirac
using ..AndersonImpurityModels: BrandbygeAdsorbate
using QuadGK


function ∂fermi(ϵ, fermidirac::FermiDirac)
    """
    Derivative of Fermi-Dirac distribution with respect to energy ϵ
    输入:
        ϵ : Energy value
        fermidirac : FermiDirac distribution object from HokseonReproduce.DistributionTools
    输出:
        ∂f : Derivative of Fermi-Dirac distribution at energy ϵ
    """

    μ = fermidirac.ϵf
    T = fermidirac.T
    βₜ = 1 / T

    ∂f = -βₜ * exp(βₜ*(ϵ-μ)) / (1 + exp(βₜ*(ϵ-μ)))^2

    return isnan(∂f) ? zero(ϵ) : ∂f
end

function widebandfriction(adsorbate_model::BrandbygeAdsorbate, r, fermidirac::FermiDirac)

    """
    Wideband Markovian friction calculation based on Brandbyge et al. model

    输入:

    adsorbate_model : BrandbygeAdsorbate model containing parameters
    r               : Position of the adsorbate from substrate
    fermidirac     : FermiDirac distribution containing Fermi level and temperature

    输出:

    Returns a scalar value representing the wideband Markovian friction at the given position.
    """


    C, α, β, Δ₀, ε∞ = getfield.(Ref(adsorbate_model), (:C, :α, :β, :Δ₀, :ε∞))

    h = ε∞ .- C .* exp.(-α .* r)
    Γ = Δ₀ .* exp.(-β .* r)

    dhdx = α .* C .* exp.(-α .* r)
    dΓdx = -β .* Δ₀ .* exp.(-β .* r)

    A(ϵ) = (1 / π) * (Γ ./ ((ϵ .- h).^2 .+ Γ.^2))
    
    kernel(ϵ) = -π * (dhdx + (ϵ .- h) .* dΓdx ./ Γ) .^ 2  .* A(ϵ).^2 * ∂fermi(ϵ, fermidirac)

    integral = quadgk(kernel, -Inf, Inf; rtol=1e-6)[1]

    return integral
end

end