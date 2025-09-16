using HokseonReproduce
using NQCModels
using FiniteDiff
using Test

energy = 1:5

adsorbate_m = BrandbygeAdsorbate()
bandmin = -10
bandmax = 10
nstates = 100
bath = NQCModels.FullGaussLegendre(nstates, bandmin, bandmax)

position = 1.0  # Example position

WeightedEHPDOS.R_matrix.(energy, Ref(bath), Ref(adsorbate_m), Ref(position))

WeightedEHPDOS.V′_matrix(bath, adsorbate_m, position)

function V′_numerical(bath, adsorbate_m, position)
    weights = bath.bathcoupling .^2
    height = nstates / (bandmax - bandmin)

    Δ_analytical(r) = HokseonReproduce.DOS(r, adsorbate_m).Γ

    V_analytical(r) = sqrt(Δ_analytical(r) / (height * pi)) .* (weights)

    V′_numerical = FiniteDiff.finite_difference_derivative(
        r -> V_analytical(r),
        position;
    )
        
    if length(weights) == 1
        V′_numerical = ones(length(bath.bathstates)) .* V′_numerical
    end
    return V′_numerical
end


@testset "V′_matrix check" begin
    V′_result = WeightedEHPDOS.V′_matrix(bath, adsorbate_m, position)[1,2:end]
    V′_num = V′_numerical(bath, adsorbate_m, position)

    @info "Difference: $(V′_result - V′_num)"
    @test sum(isapprox.(V′_result, V′_num; atol=1e-5)) == length(V′_result)
end
