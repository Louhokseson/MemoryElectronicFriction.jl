using MemoryElectronicFriction
using NQCModels
using FiniteDiff
using Test


function V′_numerical(bath::WideBandBathDiscretisation, adsorbate_m::BrandbygeAdsorbate, position)
    weights = bath.bathcoupling .^2
    bandmin = bath.bathstates[1]
    bandmax = bath.bathstates[end]
    nstates = length(bath.bathstates)
    height = nstates / (bandmax - bandmin)

    Δ_analytical(r) = MemoryElectronicFriction.DOS(r, adsorbate_m).Γ

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


@testset "V′_matrix WideBandBath BrandbygeAdsorbate" begin
    
    ## Setup
    adsorbate_m = BrandbygeAdsorbate()
    bandmin = -10
    bandmax = 10
    nstates = 100
    
    bath = NQCModels.FullGaussLegendre(nstates, bandmin, bandmax)
    position = 1.0  # Example position

    ## Test evaulation
    V′_result = WeightedEHPDOS.V′_matrix(bath, adsorbate_m, position)[1,2:end]
    V′_num = V′_numerical(bath, adsorbate_m, position)

    #@info "Difference: $(V′_result - V′_num)"
    @test sum(isapprox.(V′_result, V′_num; atol=1e-5)) == length(V′_result)
end


@testset "R_matrix coordinate check with MemoryElectronicFriction" begin

    ## setup
    energy = 1:5
    energy_length = length(energy)
    adsorbate_m = BrandbygeAdsorbate()
    bandmin = -10
    bandmax = 10
    nstates = 5
    bath = NQCModels.FullGaussLegendre(nstates, bandmin, bandmax)
    position = 1.0  # Example position
    bathstates = collect(bath.bathstates)
    coupling_vec = MemoryElectronicFriction.Vak(bath, adsorbate_m, position)


    ## test evaulation
    R_matrix_result = WeightedEHPDOS.R_matrix.(energy, Ref(bath), Ref(adsorbate_m), Ref(position))
    ## Symmetricity check
    @test all(isapprox.(R_matrix_result, transpose.(R_matrix_result); atol=1e-12, rtol=1e-8))

    ## Coordinate check where we assume that Raa and Rak and Rkk′ are correctly implemented in AndersonImpurityFrictions.ImaginaryGreens
    for i in 1:nstates+1
        for j in i:nstates+1
            if i == j == 1 
                @test sum([M[1,1] for M in R_matrix_result] .== AndersonImpurityFrictions.ImaginaryGreens.Raa.(energy, Ref(adsorbate_m), Ref(position))) == energy_length
            elseif i == 1
                @test sum([M[1,j] for M in R_matrix_result] .== AndersonImpurityFrictions.ImaginaryGreens.Rak.(energy, Ref(bathstates), Ref(j-1), Ref(adsorbate_m), Ref(position), Ref(coupling_vec[j-1]))) == energy_length
            else 
                @test sum([M[i,j] for M in R_matrix_result] .== AndersonImpurityFrictions.ImaginaryGreens.Rkk′.(energy, Ref(bathstates), Ref(i-1), Ref(j-1), Ref(adsorbate_m), Ref(position), Ref(coupling_vec[i-1]), Ref(coupling_vec[j-1]))) == energy_length
            end
        end
    end

end

@info "Test of Matrices in WeightedEHPDOS.jl finished."