using HokseonReproduce
using Test

@testset "BrandbygeAdsorbate (parameters)" begin
    m = AndersonImpurityModels.BrandbygeAdsorbate()
    @test m.Δ₀ == 0.2
    @test m.β == 1.0
    @test m.ε∞ == 5.0
    @test m.C == 3.0
    @test m.α == 0.5
end

@testset "BrandbygeAdsorbate (DOS)" begin
    r = 1.0
    m = AndersonImpurityModels.BrandbygeAdsorbate()
    lorentzian = HokseonReproduce.DOS(r, m)

    # evaluate the DOS
    ω = range(0, 6, length=1000)
    dos = HokseonReproduce.PDF.(ω, lorentzian)

    @test length(dos) == 1000
    @test all(dos .>= 0)
end