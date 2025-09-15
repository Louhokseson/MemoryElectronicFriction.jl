using HokseonReproduce
using FiniteDiff
using Unitful, UnitfulAtomic
using Test

@testset "BrandbygeAdsorbate (parameters)" begin
    m = AndersonImpurityModels.BrandbygeAdsorbate()
    @test m.Δ₀ == austrip(0.2*u"eV")
    @test m.β == 1.0
    @test m.ε∞ == austrip(5.0*u"eV")
    @test m.C == austrip(3.0*u"eV")
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


@testset "BrandbygeAdsorbate (dV_dx)" begin
    r = 5.0
    m = AndersonImpurityModels.BrandbygeAdsorbate()

    Δ′_analytical = HokseonReproduce.dΔ_dr(r, m)

    Δ′_numerical = FiniteDiff.finite_difference_derivative(x -> HokseonReproduce.DOS(x, m).Γ, r)

    @info "Analytical dΔ/dr: $Δ′_analytical"
    @info "Numerical dΔ/dr: $Δ′_numerical"

    @test isapprox(Δ′_analytical, Δ′_numerical; atol=1e-5)
end