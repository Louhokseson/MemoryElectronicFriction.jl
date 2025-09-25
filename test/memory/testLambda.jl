using HokseonReproduce
using NQCModels
using FiniteDiff
using Test

@info "Path of the HokseonReproduce" pathof(HokseonReproduce)

@testset "Singularities for Lambda Λ" begin
    bandmin = -5.0
    bandmax = 5.0
    nstates = 11

    ω = 0.5 
    bath = NQCModels.TrapezoidalRule(nstates, bandmin, bandmax)

    ## singularities comes from the Γ(ω₁,ω₁+ω) at A37 in https://doi.org/10.1103/PhysRevB.52.6042 
    singularities_expected = sort(collect(union(Set(bath.bathstates), Set(bath.bathstates .- ω))))

    @test singularities_expected == FrequencyLambda.singularities(bath, ω)

end

## Test for the principal value integral function with a analytical function
@testset "Principal value integration" begin

    ## function has singularities at x = 1, 2, 3 with integration as 0 from -Inf to Inf
    f(x,y) = 1/(x-1)/(x-3)/(x-2)
    principal_integration_result = FrequencyLambda.principal_value_integral(f, 1.0, [1.0, 2.0, 3.0])[1]

    @test isapprox(principal_integration_result, 0.0, atol=1e-5)

end

@testset "Interval with singularities" begin

    singularities = [0.0, 1.0, 2.0, 3.0]
    a, b = 0.0, 3.0
    a_new, b_new = FrequencyLambda.interval_limits(a, b, singularities, 1e-4)

    @test isapprox(abs(a_new - a), 1e-4, atol=1e-8) == isapprox(abs(b_new - b), 1e-4, atol=1e-8) == true
    
end


@info "testLambda.jl completed"