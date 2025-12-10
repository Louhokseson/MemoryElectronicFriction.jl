using HokseonReproduce
using DrWatson
using Unitful, UnitfulAtomic
using NQCModels
using Test


function buildSystemBath(params_dict::Dict{String, Any})
    @unpack nstates, width, centre, position, discretisation, impuritymodel, temperature, energy_1, energy_2 = params_dict
    bandmin = - austrip(((width / 2) - centre) * u"eV")
    bandmax = austrip(((width / 2) + centre)* u"eV")
    bath = discretisation(nstates, bandmin, bandmax)
    adsorbate_m = eval(impuritymodel)()


    energy_1_au = austrip.(energy_1*u"eV")
    energy_2_au = austrip.(energy_2*u"eV")
    temperature_au = austrip.(temperature*u"K")
    position_au = austrip.(position*u"Å")

    return bath, adsorbate_m, position_au, energy_1_au, energy_2_au, temperature_au
end



params_list = dict_list(Dict{String, Any}(
    "nstates" => [100],
    "width" => [6],
    "discretisation" => [NQCModels.TrapezoidalRule],
    "impuritymodel" => [:BrandbygeAdsorbate],
    "centre" => [0],
    "position" => [1.5],
    "temperature" => collect(5500:-500:4000),

    ## extra [] to make collect(...) as a whole a single parameter as a whole
    "energy_1" => [2.1],
    "energy_2" => [3.000000000001],
))

# just make sure that params_list is a list with Dicts
if typeof(params_list) != Vector{Dict{String, Any}}
    params_list = [params_list]
end

bath, adsorbate_m, position_au, energy_1_au, energy_2_au, temperature_au = buildSystemBath(params_list[1])  # test that it works


@testset "Gamma vs Gamma_from_vector" begin

    println("Timing Gamma_1:")
    @time Gamma_1 = WeightedEHPDOS.Gamma(energy_1_au, energy_2_au, bath, adsorbate_m, position_au)

    println("Timing Gamma_2:")
    @time Gamma_2 = WeightedEHPDOS.Gamma_from_matrix(energy_1_au, energy_2_au, bath, adsorbate_m, position_au)

    @test isapprox(Gamma_1, Gamma_2, atol=1e-6)
    @test Gamma_1 == Gamma_2
end


@testset "R_ak_ω₁_vector" begin
    V_vec = HokseonReproduce.Vak(bath, adsorbate_m, position_au)
    bathstates = collect(bath.bathstates)

    N = length(bathstates) # bath number of states

    R_ak_ω₁_vector = [ImaginaryGreens.Rak(energy_1_au, bathstates, k, adsorbate_m, position_au, V_vec[k]) for k in 1:N]

    @info "R_ak_ω₁_vector all identical elements : $(unique(R_ak_ω₁_vector)) "

    @info "V_vec all identical elements : $(length(unique(V_vec)) == 1)"

    @test length(unique(V_vec)) == 1
end