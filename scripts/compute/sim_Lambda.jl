using Distributed
using DrWatson
@quickactivate "HokseonReproduce"

using HDF5
using DelimitedFiles
#using HokseonAssistant

# Activate project everywhere
@everywhere using DrWatson
@everywhere @quickactivate "HokseonReproduce"
@everywhere using HokseonReproduce
@everywhere using Unitful, UnitfulAtomic
@everywhere using NQCModels.QuantumModels
@everywhere using NQCModels
@everywhere using HokseonAssistant

HokseonAssistant.julia_session()


function buildSystemBath(params_dict::Dict{String, Any})
    @unpack nstates, width, centre, position, discretisation, impuritymodel, temperature, energy = params_dict
    bandmin = - austrip(((width / 2) - centre) * u"eV")
    bandmax = austrip(((width / 2) + centre)* u"eV")
    bath = discretisation(nstates, bandmin, bandmax)
    adsorbate_m = eval(impuritymodel)()


    energy_au = austrip.(energy*u"eV")
    temperature_au = austrip.(temperature*u"K")
    position_au = austrip.(position*u"Å")

    return bath, adsorbate_m, position_au, energy_au, temperature_au
end



params_list = dict_list(Dict{String, Any}(
    "nstates" => [20],
    "width" => [4],
    "discretisation" => [NQCModels.TrapezoidalRule],
    "impuritymodel" => [:BrandbygeAdsorbate],
    "centre" => [0],
    "position" => [1.0],
    "temperature" => collect(5500:-500:5500),

    ## extra [] to make collect(...) as a whole a single parameter as a whole
    "energy" => [collect(0.05:0.05:0.5)],
))

# just make sure that params_list is a list with Dicts
if typeof(params_list) != Vector{Dict{String, Any}}
    params_list = [params_list]
end



for (i,params_dict) in enumerate(params_list)
    bath, adsorbate_m, position_au, energy_au, temperature_au = buildSystemBath(params_dict)

    path = datadir("sims", "lambda")

    name = savename(delete!(params_dict, "energy"); allowedtypes=(Number, String, Symbol, UnionAll)) * ".txt"

    @info string(i) * "/" * string(length(params_list)) * " run"

    @info "Position $(params_dict["position"]) Å --- Temperature $(params_dict["temperature"]) K"

    Lambda_au = FrequencyLambda.Lambda(energy_au, bath, adsorbate_m, position_au, temperature_au) ## only here uses multiprocessing 

    header = ["energy_au" "Lambda_au"]

    full_data = vcat(header, hcat(energy_au, Lambda_au))

    writedlm(path * "/" * name, full_data, ' ')
end