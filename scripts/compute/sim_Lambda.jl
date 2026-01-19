## Only for the main process 
using Distributed
using DrWatson
@quickactivate "HokseonReproduce" ## Activate project everywhere
using HDF5
using DelimitedFiles
using HokseonAssistant
HokseonAssistant.julia_build_procs() 


# Load packages everywhere
@everywhere using HokseonReproduce
@everywhere using Unitful, UnitfulAtomic
@everywhere using NQCModels.QuantumModels
@everywhere using NQCModels


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
    "width" => [6],
    "discretisation" => [NQCModels.ShenviGaussLegendre],
    "impuritymodel" => [:BrandbygeAdsorbate],
    "centre" => [0],
    "position" => [1.0],
    "temperature" => collect(5500:-500:5500),

    ## extra [] to make collect(...) as a whole a single parameter as a whole collect(0.05:0.01:0.1)
    "energy" => [[0.05,0.5]],

    "ϵ_shift" => [1e-12, 1e-13, 1e-14],
))

# just make sure that params_list is a list with Dicts
if typeof(params_list) != Vector{Dict{String, Any}}
    params_list = [params_list]
end



for (i,params_dict) in enumerate(params_list)
    bath, adsorbate_m, position_au, energy_au, temperature_au = buildSystemBath(params_dict)

    path = mkpath(datadir("sims", "lambda", string(params_dict["impuritymodel"]), string(nameof(params_dict["discretisation"])), "nstates="*string(params_dict["nstates"]) ) )

    name = savename(delete!(params_dict, "energy"); allowedtypes=(Number, String, Symbol, UnionAll)) * ".txt"

    println(string(i) * "/" * string(length(params_list)) * " run")

    @info "Position $(params_dict["position"]) Å --- Temperature $(params_dict["temperature"]) K ---"

    ϵ_shift=params_dict["ϵ_shift"]

    Lambda_au = FrequencyLambda.Lambda(energy_au, bath, adsorbate_m, position_au, temperature_au; ϵ_shift) ## only here uses multiprocessing 

    header = ["energy_au" "Lambda_au"]

    full_data = vcat(header, hcat(energy_au, Lambda_au))

    writedlm(path * "/" * name, full_data, ' ')
end