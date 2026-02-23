## Only for the main process 
using Distributed
using DrWatson
@quickactivate "HokseonReproduce" ## Activate project everywhere
import Pkg; Pkg.precompile() ## Precompile packages in master to speed up workers' precompilation
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
    "nstates" => [38],
    "width" => [4],
    "discretisation" => [NQCModels.ShenviGaussLegendre],
    "impuritymodel" => [:BrandbygeAdsorbate],
    "centre" => [0],
    "position" => [0.5],
    "temperature" => collect(5500:-500:5500),

    ## extra [] to make collect(...) as a whole a single parameter as a whole collect(0.05:0.01:0.1)
    "energy" => [collect(1.0:0.5:2.0)],

    "ϵ_shift" => [1e-13],
    "find_sing_pts_poles" => [true, false],
))

# just make sure that params_list is a list with Dicts
if typeof(params_list) != Vector{Dict{String, Any}}
    params_list = [params_list]
end

lambda_vectors = []

for (i,params_dict) in enumerate(params_list)
    bath, adsorbate_m, position_au, energy_au, temperature_au, = buildSystemBath(params_dict)

    path = mkpath(datadir("sims", "lambda", string(params_dict["impuritymodel"]), string(nameof(params_dict["discretisation"])), "nstates="*string(params_dict["nstates"]) ) )

    name = savename(delete!(params_dict, "energy"); allowedtypes=(Number, String, Symbol, UnionAll)) * ".txt"

    println(string(i) * "/" * string(length(params_list)) * " run")

    @info "Position $(params_dict["position"]) Å --- Temperature $(params_dict["temperature"]) K ---"

    ϵ_shift=params_dict["ϵ_shift"]
    find_sing_pts_poles = params_dict["find_sing_pts_poles"]

    Lambda_au = FrequencyLambda.Lambda(energy_au, bath, adsorbate_m, position_au, temperature_au; ϵ_shift, find_sing_pts_poles) ## only here uses multiprocessing 

    push!(lambda_vectors, Lambda_au)
end

@info lambda_vectors

@info lambda_vectors[1] .- lambda_vectors[2]