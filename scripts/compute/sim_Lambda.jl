using DrWatson
@quickactivate "HokseonReproduce"

# making sure that HokseonReproduce module is loaded once
if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce
end


using HokseonAssistant
using Unitful, UnitfulAtomic
using NQCModels.QuantumModels
using NQCModels
using HDF5
using DelimitedFiles
HokseonAssistant.julia_session()

"""
function sim_Lambda(energy_vec_au, bath, adsorbate_model, adsorbate_position_au, temperature_au)

    # Calculate Lambda_au for all energies in a single pass.
    Lambda_au_vec = FrequencyLambda.Lambda(energy_vec_au, bath, adsorbate_model, adsorbate_position_au, temperature_au)
    # Use broadcasting to convert the results to the desired units.
    Lambda_nau_vec = ustrip.(auconvert.(u"fs^-2", Lambda_au_vec))

    return Lambda_nau_vec, Lambda_au_vec
end
"""

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
    "discretisation" => [NQCModels.TrapezoidalRule],
    "impuritymodel" => [:BrandbygeAdsorbate],
    "centre" => [0],
    "position" => [1.0],
    "temperature" => collect(5500:-500:4000),

    ## extra [] to make collect(...) as a whole a single parameter as a whole
    "energy" => [collect(0.05:0.001:0.5)],
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

    Lambda_au =  FrequencyLambda.Lambda(energy_au, bath, adsorbate_m, position_au, temperature_au)

    header = ["energy_au" "Lambda_au"]

    full_data = vcat(header, hcat(energy_au, Lambda_au))

    writedlm(path * "/" * name, full_data, ' ')
end