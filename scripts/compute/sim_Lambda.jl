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
    "nstates" => [5],
    "width" => [6],
    "temperature" => [300.0],
    "discretisation" => [NQCModels.TrapezoidalRule],
    "impuritymodel" => [:BrandbygeAdsorbate],
    "centre" => [0],
    "position" => [0.1,0.2,0.3],
    "temperature" => collect(5500:-500:4000),

    ## extra [] to make collect(...) as a whole a single parameter as a whole
    "energy" => [collect(0.05:0.001:0.5)],
))

# just make sure that params_list is a list with Dicts
if typeof(params_list) != Vector{Dict{String, Any}}
    params_list = [params_list]
end



for params_dict in params_list
    bath, adsorbate_m, position_au, energy_au, temperature_au = buildSystemBath(params_dict)
    Lambda_au_vec =  FrequencyLambda.Lambda(energy_au, bath, adsorbate_m, position_au, temperature_au)


    path = datadir("sims", "lambda")

    name = savename(params_dict; allowedtypes=(Number, String, Symbol, UnionAll)) * ".h5"

    h5write(path * "/" * name, "Lambda_au_vec", Lambda_au_vec)
end