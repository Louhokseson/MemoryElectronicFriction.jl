using DrWatson
@quickactivate "HokseonReproduce"

# making sure that HokseonReproduce module is loaded once
if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce
end

using HokseonAssistant
using Unitful, UnitfulAtomic
using NQCModels

HokseonAssistant.julia_session()

all_params = Dict{String, Any}(
    "nstates" => 20,
    "width" => 6,
    "centre" => 0,
    "position" => collect(0.1:0.1:0.5),
)

params_list = dict_list(all_params)
# just make sure that params_list is a list with Dicts
if !(params_list isa Vector)
    params_list = [params_list]
end

@unpack nstates, width, centre, position = all_params
bandmin = - austrip(((width / 2) - centre) * u"eV")
bandmax = austrip(((width / 2) + centre)* u"eV")
bath = NQCModels.TrapezoidalRule(nstates, bandmin, bandmax)
position_au = austrip.(position*u"Å")
energies = collect(0.05:0.001:0.074)
energies_au = austrip.(energies*u"eV")
temperatures = 5500
temperatures_au = austrip.(temperatures*u"K")
adsorbate_m = AndersonImpurityModels.BrandbygeAdsorbate()

function sim_Lambda(energy_vec_au, bath, adsorbate_model, adsorbate_position_au, temperature_au)
    # Calculate Lambda_au for all energies in a single pass.

    Lambda_au_vec = FrequencyLambda.Lambda(energy_vec_au, bath, adsorbate_model, adsorbate_position_au, temperature_au)
    # Use broadcasting to convert the results to the desired units.
    Lambda_nau_vec = ustrip.(auconvert.(u"fs^-2", Lambda_au_vec))

    return Lambda_nau_vec, Lambda_au_vec
end

energy = [energies_au[2]]

"""
fermidirac = DistributionTools.FermiDirac(0.0, temperatures_au)
bounds = FrequencyLambda.effective_bounds(bath, fermidirac, energy)

bath_sing_pts = FrequencyLambda.bath_singularities(bath, energy)

ω₁_range_min, ω₁_range_max = FrequencyLambda.ω₁_effective_range(fermidirac, energy)

midpoints = FrequencyLambda.piecewise_midpoint_cauchy_interval(bath_sing_pts)
"""

@info "Starting Lambda calculation for energy = $energy au"

t = @elapsed begin
    Lambda_nau_vec, Lambda_au_vec =
        sim_Lambda(energy, bath, adsorbate_m, position_au[1], temperatures_au)
end

@info "Lambda in au: $Lambda_au_vec"

@info "Lambda call took $t seconds"



