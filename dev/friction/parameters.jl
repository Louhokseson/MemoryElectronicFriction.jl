using DrWatson
### Parameters ###
all_params = Dict{String, Any}(
    "nstates" => [150],
    "width" => [50],
    "temperature" => [300.0],
    "discretisation" => [:GapGaussLegendre],
    "impuritymodel" => :Hokseon,
    "gap" => [0.49],
    "centre" => [0],
    "position" => [1.0],
)

params_list = dict_list(all_params)
# just make sure that params_list is a list with Dicts
if typeof(params_list) != Vector{Dict{String, Any}}
    params_list = [params_list]
end

@unpack nstates, width, temperature, discretisation, impuritymodel, gap, centre, position = params_list[1]
bandmin = - austrip(((width / 2) - centre) * u"eV")
bandmax = austrip(((width / 2) + centre)* u"eV")
bath = GapGaussLegendre(nstates, bandmin, bandmax, austrip(gap * u"eV"))