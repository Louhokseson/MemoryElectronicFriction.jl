using DrWatson
@quickactivate "HokseonReproduce"

# making sure that HokseonReproduce module is loaded once
if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce
end

using CairoMakie
using HokseonPlots
using HokseonAssistant
using ColorSchemes
using Colors
using Unitful, UnitfulAtomic
using DelimitedFiles
using LaTeXStrings, Printf
using NQCModels
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;
HokseonAssistant.julia_session()


params_list = dict_list(Dict{String, Any}(
    "nstates" => [20],
    "width" => [6],
    "discretisation" => [NQCModels.TrapezoidalRule],
    "impuritymodel" => [:BrandbygeAdsorbate],
    "centre" => [0],
    "position" => [1.0],
    "temperature" => collect(5500:-500:5500),

    ## extra [] to make collect(...) as a whole a single parameter as a whole
    "energy" => [collect(0.05:0.001:0.5)],
))

# just make sure that params_list is a list with Dicts
if typeof(params_list) != Vector{Dict{String, Any}}
    params_list = [params_list]
end


function plot_Lambda_data()

    fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="Energy / eV", ylabel= "Lambda / a.u.⁻²",limits=(nothing, nothing, nothing, nothing))
    

    for params_dict in params_list
        name = savename(delete!(params_dict, "energy"); allowedtypes=(Number, String, Symbol, UnionAll)) * ".txt"

        @unpack nstates, width, centre, position, temperature = params_dict

        path = datadir("sims", "lambda")

        data = readdlm(path * "/" * name, ' ', Float64; header=true)[1]

        energy_au = data[:, 1]

        energy_eV = ustrip.(auconvert.(u"eV", energy_au))

        Lambda_au = data[:, 2]

        lines!(ax, energy_eV, Lambda_au; color=colormap[2], linewidth=2, label="T = $(temperature) K")

        title_text = "pos=$(position) Å, nstates=$(nstates), width=$(width) eV"

    Legend(fig[1,1], ax, title_text, titleposition = :top, tellwidth=false, tellheight=false, valign=:top, halign=:left, margin=(5, 5, 5, 5), orientation=:vertical)
    end


    return fig

end

plot_Lambda_data()


