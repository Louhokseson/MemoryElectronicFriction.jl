using DrWatson
@quickactivate "MemoryElectronicFriction"

# making sure that MemoryElectronicFriction module is loaded once
if !isdefined(Main, :MemoryElectronicFriction)
    include(srcdir("MemoryElectronicFriction.jl"))
    using .MemoryElectronicFriction
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
    "nstates" => [30],
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

function interval_bracket!(ax, x1, x2, y; height=0.03, kwargs...)
    # vertical ticks
    linesegments!(ax, [x1, x1], [y-height, y+height]; kwargs...)
    linesegments!(ax, [x2, x2], [y-height, y+height]; kwargs...)
    # horizontal connector
    linesegments!(ax, [x1, x2], [y, y]; kwargs...)
end


function plot_Lambda_data()

    @unpack width, centre = first(params_list)
    #x_min = centre - width / 2 - 0.5
    #x_max = centre + width / 2 + 0.5
    x_min = -4.5
    x_max = 4.5


    fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="ω / eV", ylabel= "Λ(ω) / ps⁻²",limits=(x_min, x_max, 1.7, 3))
    

    for (i,params_dict) in enumerate(params_list)
        name = savename(delete!(params_dict, "energy"); allowedtypes=(Number, String, Symbol, UnionAll)) * ".txt"

        @unpack nstates, width, centre, position, temperature, discretisation = params_dict

        bandmin = - austrip(((width / 2) - centre) * u"eV")
        bandmax = austrip(((width / 2) + centre)* u"eV")
        bath = discretisation(nstates, bandmin, bandmax)

        bath_energies_au = bath.bathstates 

        scatter!(ax, ustrip.(auconvert.(u"eV", bath_energies_au)), fill(2.0, length(bath_energies_au)); color=:black, markersize=8, marker=:vline, label = (i == 1 ? "Bath states" : nothing))




        ## read text data of Λ(ω)
        path = datadir("sims", "lambda")

        data = readdlm(path * "/" * name, ' ', Float64; header=true)[1]

        energy_au = data[:, 1]

        energy_eV = ustrip.(auconvert.(u"eV", energy_au))

        Lambda_au = data[:, 2]

        Lambda_nau = Lambda_au ./ ustrip.(auconvert.(u"ps", 1.0))^2

        lines!(ax, energy_eV, Lambda_nau; color=colormap[i], linewidth=2, label="T = $(temperature)")
        

        C = 6 # tuneable constant
        ω_vec = austrip.(energy_eV .* u"eV")
        T = austrip(temperature*u"K")
        ω₁_min = minimum(ustrip.(auconvert.(u"eV", centre .- ω_vec .- C .* T)))
        ω₁_max = ustrip(auconvert(u"eV",centre + C * T))

        interval_bracket!(ax, ω₁_min, ω₁_max, 2.0;
            color = colormap[i], linewidth = 3
        )




    end

    @unpack nstates, width, centre, position, temperature = first(params_list)

    title_text = "pos=$(position) Å, nstates=$(nstates), width=$(width) eV"
    Legend(fig[1,1], ax, title_text, titleposition = :top, tellwidth=false, tellheight=false, valign=:top, halign=:left, margin=(5, 5, 5, 5), orientation=:vertical)

    return fig

end

plot_Lambda_data()


