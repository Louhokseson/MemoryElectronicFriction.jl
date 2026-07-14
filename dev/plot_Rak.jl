using DrWatson
@quickactivate "MemoryElectronicFriction"

include("dev_Gamma.jl")



# Plotting packages
using CairoMakie
using JamesPlots
using ColorSchemes
using Colors
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = JamesPlots.NICECOLORS;



function plot_Rak(energies, k, distance)
    Raks = Rak.(energies,k, distance)

    ## Plotting set up
    fig = Figure(size=(JamesPlots.RESOLUTION[1]*2, 3*JamesPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="Energy / eV", ylabel= "Coupling Density",limits=(nothing, nothing, nothing, nothing))

    lines!(ax, energies, Raks; color = colorscheme[2], linewidth = 2, label = "Ra$(k) with \n distance = $distance")
    Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:top, halign=:left, margin=(5, 5, 5, 5), orientation=:vertical)

    return fig
end

energies = collect(range(0.0, 5.0, 100))

k = 49

distance = 0.5

plot_Rak(energies, k, distance)