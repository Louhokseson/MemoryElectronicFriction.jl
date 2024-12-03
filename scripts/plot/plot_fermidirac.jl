using DrWatson
@quickactivate "HokseonReproduce"
include(srcdir("distribution.jl"))

using CairoMakie
using CairoMakie
using JamesPlots
using ColorSchemes
using Colors
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = JamesPlots.NICECOLORS;

function plot_fermidirac(energies=collect(range(-30,30,100)), fermi_level = 0.0,temperature = 0.000)
    
    fermi = FermiDirac.(energies, fermi_level, temperature)
    ## Plotting set up
    fig = Figure(size=(JamesPlots.RESOLUTION[1]*2, 3*JamesPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="Energy / eV", ylabel= "Probability",limits=(nothing, nothing, nothing, nothing))

    lines!(ax, energies, fermi; color = colorscheme[2], linewidth = 2, label = "Fermi distribution\n T = $(temperature) K")

    Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:bottom, halign=:left, margin=(5, 5, 5, 5), orientation=:vertical)

    return fig
end

plot_fermidirac()