using DrWatson
@quickactivate "HokseonReproduce"

include(srcdir("HokseonReproduce.jl")); using .HokseonReproduce

using CairoMakie
using JamesPlots
using ColorSchemes
using Colors
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = JamesPlots.NICECOLORS;


function plot_brandbyge_dos(r = 1.0, Δϵ = 1.0)
    m = BrandbygeModels.BrandbygeAborbate()
    dos_r, dos = HokseonReproduce.DOS(r, Δϵ, m)
    ## Plotting set up
    fig = Figure(size=(JamesPlots.RESOLUTION[1]*2, 3*JamesPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="Energy / eV", ylabel= "DOS",limits=(nothing, nothing, nothing, nothing))


    lines!(ax, dos_r, dos; color = colormap[1], linewidth = 2, label = "Brandbyge \n impurity at $r")
    band!(ax, dos_r, zeros(length(dos_r)), dos; color=(colormap[1],0.3))
    Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:top, halign=:left, margin=(5, 5, 5, 5), orientation=:vertical)
    return fig
end

plot_brandbyge_dos(1, 2.0)