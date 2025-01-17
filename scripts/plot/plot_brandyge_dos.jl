using DrWatson
@quickactivate "HokseonReproduce"

# making sure that HokseonReproduce module is loaded once
if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce
end

using CairoMakie
using HokseonPlots
using ColorSchemes
using Colors
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;


function plot_brandbyge_dos(r = 1.0)
    m = BrandbygeModels.BrandbygeAdsorbate()
    lorentzian = HokseonReproduce.DOS(r, m)

    # evaluate the DOS
    ω = range(0, 6, length=1000)
    dos = HokseonReproduce.PDF.(ω, lorentzian)

    ## Plotting set up
    fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="Energy / eV", ylabel= "Density of States",limits=(nothing, nothing, nothing, nothing))


    lines!(ax, ω, dos; color = colormap[1], linewidth = 2, label = "Brandbyge \n impurity at $r")
    band!(ax, ω, zeros(length(ω)), dos; color=(colormap[1],0.3))
    Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:top, halign=:left, margin=(5, 5, 5, 5), orientation=:vertical)
    return fig
end

plot_brandbyge_dos(1.0)