using DrWatson
@quickactivate "HokseonReproduce"

# making sure that HokseonReproduce module is loaded only once
if !isdefined(Main, :HokseonReproduce)
    include(srcdir("HokseonReproduce.jl"))
    using .HokseonReproduce
end

# Plotting packages
using CairoMakie
using HokseonPlots
using ColorSchemes
using Colors
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;

function plot_deltagauss(energies=collect(range(-1,1,1000)), x = 0.0,σ = 1e-3)
    
    # Fermi-Dirac distribution
    deltagauss = DistributionTools.Gaussian(x, σ)

    delta_pdf = HokseonReproduce.PDF.(energies,deltagauss)
    ## Plotting set up
    fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="Energy / eV", ylabel= "Probability Density",limits=(nothing, nothing, nothing, nothing))

    lines!(ax, energies, delta_pdf; color = colorscheme[2], linewidth = 2, label = "delta Gaussian\n σ = $(σ) K")

    Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:top, halign=:left, margin=(5, 5, 5, 5), orientation=:vertical)

    return fig
end

plot_deltagauss()