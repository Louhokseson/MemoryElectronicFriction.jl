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
using Unitful, UnitfulAtomic
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;

function plot_fermidirac(energies_au, fermi_level = 0.0,temperature = 0.000)
    
    # Fermi-Dirac distribution
    fermidirac = DistributionTools.FermiDirac(fermi_level, temperature)


    fermi_pdf = HokseonReproduce.PDF.(energies_au,fermidirac)

    ## Plotting set up
    fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="Energy / eV", ylabel= "Probability",limits=(nothing, nothing, nothing, nothing))

    lines!(ax, energies_au, fermi_pdf; color = colorscheme[2], linewidth = 2, label = "Fermi distribution\n T = $(auconvert.(u"K",temperature))")

    Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:bottom, halign=:left, margin=(5, 5, 5, 5), orientation=:vertical)

    return fig
end

energies=collect(range(-3,3,100))
energies_au = austrip.(energies*u"eV") # Convert to eV
temperature = austrip(5500*u"K")
plot_fermidirac(energies_au, 0.0, temperature)