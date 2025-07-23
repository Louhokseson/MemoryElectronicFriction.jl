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



function plot_lorentzian(center_au, width_au)

    energies = collect(range(-10, 10, 100000)) # Energy range in eV
    energies_au = austrip.(energies*u"eV") # Convert to a.u.
    
    # Fermi-Dirac distribution
    lorentzian = DistributionTools.Lorentzian(center_au, width_au)


    lorentzian_pdf_au = HokseonReproduce.PDF.(energies_au,lorentzian)

    lorentzian_pdf = ustrip.(auconvert.(u"eV^-1",lorentzian_pdf_au)) # Convert to eV⁻¹

    ## Plotting set up
    fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="Energy / eV", ylabel= "Distribution / eV⁻¹",limits=(nothing, nothing, nothing, nothing))

    lines!(ax, energies, lorentzian_pdf; color = colorscheme[2], linewidth = 2, label = "Gassian distribution\n Var = $(auconvert.(u"eV",width_au))")

    Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:top, halign=:left, margin=(5, 5, 5, 5), orientation=:vertical)

    return fig
end

center_au = austrip(0.0*u"eV") # Center of the Dirac delta function in a.u.

width_au = austrip(1*u"eV") # Width of the Dirac delta function in a.u.

plot_lorentzian(center_au, width_au)