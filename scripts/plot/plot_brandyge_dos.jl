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
using Unitful, UnitfulAtomic
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;


function plot_brandbyge_dos(r = austrip(1.0*u"Å"))
    m = AndersonImpurityModels.BrandbygeAdsorbate()
    lorentzian = HokseonReproduce.DOS(r, m)

    # evaluate the DOS


    ω = range(-5, 5, length=1000)

    ω_au = austrip.(ω * u"eV") # Convert to atomic units
    dos_au = HokseonReproduce.PDF.(ω_au, lorentzian)

    dos = ustrip.(auconvert.(u"eV^-1",dos_au)) # Convert to eV



    ## Plotting set up
    fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="Energy / eV", ylabel= "Density of States / eV⁻¹",limits=(nothing, nothing, nothing, nothing))


    lines!(ax, ω, dos; color = colormap[1], linewidth = 2, label = "Brandbyge \n impurity at $(auconvert(u"Å",r))")
    band!(ax, ω, zeros(length(ω)), dos; color=(colormap[1],0.3))
    Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:top, halign=:left, margin=(5, 5, 5, 5), orientation=:vertical)
    return fig
end

r = austrip(0.5*u"Å")

plot_brandbyge_dos(r)