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


fermi_level = 0.0  # eV

temperature = 5500  # K
temperature_au = austrip(temperature * u"K")
fermi_level_au = austrip(fermi_level * u"eV")




fermidirac = DistributionTools.FermiDirac(fermi_level_au, temperature_au)

Fermi_Dirac_bracket(ω::Real, ω₁::Real) = (HokseonReproduce.PDF.(ω + ω₁,fermidirac) - HokseonReproduce.PDF.(ω₁,fermidirac))


function Fermi_Dirac_bracket(ω::Real, ω₁::AbstractVector)
    if Threads.nthreads() == 1
        # Single-thread → broadcast
        return Fermi_Dirac_bracket.(Ref(ω), ω₁)
    else
        # Multi-threaded loop
        Fermi_Dirac_bracket_au_vec = Vector{Float64}(undef, length(ω₁))
        Threads.@threads for i in eachindex(ω₁)
            Fermi_Dirac_bracket_au_vec[i] = Fermi_Dirac_bracket(ω, ω₁[i])
        end
        return Fermi_Dirac_bracket_au_vec
    end
end



function plot_Fermi_Dirac_bracket(ω::Real, ω₁::AbstractVector)

    """
    ω₁ : Energy vector ω₁ in a.u. from eV
    ω : Energy value ω in a.u. from eV
    """

    Fermi_Dirac_bracket_values_au = Fermi_Dirac_bracket(ω, ω₁)

    ω₁_range_min, ω₁_range_max = ω₁_range(fermidirac, ω)


    ## Plotting set up
    fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
    ax = MyAxis(fig[1,1], xlabel="ω₁ / eV", ylabel= "n(ω+ω₁) - n(ω₁) / eV⁻¹",limits=(-5, 5, nothing, nothing))


    ω₁_ev = ω₁ * 27.211386245988   # if ω₁ already represents an energy in Hartree

    Fermi_Dirac_bracket_values_ev = Fermi_Dirac_bracket_values_au * (1/27.211386245988)  # converting from a.u. to eV⁻¹

    lines!(ax, ω₁_ev, Fermi_Dirac_bracket_values_ev, color = colormap[2], linewidth = 2, label = "ω = $(round(ω * 27.211386245988, digits=2)) eV, T = $temperature K")


    scatter!(ax, [ω₁_range_min * 27.211386245988, ω₁_range_max * 27.211386245988],
             [0.0, 0.0], color = :red, markersize = 8,
             label = "non-zero interval")
    
    Legend(fig[1,1], ax, titleposition = :top, tellwidth=false, tellheight=false, valign=:bottom, halign=:left, margin=(5, 5, 5, 5), orientation=:vertical)

    #display(fig)

    return fig
end

function ω₁_range(fermidirac, ω::Real; C::Real = 6)
    if ω >= 0
        # Positive ω case
        ω₁_min = fermidirac.ϵf - ω - C * fermidirac.T
        ω₁_max = fermidirac.ϵf + C * fermidirac.T
    else
        # Negative ω case
        ω₁_min = fermidirac.ϵf - C * fermidirac.T
        ω₁_max = fermidirac.ϵf - ω + C * fermidirac.T
    end

    return (ω₁_min, ω₁_max)
end


ω = austrip(0.5 * u"eV")
ω₁ = austrip.(collect(-5:0.01:5) .* u"eV")

fig = plot_Fermi_Dirac_bracket(ω, ω₁)



