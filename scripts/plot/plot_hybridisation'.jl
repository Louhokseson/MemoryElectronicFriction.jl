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
using NQCModels
using Unitful, UnitfulAtomic
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;



positions = range(0.5, 1.5, length=100)
positions_au = austrip.(positions*u"Å")
nstates = 10
bandmin = austrip(-3.0* u"eV")
bandmax = austrip(3.0* u"eV")
adsorbate_m = AndersonImpurityModels.BrandbygeAdsorbate()

bath = NQCModels.TrapezoidalRule(nstates, bandmin, bandmax)
#A′ =  AndersonImpurityFrictions.WeightedEHPDOS.A′_matrix(bath, adsorbate_m, position)
#HokseonReproduce.A′ak(bath,adsorbate_m, position)



dΔ_dr_au = HokseonReproduce.dΔ_dr.(positions_au, adsorbate_m)

dΔ_dr_evA = ustrip.(auconvert.(u"eV/Å", dΔ_dr_au))

fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
ax = MyAxis(fig[1,1], xlabel="Position / Å", ylabel= "dΔ/dr / eV/Å", limits=(nothing, nothing, nothing, nothing))

lines!(ax, positions, dΔ_dr_evA; color = colormap[1], linewidth = 2, label = "dΔ/dr")

fig

auconvert.(u"eV",AndersonImpurityFrictions.WeightedEHPDOS.A′_matrix(bath, adsorbate_m, positions_au[1]))

auconvert.(u"eV",HokseonReproduce.dΔ_dr.(positions_au[1], adsorbate_m))^ 2

width = auconvert.(u"eV",bath.bathstates[end] - bath.bathstates[1])