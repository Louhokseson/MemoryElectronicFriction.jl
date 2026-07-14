## Only for the main process 
using Distributed
using DrWatson
@quickactivate "HokseonReproduce" ## Activate project everywhere
import Pkg; Pkg.precompile() ## Precompile packages in master to speed up workers' precompilation
using HDF5
using DelimitedFiles
using HokseonAssistant
using CairoMakie
using HokseonPlots
using ColorSchemes
using NQCModels
using NQCDynamics
using NQCCalculators
using Colors
colorscheme = ColorScheme(parse.(Colorant, ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]));
colormap = HokseonPlots.NICECOLORS;
HokseonAssistant.julia_build_procs() 


# Load packages everywhere
@everywhere using HokseonReproduce
@everywhere using Unitful, UnitfulAtomic
@everywhere using NQCModels.QuantumModels
@everywhere using NQCModels


function buildSystemBath(params_dict::Dict{String, Any})
    @unpack  position, impuritymodel, temperature, energy = params_dict

    if impuritymodel == :ErpenbeckThossAdsorbate
        Γ = params_dict["Γ"]
        adsorbate_m = ErpenbeckThossAdsorbate(Γ=austrip(Γ*u"eV"))
    else
        adsorbate_m = eval(impuritymodel)()
    end


    energy_au = austrip.(energy*u"eV")
    temperature_au = austrip.(temperature*u"K")
    position_au = austrip.(position*u"Å")

    return adsorbate_m, position_au, energy_au, temperature_au
end

function NQCD_MarkovianFriction(params_dict::Dict{String, Any})
    @unpack  position, impuritymodel, temperature = params_dict

    temperature_au = austrip.(temperature*u"K")
    position_au = austrip.(position*u"Å")

    if impuritymodel == :ErpenbeckThossAdsorbate
        Γ = params_dict["Γ"]
        
        adsorbate_m = NQCModels.ErpenbeckThoss(;Γ=austrip(Γ*u"eV"))
        M = 1000
        bw = austrip(100u"eV")
        model = WideBandBath(adsorbate_m; step=(2bw)/M, bandmin=-bw, bandmax=bw)
        atoms = Atoms(1u"u")
        sim = Simulation{DiabaticMDEF}(atoms, model, friction_method=NQCCalculators.WideBandExact(model.ρ, 1/temperature_au))

        MarkovianFriction = [only(NQCCalculators.evaluate_friction(sim.cache, hcat(x))) for x in position_au]
        return MarkovianFriction
    end
end



params_list = dict_list(Dict{String, Any}(
    "impuritymodel" => [:ErpenbeckThossAdsorbate],
    "centre" => [0],
    "position" => [2.5],
    "temperature" => [2500, 500, 1],

    ## extra [] to make collect(...) as a whole a single parameter as a whole collect(0.05:0.01:0.1)
    "energy" => [collect(0.000001:0.001:30.0)],

    "Γ" => [0.8], # if impuritymodel is ErpenbeckThossAdsorbate, then Γ is used to compute V̄ₖ, which is the prefactor of the Lorentzian width Δ = 2π * Vₖ^2
))

# just make sure that params_list is a list with Dicts
if typeof(params_list) != Vector{Dict{String, Any}}
    params_list = [params_list]
end

fig = Figure(size=(HokseonPlots.RESOLUTION[1]*2, 3*HokseonPlots.RESOLUTION[2]), figure_padding=(1, 2, 1, 1), fonts=(;regular=projectdir("fonts", "MinionPro-Capt.otf")))
ax = MyAxis(fig[1,1], xlabel="ω / eV", ylabel= "Λ(ω) / u⋅ps⁻¹",limits=(0.0, nothing, nothing, nothing))

linestyles = [:solid, :solid, :dashdotdot]
colors = [:blue, :green, :red]
for (i,params_dict) in enumerate(params_list)

    MarkovianFriction_NQCD_au = NQCD_MarkovianFriction(params_dict)

    #MarkovianFriction_NQCD = ustrip.(MarkovianFriction_NQCD_au .* auconvert.(u"u", 1) ./ auconvert.(u"ps",1))

    local adsorbate_m, position_au, energy_au, temperature_au = buildSystemBath(params_dict)
    Λ_au = FrequencyLambda.Lambda.(energy_au, Ref(adsorbate_m), Ref(position_au), Ref(temperature_au))

    #γ_au = MarkovianLambda.widebandfriction(adsorbate_m, position_au, temperature_au)

    γ = ones(length(energy_au)) .* ustrip.(MarkovianFriction_NQCD_au[1] .* auconvert.(u"u", 1) ./ auconvert.(u"ps",1))

    ω_ev = ustrip.(auconvert.(u"eV", energy_au))

    Λ = ustrip.(Λ_au .* auconvert.(u"u", 1) ./ auconvert.(u"ps",1))


    lines!(ax, ω_ev, Λ; color=colors[i], linestyle = linestyles[i], linewidth=2, label = "T = $(params_dict["temperature"]) K" )
    lines!(ax, ω_ev, γ; color=colors[i], linestyle=:dash, linewidth=2 )
end

adsorbate_m, position_au, energy_au, temperature_au = buildSystemBath(params_list[1])

fermidirac = DistributionTools.FermiDirac(0.0, temperature_au)

(ω₁_min, ω₁_max) = FrequencyLambda.ω₁_effective_range(fermidirac, 0., C = 12.0)

h_au = HokseonReproduce.adsorbate_h(position_au, adsorbate_m)

h_ev = ustrip(h_au * auconvert.(u"eV", 1))

Δ_au = HokseonReproduce.Δ(position_au, adsorbate_m)

Δ_ev = ustrip(Δ_au * auconvert.(u"eV", 1))

#xline_band = abs(h_ev + ustrip(ω₁_max * auconvert.(u"eV", 1)))

xline_band = abs(h_ev - 0.0)

vlines!(ax, xline_band; linestyle=:dash, color=:black, linewidth=2, label="h(x) to 0.0 eV")

Legend(fig[1,1], ax, tellwidth=false, tellheight=false, valign=:top, halign=:right, margin=(5, 5, 5, 5), orientation=:vertical)


Label(fig[1,1], " $(params_list[1]["impuritymodel"]) \n dash: NQCD Markovian \n Position = $(params_list[1]["position"]) Å \n Γ = $(params_list[1]["Γ"]) eV \n h = $(round(h_ev, digits=4)) eV \n Δ = $(round(Δ_ev, digits=4)) eV"; tellwidth=false, tellheight=false, valign=:center, halign=:right, padding=(10,10,10,10),fontsize=16)

display(fig)